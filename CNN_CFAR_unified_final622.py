# -*- coding: utf-8 -*-
"""
CNN_CFAR_unified.py

Adapted U-Net/CNN-inversion + CFAR baseline for AWPL-04-26-1519.

This file is NOT an exact reproduction of Schenone et al., IEEE TAP 2025,
"An Antenna Array Diagnosis Approach Based on CNN Inversion and CFAR Detection",
because the original method requires aperture-field labels on a dense aperture grid
and a reference-array far-field/aperture-field pair. The current SBHA/AEP dataset
contains element-level complex excitations and AEP far-field responses, but not
96x96 aperture tangential-field maps.

Therefore this script implements a fair AEP-dataset adaptation of that idea:
    complex far-field feature -> CNN/U-Net-style inversion -> element-level
    differential excitation image -> cell-averaging CFAR fault map.
It uses the same Config, data generator, SNR, labels, and metrics from
unified_pipeline_final622.py (or unified_pipeline.py if renamed).

Standalone usage:
    python CNN_CFAR_unified.py --aep_mat_path AEP_Matrix.mat --save_dir ./out_cnn_cfar
    python CNN_CFAR_unified.py --quick --cpu

Integration usage:
    from CNN_CFAR_unified import train_cnn_cfar, evaluate_cnn_cfar_method
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the unified pipeline under either likely filename.
try:
    up = importlib.import_module("unified_pipeline_final622")
except Exception:
    try:
        up = importlib.import_module("unified_pipeline_final621")
    except Exception:
        up = importlib.import_module("unified_pipeline")


class TinyUNetInversion(nn.Module):
    """Small U-Net-style inversion network for a 5x5 element image.

    Input:  [B, 4K] real-valued complex far-field feature.
    Output: [B, 2, Ny, Nx] differential excitation map, channels are
            Re(w - 1) and Im(w - 1).

    The first FC layer plays the same role as the FC size-conversion layer in
    the TAP 2025 U-Net-CFAR paper, converting far-field samples to an image-like
    latent representation. The encoder/decoder then performs local 2-D feature
    refinement before cropping to the 5x5 element grid.
    """
    def __init__(self, input_dim: int, grid_size: int = 5, base_ch: int = 32, latent_hw: int = 8) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.latent_hw = int(latent_hw)
        self.base_ch = int(base_ch)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, base_ch * latent_hw * latent_hw),
            nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1), nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1), nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
        )
        self.up = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(base_ch, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        z = self.fc(x).view(b, self.base_ch, self.latent_hw, self.latent_hw)
        e1 = self.enc1(z)
        e2 = self.down(e1)
        u = self.up(e2)
        if u.shape[-2:] != e1.shape[-2:]:
            u = F.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(torch.cat([u, e1], dim=1))
        y = self.out(y)
        # Center crop from latent_hw x latent_hw to grid_size x grid_size.
        h, w = y.shape[-2:]
        top = (h - self.grid_size) // 2
        left = (w - self.grid_size) // 2
        return y[:, :, top:top + self.grid_size, left:left + self.grid_size]


def labels_to_delta_map(batch_y: torch.Tensor, cfg: Any, grid_size: int) -> torch.Tensor:
    true_amp = batch_y[:, :cfg.N]
    true_cos = batch_y[:, cfg.N:2 * cfg.N]
    true_sin = batch_y[:, 2 * cfg.N:3 * cfg.N]
    w_re = true_amp * true_cos
    w_im = true_amp * true_sin
    delta_re = w_re - 1.0
    delta_im = w_im
    target = torch.stack([delta_re, delta_im], dim=1)  # [B, 2, N]
    return target.view(batch_y.size(0), 2, grid_size, grid_size)


def delta_map_to_amp_phase(delta_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # delta_map: [B, 2, G, G]
    delta_re = delta_map[:, 0].reshape(delta_map.shape[0], -1)
    delta_im = delta_map[:, 1].reshape(delta_map.shape[0], -1)
    w = (1.0 + delta_re) + 1j * delta_im
    pred_amp = np.abs(w).astype(np.float32)
    pred_phase_deg = np.rad2deg(np.angle(w)).astype(np.float32)
    return pred_amp, pred_phase_deg


def ca_cfar_2d(score: np.ndarray, gamma: float = 0.2, train_radius: int = 1, exclude_cut: bool = True) -> np.ndarray:
    """Cell-averaging CFAR on a batch of element-score maps.

    score: [B, G, G], nonnegative anomaly score.
    gamma: desired false alarm probability in the CA-CFAR threshold-factor formula.
    """
    score = np.asarray(score, dtype=np.float64)
    b, g1, g2 = score.shape
    pred = np.zeros_like(score, dtype=np.int32)
    pad = train_radius
    for n in range(b):
        padded = np.pad(score[n], pad_width=pad, mode="edge")
        for i in range(g1):
            for j in range(g2):
                win = padded[i:i + 2 * pad + 1, j:j + 2 * pad + 1].copy()
                if exclude_cut:
                    win[pad, pad] = np.nan
                cells = win[np.isfinite(win)].reshape(-1)
                if cells.size == 0:
                    noise = np.median(score[n])
                    t = 1
                else:
                    # Robust average: prevents one strong adjacent fault from setting an impossible threshold.
                    noise = float(np.mean(cells))
                    t = int(cells.size)
                beta = t * (float(gamma) ** (-1.0 / max(t, 1)) - 1.0)
                threshold = beta * max(noise, 1e-12)
                pred[n, i, j] = int(score[n, i, j] >= threshold)
    return pred.reshape(b, -1)


def train_cnn_cfar(cfg: Any, input_dim: int, A_Etheta: np.ndarray, A_Ephi: np.ndarray, K: int,
                   device: torch.device, out_dir: Path, tag: str = "cnn_cfar") -> Tuple[nn.Module, Dict[str, Any]]:
    grid = int(round(math.sqrt(cfg.N)))
    if grid * grid != cfg.N:
        raise ValueError("CNN-CFAR adaptation expects a square element grid; cfg.N must be a perfect square.")
    up.seed_everything(cfg.seed, cfg.deterministic_cudnn)
    model = TinyUNetInversion(input_dim=input_dim, grid_size=grid).to(device)

    train_ds = up.SyntheticPhasedArrayDataset(
        cfg, cfg.train_samples, base_seed=cfg.seed + 100_000,
        A_Etheta=A_Etheta, A_Ephi=A_Ephi, K=K, scenario="mixed",
        sampler=cfg.train_sampler, target_snr_db=cfg.train_snr_db, noise_kind="gaussian",
        unlabeled_fraction=0.0,  # this baseline is fully supervised, like the original U-Net training.
    )
    val_ds = up.SyntheticPhasedArrayDataset(
        cfg, cfg.val_samples, base_seed=cfg.seed + 200_000,
        A_Etheta=A_Etheta, A_Ephi=A_Ephi, K=K, scenario="mixed",
        sampler=cfg.train_sampler, target_snr_db=cfg.train_snr_db, noise_kind="gaussian",
        unlabeled_fraction=0.0,
    )
    train_loader = up.build_loader(cfg, train_ds, shuffle=True, device=device, seed=cfg.seed + 300_000)
    val_loader = up.build_loader(cfg, val_ds, shuffle=False, device=device, seed=cfg.seed + 400_000)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best_state = None
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    hist: List[Dict[str, float]] = []
    t0 = time.perf_counter()
    print(f"\n=== Training {tag} (adapted U-Net/CNN inversion + CFAR) ===")
    for ep in range(1, cfg.epochs + 1):
        model.train()
        sum_loss, count = 0.0, 0
        for batch in train_loader:
            bx, by = batch[0].to(device, dtype=torch.float32), batch[1].to(device, dtype=torch.float32)
            target = labels_to_delta_map(by, cfg, grid)
            pred = model(bx)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            opt.step()
            sum_loss += float(loss.detach().cpu()) * bx.size(0)
            count += bx.size(0)
        train_loss = sum_loss / max(count, 1)
        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                bx, by = batch[0].to(device, dtype=torch.float32), batch[1].to(device, dtype=torch.float32)
                target = labels_to_delta_map(by, cfg, grid)
                pred = model(bx)
                loss = F.mse_loss(pred, target)
                val_sum += float(loss.detach().cpu()) * bx.size(0)
                val_count += bx.size(0)
        val_loss = val_sum / max(val_count, 1)
        scheduler.step()
        hist.append({"epoch": ep, "train_mse": train_loss, "val_mse": val_loss})
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 10 == 0 or ep == cfg.epochs:
            print(f"{tag} epoch {ep:4d} | train {train_loss:.6e} | val {val_loss:.6e}")
        if bad >= cfg.patience:
            print(f"{tag}: early stopping at epoch {ep}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(out_dir / f"history_{tag}.csv", index=False, encoding="utf-8-sig")
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_time_sec": train_time,
        "config": asdict(cfg),
        "note": "Adapted U-Net/CNN inversion + CFAR baseline; output is element-level delta excitation map, not dense aperture field.",
    }, out_dir / f"best_checkpoint_{tag}.pt")
    return model, {"tag": tag, "best_epoch": best_epoch, "best_val_loss": best_val, "train_time_sec": train_time,
                   "n_params": int(sum(p.numel() for p in model.parameters()))}


@torch.no_grad()
def evaluate_cnn_cfar_method(model: nn.Module, cfg: Any, A_Etheta: np.ndarray, A_Ephi: np.ndarray, K: int,
                             device: torch.device, target_snr_db: Optional[float] = None,
                             gamma: float = 0.2, base_seed: Optional[int] = None,
                             eval_samples: Optional[int] = None) -> Dict[str, Any]:
    grid = int(round(math.sqrt(cfg.N)))
    n = cfg.eval_samples if eval_samples is None else int(eval_samples)
    snr = cfg.nominal_eval_snr_db if target_snr_db is None else float(target_snr_db)
    seed = cfg.seed + 570_000 if base_seed is None else int(base_seed)
    ds = up.SyntheticPhasedArrayDataset(
        cfg, n, base_seed=seed, A_Etheta=A_Etheta, A_Ephi=A_Ephi, K=K,
        scenario="mixed", sampler=cfg.eval_sampler, target_snr_db=snr, noise_kind="gaussian",
        unlabeled_fraction=0.0,
    )
    loader = up.build_loader(cfg, ds, shuffle=False, device=device, seed=seed + 77)
    model.eval()
    pred_maps, true_amp, true_phase, true_bin = [], [], [], []
    t0 = time.perf_counter()
    for batch in loader:
        bx, by = batch[0].to(device, dtype=torch.float32), batch[1]
        pm = model(bx).detach().cpu().numpy()
        pred_maps.append(pm)
        y = by.numpy()
        true_amp.append(y[:, :cfg.N])
        true_phase.append(np.rad2deg(np.arctan2(y[:, 2 * cfg.N:3 * cfg.N], y[:, cfg.N:2 * cfg.N])))
        true_bin.append(y[:, 3 * cfg.N:].astype(np.int32))
    infer_ms = 1000.0 * (time.perf_counter() - t0) / max(n, 1)
    pred_map = np.concatenate(pred_maps, axis=0)
    true_amp_np = np.concatenate(true_amp, axis=0)
    true_phase_np = np.concatenate(true_phase, axis=0)
    true_bin_np = np.concatenate(true_bin, axis=0)
    score = np.sqrt(pred_map[:, 0] ** 2 + pred_map[:, 1] ** 2)
    pred_bin = ca_cfar_2d(score, gamma=gamma, train_radius=1, exclude_cut=True)
    pred_amp, pred_phase = delta_map_to_amp_phase(pred_map)
    metrics = up.compute_metrics_from_arrays(
        true_bin_np, pred_bin, true_amp_np, true_phase_np, pred_amp, pred_phase,
        np.full(true_bin_np.shape[0], np.nan), cfg,
    )
    metrics["Infer (ms/sample)"] = infer_ms
    metrics["gamma"] = gamma
    return {"metrics": metrics, "pred_binary": pred_bin, "score": score,
            "pred_amp": pred_amp, "pred_phase_deg": pred_phase}


def main() -> None:
    ap = argparse.ArgumentParser(description="Adapted CNN inversion + CFAR baseline for the unified AWPL benchmark")
    ap.add_argument("--aep_mat_path", type=str, default="AEP_Matrix.mat")
    ap.add_argument("--save_dir", type=str, default="./out_cnn_cfar")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.2)
    args = ap.parse_args()

    cfg = up.Config()
    cfg.aep_mat_path = args.aep_mat_path
    cfg.save_dir = args.save_dir
    if args.quick:
        cfg.train_samples, cfg.val_samples, cfg.eval_samples = 64, 32, 32
        cfg.epochs, cfg.patience, cfg.batch_size = 1, 1, 32
        cfg.num_workers, cfg.persistent_workers = 0, False
    out_dir = Path(cfg.save_dir)
    up.ensure_dir(out_dir)
    up.seed_everything(cfg.seed, cfg.deterministic_cudnn)
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.use_gpu_if_available and not args.cpu) else "cpu")
    print("Using device:", device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(getattr(cfg, "cpu_num_threads", 4))))
    A_Etheta, A_Ephi, K = up.load_aep_matrix(cfg.aep_mat_path, cfg)
    input_dim = 4 * K
    print(f"Loaded AEP matrix: K={K}, N={cfg.N}, input_dim={input_dim}")
    model, summary = train_cnn_cfar(cfg, input_dim, A_Etheta, A_Ephi, K, device, out_dir)
    result = evaluate_cnn_cfar_method(model, cfg, A_Etheta, A_Ephi, K, device, gamma=args.gamma)
    m = result["metrics"]
    row = {"best_epoch": summary["best_epoch"], "best_val_loss": summary["best_val_loss"],
           "train_time_sec": summary["train_time_sec"], "gamma": args.gamma,
           "S-E F1 (%)": m["f1_pct"], "EMA (%)": m["ema_pct"],
           "Precision (%)": m["precision_pct"], "Recall (%)": m["recall_pct"],
           "Infer (ms/sample)": m["Infer (ms/sample)"]}
    pd.DataFrame([row]).to_csv(out_dir / "summary_cnn_cfar.csv", index=False, encoding="utf-8-sig")
    print(f"Adapted CNN-CFAR @ {cfg.nominal_eval_snr_db} dB: F1={m['f1_pct']:.2f}%, EMA={m['ema_pct']:.2f}%")
    print("Saved to:", out_dir / "best_checkpoint_cnn_cfar.pt")


if __name__ == "__main__":
    main()
