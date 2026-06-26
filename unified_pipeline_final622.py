from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 0. Configuration
# =============================================================================

NumberOrRange = Union[float, Tuple[float, float], List[float]]


@dataclass
class Config:
    # Paths
    aep_mat_path: str = "AEP_Matrix.mat"
    save_dir: str = "PI_complete_outputsfinal622"

    # Array / task
    N: int = 25
    amp_fault_threshold: float = 0.8
    phase_fault_threshold_deg: float = 10.0
    amp_max_value: float = 1.05
    fault_probability_threshold: float = 0.5

    train_samples: int = 200_000
    val_samples: int = 50_000
    eval_samples: int = 10_000

    # Data generation
    healthy_sample_prob: float = 0.15
    max_injected_elements: int = 6
    healthy_amp_min: float = 0.95
    healthy_amp_max: float = 1.05
    healthy_phase_min_deg: float = -10.0
    healthy_phase_max_deg: float = 10.0
    impaired_amp_min: float = 0.0
    impaired_amp_max: float = 1.05
    impaired_phase_min_deg: float = -180.0
    impaired_phase_max_deg: float = 180.0

    train_sampler: str = "stratified"  # "stratified" or "uniform"
    eval_sampler: str = "uniform"
    severe_amp_focus_threshold: float = 0.5
    phase_center_jitter_deg: float = 8.0
    phase_fault_centers_deg: Tuple[float, ...] = (
        -150.0, -120.0, -90.0, -60.0, -30.0,
        30.0, 60.0, 90.0, 120.0, 150.0, 180.0,
    )

    # Noise model. Training uses per-sample SNR ~ U[0, 40] dB by default.
    train_snr_db: Tuple[float, float] = (0.0, 40.0)
    nominal_eval_snr_db: float = 25.0
    snr_sweep_db: Tuple[float, ...] = (60.0, 40.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0)
    noise_kinds: Tuple[str, ...] = ("gaussian", "uniform", "laplacian")

    # Training
    seed: int = 42
    epochs: int = 500
    batch_size: int = 128
    lr: float = 8e-4
    weight_decay: float = 1e-4
    patience: int = 30
    grad_clip_norm: float = 5.0

    # Numerical epsilons
    eps_loss: float = 1e-8
    eps_phase_norm: float = 1e-6

    # Loss weights
    lambda_amp: float = 2.5
    lambda_phase_trig: float = 1.0
    lambda_phase_angle: float = 1.0
    lambda_phase_norm: float = 0.01
    # Direct complex-excitation consistency couples amplitude and phase in the
    # same variable that enters the AEP forward operator, improving inversion
    # accuracy relative to independent amp/cos/sin losses.
    lambda_complex_w: float = 0.50
    lambda_bce: float = 2.0

    lambda_phys: float = 20.0


    unlabeled_fraction: float = 0.65

    physics_anchor_clean: bool = True

    phys_warmup_epochs: int = 30

    physics_on_labeled: bool = True


    tta_enable: bool = True
    tta_iters: int = 160
    tta_lr: float = 1.0e-2
    # Optional curriculum-free trust region: clamp amplitude to [0, amp_max].
    tta_amp_clamp: bool = True
    # After TTPR, the final fault probability can use the refined physical
    # amplitude/phase in addition to the learned fault head. This makes the
    # inference-time AEP refinement visible in the actual fault map, not only in
    # the reported amplitude/phase MAE.
    tta_decision_mode: str = "fused_mean"  # head_only | refined_reg | fused_max | fused_mean | fused_and
    tta_head_weight: float = 0.70
    fault_score_tau_amp: float = 0.025
    fault_score_tau_phase_deg: float = 2.5

    # Imbalance / fault-aware weights
    bce_pos_weight: float = 3.0
    amp_fault_weight: float = 4.0
    phase_fault_weight: float = 3.0
    severe_amp_extra_weight: float = 2.0
    smooth_l1_beta_amp: float = 0.05
    smooth_l1_beta_phase: float = 0.05

    # Weak-amplitude phase mask, directly addressing Reviewer 2 Comment 3.
    # Phase-related losses and phase MAE statistics are excluded for a < threshold.
    phase_amp_mask_threshold: float = 0.2

    # AEP mismatch levels: (relative amplitude std, phase std in degrees)
    aep_mismatch_levels: Tuple[Tuple[float, float], ...] = (
        (0.00, 0.0),
        (0.01, 1.0),
        (0.03, 3.0),
        (0.05, 5.0),
        (0.10, 10.0),
    )

    # Threshold sensitivity grid
    amp_threshold_grid: Tuple[float, ...] = (0.75, 0.80, 0.85)
    phase_threshold_grid_deg: Tuple[float, ...] = (8.0, 10.0, 12.0)

    # Variants
    train_no_bce_ablation: bool = True

    # ----- Baseline architectures and CS settings (unified fair benchmark) -----
    # ANN/DNN MLP widths (kept identical to the original baseline scripts).
    ann_hidden: int = 384
    dnn_hidden: int = 512
    dnn_depth: int = 5
    dnn_dropout: float = 0.1
    # Adapted-cGAN.
    gan_noise_dim: int = 32
    cgan_hidden: int = 384
    gan_adv_lambda: float = 0.2
    gan_l1_lambda: float = 0.5
    cgan_infer_avg_seeds: int = 8
    # Whether to include the learning-based baselines / CS baselines in main().
    run_learning_baselines: bool = True
    run_cs_baselines: bool = True
    run_cnn_cfar_baseline: bool = True  # SOTA: adapted CNN-inversion + CFAR (TAP 2025)
    cnn_cfar_gamma: float = 0.2  # CFAR desired false-alarm probability (paper's best)
    # CS baselines. M_obs_list controls the observation-point settings reported:
    #   50  -> conventional sparse setting; K (=722) -> equal-observation setting.
    cs_methods: Tuple[str, ...] = ("OMP", "Lasso", "FISTA")
    cs_M_obs_list: Tuple[int, ...] = (50, 722)
    cs_omp_k: int = 8          # >= max injected faults (6); sparse-deviation prior
    cs_lasso_alpha: float = 1e-3
    lasso_max_iter: int = 50_000   # raised from 5000 to curb ConvergenceWarning
    lasso_tol: float = 1e-3        # slightly relaxed tolerance for the weakly-sparse system
    fista_max_iter: int = 500
    fista_tol: float = 1e-5

    # Runtime
    use_gpu_if_available: bool = True
    num_workers: int = min(8, max(1, os.cpu_count() or 1))
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True
    deterministic_cudnn: bool = True
    cpu_num_threads: int = 4


# =============================================================================
# 1. Utilities
# =============================================================================


def seed_everything(seed: int, deterministic_cudnn: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic_cudnn
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_aep_matrix(mat_path: str, cfg: Config) -> Tuple[np.ndarray, np.ndarray, int]:
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Cannot find AEP matrix file: {mat_path}")
    mat_data = sio.loadmat(mat_path)
    if "A_Etheta" not in mat_data or "A_Ephi" not in mat_data:
        raise KeyError("AEP_Matrix.mat must contain A_Etheta and A_Ephi")
    A_Etheta = np.asarray(mat_data["A_Etheta"], dtype=np.complex64)
    A_Ephi = np.asarray(mat_data["A_Ephi"], dtype=np.complex64)
    if A_Etheta.shape != A_Ephi.shape:
        raise ValueError(f"A_Etheta shape={A_Etheta.shape} != A_Ephi shape={A_Ephi.shape}")
    K, N_from_mat = A_Etheta.shape
    if N_from_mat != cfg.N:
        raise ValueError(f"AEP matrix N={N_from_mat} does not match cfg.N={cfg.N}")
    return A_Etheta, A_Ephi, K


def choose_snr_db(target_snr_db: NumberOrRange, rng: np.random.Generator) -> float:
    if isinstance(target_snr_db, (tuple, list)):
        if len(target_snr_db) != 2:
            raise ValueError("target_snr_db range must have length 2")
        lo, hi = float(target_snr_db[0]), float(target_snr_db[1])
        return float(rng.uniform(lo, hi))
    return float(target_snr_db)


def wrap_phase_deg(delta_deg: np.ndarray) -> np.ndarray:
    return (delta_deg + 180.0) % 360.0 - 180.0


def wrap_phase_rad(delta_rad: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(delta_rad), torch.cos(delta_rad))


def build_feature(E_theta: np.ndarray, E_phi: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [E_theta.real, E_theta.imag, E_phi.real, E_phi.imag]
    ).astype(np.float32)


def diagnose_rule_np(
    amp: np.ndarray,
    phase_deg: np.ndarray,
    amp_threshold: float,
    phase_threshold_deg: float,
) -> np.ndarray:
    return ((amp < amp_threshold) | (np.abs(phase_deg) > phase_threshold_deg)).astype(np.int32)


def soft_fault_probability_torch(
    amp: torch.Tensor,
    cos_v: torch.Tensor,
    sin_v: torch.Tensor,
    cfg: "Config",
) -> torch.Tensor:

    phase_abs_deg = torch.abs(torch.rad2deg(torch.atan2(sin_v, cos_v)))
    amp_score = torch.sigmoid((cfg.amp_fault_threshold - amp) / cfg.fault_score_tau_amp)
    phase_score = torch.sigmoid((phase_abs_deg - cfg.phase_fault_threshold_deg) / cfg.fault_score_tau_phase_deg)
    return 1.0 - (1.0 - amp_score) * (1.0 - phase_score)


def fuse_fault_probability(
    head_prob: torch.Tensor,
    amp: torch.Tensor,
    cos_v: torch.Tensor,
    sin_v: torch.Tensor,
    cfg: "Config",
) -> torch.Tensor:
    reg_prob = soft_fault_probability_torch(amp, cos_v, sin_v, cfg)
    mode = str(cfg.tta_decision_mode).lower()
    if mode == "head_only":
        return head_prob
    if mode == "refined_reg":
        return reg_prob
    if mode == "fused_max":
        return torch.maximum(head_prob, reg_prob)
    if mode == "fused_and":
        return head_prob * reg_prob
    if mode == "fused_mean":
        w = float(cfg.tta_head_weight)
        return torch.clamp(w * head_prob + (1.0 - w) * reg_prob, 0.0, 1.0)
    raise ValueError(f"Unknown tta_decision_mode: {cfg.tta_decision_mode}")


def draw_real_noise(kind: str, rng: np.random.Generator, shape: Union[int, Tuple[int, ...]], sigma: float) -> np.ndarray:
    kind = kind.lower()
    if kind == "gaussian":
        return sigma * rng.standard_normal(shape)
    if kind == "uniform":
        a = sigma * math.sqrt(3.0)  # Var(U[-a,a]) = sigma^2
        return rng.uniform(-a, a, shape)
    if kind == "laplacian":
        b = sigma / math.sqrt(2.0)  # Var(Laplace(0,b)) = sigma^2
        return rng.laplace(0.0, b, shape)
    raise ValueError(f"Unknown noise kind: {kind}")


def perturb_aep(
    A: np.ndarray,
    rng: np.random.Generator,
    amp_sigma: float,
    phase_sigma_deg: float,
) -> np.ndarray:
    if amp_sigma == 0.0 and phase_sigma_deg == 0.0:
        return A.copy()
    amp_factor = 1.0 + rng.normal(0.0, amp_sigma, A.shape)
    phase_factor = np.exp(1j * np.deg2rad(rng.normal(0.0, phase_sigma_deg, A.shape)))
    return (A.astype(np.complex128) * amp_factor * phase_factor).astype(np.complex64)


# =============================================================================
# 2. Synthetic data generation
# =============================================================================


def sample_faulty_amplitude(rng: np.random.Generator, cfg: Config, sampler: str) -> float:
    sampler = sampler.lower()
    if sampler == "uniform":
        return float(rng.uniform(cfg.impaired_amp_min, cfg.impaired_amp_max))
    if sampler == "stratified":
        p = rng.random()
        if p < 0.40:
            return float(rng.uniform(0.0, 0.4))
        if p < 0.80:
            return float(rng.uniform(0.4, cfg.amp_fault_threshold))
        return float(rng.uniform(cfg.amp_fault_threshold, cfg.impaired_amp_max))
    raise ValueError(f"Unknown sampler: {sampler}")


def sample_faulty_phase_deg(rng: np.random.Generator, cfg: Config, sampler: str) -> float:
    sampler = sampler.lower()
    if sampler == "uniform":
        val = float(rng.uniform(cfg.impaired_phase_min_deg, cfg.impaired_phase_max_deg))
    elif sampler == "stratified":
        if rng.random() < 0.75:
            center = float(rng.choice(np.array(cfg.phase_fault_centers_deg, dtype=np.float32)))
            val = center + float(rng.normal(0.0, cfg.phase_center_jitter_deg))
        else:
            val = float(rng.uniform(cfg.impaired_phase_min_deg, cfg.impaired_phase_max_deg))
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
    while val > 180.0:
        val -= 360.0
    while val < -180.0:
        val += 360.0
    return float(val)


def generate_sample(
    cfg: Config,
    rng: np.random.Generator,
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    scenario: str = "mixed",
    sampler: str = "uniform",
    target_snr_db: NumberOrRange = 25.0,
    noise_kind: str = "gaussian",
    return_meta: bool = False,
) -> Dict[str, Any]:
    scenario = scenario.lower()
    if scenario not in {"amp_only", "phase_only", "mixed"}:
        raise ValueError(f"Unknown scenario: {scenario}")

    amp_true = rng.uniform(cfg.healthy_amp_min, cfg.healthy_amp_max, cfg.N).astype(np.float32)
    phase_true_deg = rng.uniform(cfg.healthy_phase_min_deg, cfg.healthy_phase_max_deg, cfg.N).astype(np.float32)
    injected_indices = np.array([], dtype=np.int64)

    if rng.random() > cfg.healthy_sample_prob:
        num_injected = int(rng.integers(1, cfg.max_injected_elements + 1))
        injected_indices = rng.choice(cfg.N, size=num_injected, replace=False)
        for idx in injected_indices:
            if scenario == "amp_only":
                amp_true[idx] = sample_faulty_amplitude(rng, cfg, sampler)
            elif scenario == "phase_only":
                phase_true_deg[idx] = sample_faulty_phase_deg(rng, cfg, sampler)
            else:  # mixed
                fault_type = rng.random()
                if fault_type < 1.0 / 3.0:
                    amp_true[idx] = sample_faulty_amplitude(rng, cfg, sampler)
                elif fault_type < 2.0 / 3.0:
                    phase_true_deg[idx] = sample_faulty_phase_deg(rng, cfg, sampler)
                else:
                    amp_true[idx] = sample_faulty_amplitude(rng, cfg, sampler)
                    phase_true_deg[idx] = sample_faulty_phase_deg(rng, cfg, sampler)

    phase_true_rad = np.deg2rad(phase_true_deg).astype(np.float32)
    w_true = (amp_true * np.exp(1j * phase_true_rad)).astype(np.complex64)
    E_theta_clean = A_Etheta @ w_true
    E_phi_clean = A_Ephi @ w_true
    clean_feature = build_feature(E_theta_clean, E_phi_clean)

    snr_db = choose_snr_db(target_snr_db, rng)
    snr_linear = 10.0 ** (snr_db / 10.0)
    signal_power = float(np.mean(clean_feature.astype(np.float64) ** 2))
    noise_sigma = math.sqrt(signal_power / (snr_linear + 1e-12))
    noise = draw_real_noise(noise_kind, rng, clean_feature.shape, noise_sigma).astype(np.float32)
    feature = (clean_feature + noise).astype(np.float32)

    fault_label = diagnose_rule_np(
        amp_true,
        phase_true_deg,
        cfg.amp_fault_threshold,
        cfg.phase_fault_threshold_deg,
    ).astype(np.float32)
    label = np.concatenate(
        [
            amp_true.astype(np.float32),
            np.cos(phase_true_rad).astype(np.float32),
            np.sin(phase_true_rad).astype(np.float32),
            fault_label.astype(np.float32),
        ]
    ).astype(np.float32)

    out: Dict[str, Any] = {"feature": feature, "label": label, "clean_feature": clean_feature.astype(np.float32)}
    if return_meta:
        out.update(
            {
                "amp_true": amp_true,
                "phase_true_deg": phase_true_deg,
                "fault_binary": fault_label.astype(np.int32),
                "injected_indices": injected_indices,
                "target_snr_db": snr_db,
                "noise_sigma": noise_sigma,
                "noise_kind": noise_kind,
                "scenario": scenario,
                "sampler": sampler,
            }
        )
    return out


class SyntheticPhasedArrayDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        num_samples: int,
        base_seed: int,
        A_Etheta: np.ndarray,
        A_Ephi: np.ndarray,
        K: int,
        scenario: str,
        sampler: str,
        target_snr_db: NumberOrRange,
        noise_kind: str = "gaussian",
        unlabeled_fraction: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.num_samples = int(num_samples)
        self.base_seed = int(base_seed)
        self.A_Etheta = A_Etheta
        self.A_Ephi = A_Ephi
        self.K = int(K)
        self.scenario = scenario
        self.sampler = sampler
        self.target_snr_db = target_snr_db
        self.noise_kind = noise_kind
        self.unlabeled_fraction = float(unlabeled_fraction)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.base_seed + int(idx))
        s = generate_sample(
            self.cfg,
            rng,
            self.A_Etheta,
            self.A_Ephi,
            self.K,
            scenario=self.scenario,
            sampler=self.sampler,
            target_snr_db=self.target_snr_db,
            noise_kind=self.noise_kind,
            return_meta=False,
        )
        # Deterministic labeled/unlabeled split (reproducible, independent of the
        # data-generation RNG so the split is identical across model variants).
        if self.unlabeled_fraction > 0.0:
            split_rng = np.random.default_rng(self.base_seed * 2_654_435_761 + int(idx) + 777)
            is_labeled = 1.0 if split_rng.random() >= self.unlabeled_fraction else 0.0
        else:
            is_labeled = 1.0
        return (
            torch.from_numpy(s["feature"]),
            torch.from_numpy(s["label"]),
            torch.from_numpy(s["clean_feature"]),
            torch.tensor(is_labeled, dtype=torch.float32),
        )


def build_loader(cfg: Config, dataset: Dataset, shuffle: bool, device: torch.device, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    kwargs: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and device.type == "cuda"),
        generator=g,
    )
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = cfg.persistent_workers
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(**kwargs)


# =============================================================================
# 3. Model: trainable 1D-ResNet backbone + frozen AEP physics module
# =============================================================================


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class AEPForwardOperator(nn.Module):
    """Frozen AEP-based electromagnetic forward-model module.

    The module implements the sampled AEP superposition equation E = A w at the
    observation points. It has zero trainable parameters; A_theta and A_phi are
    registered as persistent buffers and therefore move with the model and are
    stored in checkpoints.
    """

    def __init__(self, A_Etheta_np: np.ndarray, A_Ephi_np: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("A_theta", torch.tensor(A_Etheta_np, dtype=torch.complex64), persistent=True)
        self.register_buffer("A_phi", torch.tensor(A_Ephi_np, dtype=torch.complex64), persistent=True)

    def forward(self, pred_amp: torch.Tensor, pred_cos: torch.Tensor, pred_sin: torch.Tensor) -> torch.Tensor:
        w_pred = torch.complex(pred_amp * pred_cos, pred_amp * pred_sin)  # [B, N]
        E_theta_pred = torch.matmul(w_pred, self.A_theta.T)  # [B, K]
        E_phi_pred = torch.matmul(w_pred, self.A_phi.T)      # [B, K]
        return torch.cat(
            [E_theta_pred.real, E_theta_pred.imag, E_phi_pred.real, E_phi_pred.imag],
            dim=1,
        )

    def physics_residual(
        self,
        pred_amp: torch.Tensor,
        pred_cos: torch.Tensor,
        pred_sin: torch.Tensor,
        batch_x: torch.Tensor,
        eps: float,
        target_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # The reconstructed field from the predicted excitation is compared with
        # `target_feature` if given (e.g. the clean field, Method B), otherwise
        # with the measured (noisy) feature batch_x. Normalization always uses
        # the measured-field power so the metric scale is unchanged.
        pred_feature = self.forward(pred_amp, pred_cos, pred_sin)
        tgt = batch_x if target_feature is None else target_feature
        mse_per_sample = torch.mean((pred_feature - tgt) ** 2, dim=1)
        power_per_sample = torch.mean(batch_x ** 2, dim=1).clamp_min(eps)
        return mse_per_sample / power_per_sample


class CNNHybridNet(nn.Module):
    def __init__(self, input_dim: int, num_elements: int) -> None:
        super().__init__()
        if input_dim % 4 != 0:
            raise ValueError(f"input_dim={input_dim} is not divisible by 4")
        self.seq_len = input_dim // 4
        self.stem = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.res_layer1 = ResidualBlock(32)
        self.downsample = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.res_layer2 = ResidualBlock(64)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.mlp = nn.Sequential(
            nn.Linear(64 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_elements * 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 4, self.seq_len)
        x = self.stem(x)
        x = self.res_layer1(x)
        x = self.downsample(x)
        x = self.res_layer2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)


class PI1DResNet(nn.Module):
    def __init__(self, input_dim: int, num_elements: int, A_Etheta_np: np.ndarray, A_Ephi_np: np.ndarray) -> None:
        super().__init__()
        self.backbone = CNNHybridNet(input_dim=input_dim, num_elements=num_elements)
        self.aep = AEPForwardOperator(A_Etheta_np, A_Ephi_np)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ANNMLP(nn.Module):
    """Shallow MLP baseline (adapted from the phaseless-ANN diagnosis works)."""

    def __init__(self, input_dim: int, num_elements: int, hidden: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, num_elements * 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DNNMLP(nn.Module):
    """Deeper MLP baseline."""

    def __init__(self, input_dim: int, num_elements: int, hidden: int = 512, depth: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden, num_elements * 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CGANGenerator(nn.Module):
    """Conditional generator: (noisy feature, noise) -> raw excitation prediction."""

    def __init__(self, input_dim: int, noise_dim: int, out_dim: int, hidden: int = 384) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + noise_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z], dim=1))


class CGANDiscriminator(nn.Module):
    """Conditional discriminator operating on the PHYSICAL (decoded) target."""

    def __init__(self, cond_dim: int, target_dim: int, hidden: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + target_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cond, target], dim=1))


def decode_to_physical_target(raw_pred: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Map a raw (4N) network/generator output to a physically scaled (4N) target
    [amplitude in [0, amax], cos, sin in [-1,1], fault probability in [0,1]],
    used so that the cGAN discriminator compares real and fake targets on the
    SAME scale (fixes the adapted-cGAN input-scale mismatch)."""
    pred_amp, pred_cos, pred_sin, fault_logits, _ = decode_prediction(raw_pred, cfg)
    fault_prob = torch.sigmoid(fault_logits)
    return torch.cat([pred_amp, pred_cos, pred_sin, fault_prob], dim=1)


def label_to_physical_target(batch_y: torch.Tensor, cfg: Config) -> torch.Tensor:
    """The ground-truth label is already on the physical scale
    [amp, cos, sin, fault in {0,1}]; returned unchanged for clarity/symmetry."""
    return batch_y


def train_standard_model(
    model: nn.Module,
    cfg: Config,
    tag: str,
    input_dim: int,
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
    out_dir: Path,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train ANN/DNN baselines through the UNIFIED data pipeline and the SAME
    supervised objective used by the PI variants (with lambda_phys = 0 and no
    AEP operator). Identical seed, optimizer, scheduler and early stopping."""
    seed_everything(cfg.seed, cfg.deterministic_cudnn)
    model = model.to(device)

    # A frozen AEP operator is attached only to reuse compute_losses uniformly;
    # with lambda_phys = 0 it contributes nothing to the gradient.
    aep_op = AEPForwardOperator(A_Etheta, A_Ephi).to(device)
    cfg_local = replace(cfg, lambda_phys=0.0)
    bce = make_bce(cfg_local, device)

    train_ds = SyntheticPhasedArrayDataset(
        cfg_local, cfg_local.train_samples, base_seed=cfg_local.seed + 100_000,
        A_Etheta=A_Etheta, A_Ephi=A_Ephi, K=K, scenario="mixed",
        sampler=cfg_local.train_sampler, target_snr_db=cfg_local.train_snr_db, noise_kind="gaussian",
        unlabeled_fraction=cfg_local.unlabeled_fraction,
    )
    val_ds = SyntheticPhasedArrayDataset(
        cfg_local, cfg_local.val_samples, base_seed=cfg_local.seed + 200_000,
        A_Etheta=A_Etheta, A_Ephi=A_Ephi, K=K, scenario="mixed",
        sampler=cfg_local.train_sampler, target_snr_db=cfg_local.train_snr_db, noise_kind="gaussian",
        unlabeled_fraction=cfg_local.unlabeled_fraction,
    )
    train_loader = build_loader(cfg_local, train_ds, shuffle=True, device=device, seed=cfg_local.seed + 300_000)
    val_loader = build_loader(cfg_local, val_ds, shuffle=False, device=device, seed=cfg_local.seed + 400_000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_local.lr, weight_decay=cfg_local.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_local.epochs)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    history: List[Dict[str, Any]] = []
    print(f"\n=== Training {tag} (unified pipeline) ===")
    print(f"lambda_phys=0.0 (baseline), lambda_bce={cfg_local.lambda_bce}, "
          f"train_samples={cfg_local.train_samples}, val_samples={cfg_local.val_samples}")
    t0 = time.perf_counter()
    for ep in range(1, cfg_local.epochs + 1):
        train_stats = run_one_epoch(model, train_loader, optimizer, cfg_local, bce, device, aep_op)
        val_stats = run_one_epoch(model, val_loader, None, cfg_local, bce, device, aep_op)
        scheduler.step()
        row = {"epoch": ep, "lr": scheduler.get_last_lr()[0]}
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"val_{k}": v for k, v in val_stats.items()})
        history.append(row)
        if val_stats["total"] < best_val - 1e-7:
            best_val = val_stats["total"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1 or ep == cfg_local.epochs:
            print(f"{tag} epoch {ep:4d} | train {train_stats['total']:.5f} | "
                  f"val {val_stats['total']:.5f} | lr {scheduler.get_last_lr()[0]:.2e}")
        if bad >= cfg_local.patience:
            print(f"{tag} early stopping at epoch {ep}")
            break
    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    ensure_dir(out_dir)
    pd.DataFrame(history).to_csv(out_dir / f"history_{tag}.csv", index=False, encoding="utf-8-sig")
    torch.save(
        {"epoch": best_epoch, "best_val_loss": best_val, "model_state_dict": model.state_dict(),
         "config": asdict(cfg_local), "model_type": tag},
        out_dir / f"best_checkpoint_{tag}.pt",
    )
    summary = {"tag": tag, "best_epoch": best_epoch, "best_val_loss": best_val,
               "train_time_sec": train_time, "n_params": int(sum(p.numel() for p in model.parameters())),
               "history": history}
    return model, summary


def train_adapted_cgan(
    cfg: Config,
    input_dim: int,
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
    out_dir: Path,
) -> Tuple[CGANGenerator, Dict[str, Any]]:
    """Train the adapted-cGAN baseline through the UNIFIED data pipeline.

    Fairness-critical fix: the discriminator compares real vs. generated
    targets in the SAME physical (decoded) representation
    (amplitude/cos/sin/fault-probability), instead of comparing a physical
    label against an unbounded raw generator output. The supervised objective
    reuses the SAME `compute_losses` (lambda_phys = 0) as the other baselines.
    """
    seed_everything(cfg.seed, cfg.deterministic_cudnn)
    cfg_local = replace(cfg, lambda_phys=0.0)

    generator = CGANGenerator(input_dim=input_dim, noise_dim=cfg_local.gan_noise_dim,
                              out_dim=cfg_local.N * 4, hidden=cfg_local.cgan_hidden).to(device)
    discriminator = CGANDiscriminator(cond_dim=input_dim, target_dim=cfg_local.N * 4,
                                      hidden=cfg_local.cgan_hidden).to(device)
    aep_op = AEPForwardOperator(A_Etheta, A_Ephi).to(device)
    bce = make_bce(cfg_local, device)

    train_ds = SyntheticPhasedArrayDataset(
        cfg_local, cfg_local.train_samples, base_seed=cfg_local.seed + 100_000,
        A_Etheta=A_Etheta, A_Ephi=A_Ephi, K=K, scenario="mixed",
        sampler=cfg_local.train_sampler, target_snr_db=cfg_local.train_snr_db, noise_kind="gaussian",
        unlabeled_fraction=cfg_local.unlabeled_fraction,
    )
    val_ds = SyntheticPhasedArrayDataset(
        cfg_local, cfg_local.val_samples, base_seed=cfg_local.seed + 200_000,
        A_Etheta=A_Etheta, A_Ephi=A_Ephi, K=K, scenario="mixed",
        sampler=cfg_local.train_sampler, target_snr_db=cfg_local.train_snr_db, noise_kind="gaussian",
        unlabeled_fraction=cfg_local.unlabeled_fraction,
    )
    train_loader = build_loader(cfg_local, train_ds, shuffle=True, device=device, seed=cfg_local.seed + 300_000)
    val_loader = build_loader(cfg_local, val_ds, shuffle=False, device=device, seed=cfg_local.seed + 400_000)

    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg_local.lr, weight_decay=cfg_local.weight_decay)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=cfg_local.lr, weight_decay=cfg_local.weight_decay)

    best_state_g = None
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    history: List[Dict[str, Any]] = []
    print("\n=== Training adapted_cgan (unified pipeline) ===")
    print(f"lambda_phys=0.0 (baseline), lambda_bce={cfg_local.lambda_bce}, "
          f"train_samples={cfg_local.train_samples}, val_samples={cfg_local.val_samples}")
    t0 = time.perf_counter()
    for ep in range(1, cfg_local.epochs + 1):
        generator.train()
        discriminator.train()
        sum_g = 0.0
        count = 0
        for batch in train_loader:
            # Dataset yields (x, y) or (x, y, clean_x, label_mask). The cGAN is a
            # baseline (lambda_phys=0): unlabeled samples must NOT supervise the
            # discriminator / L1 / data losses, exactly as for the other baselines.
            if len(batch) == 4:
                batch_x, batch_y, clean_x, label_mask = batch
                clean_x = clean_x.to(device, non_blocking=True)
                label_mask = label_mask.to(device, non_blocking=True)
            else:
                batch_x, batch_y = batch
                clean_x, label_mask = None, None
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            bs = batch_x.size(0)
            if label_mask is None:
                lab = torch.ones(bs, device=device, dtype=batch_x.dtype)
            else:
                lab = label_mask.to(device, dtype=batch_x.dtype)
            n_lab = lab.sum().clamp_min(1.0)

            # ---- Discriminator step (decoded, physical-scale targets) ----
            # Only labeled samples have a valid "real" target; mask the rest.
            z = torch.randn(bs, cfg_local.gan_noise_dim, device=device)
            fake_raw = generator(batch_x, z).detach()
            real_phys = label_to_physical_target(batch_y, cfg_local)
            fake_phys = decode_to_physical_target(fake_raw, cfg_local)
            pred_real = discriminator(batch_x, real_phys).squeeze(-1)
            pred_fake = discriminator(batch_x, fake_phys).squeeze(-1)
            loss_d_real = (((pred_real - 1.0) ** 2) * lab).sum() / n_lab
            loss_d_fake = torch.mean((pred_fake - 0.0) ** 2)  # fake target valid for all
            loss_d = 0.5 * loss_d_real + 0.5 * loss_d_fake
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---- Generator step ----
            z = torch.randn(bs, cfg_local.gan_noise_dim, device=device)
            fake_raw = generator(batch_x, z)
            fake_phys = decode_to_physical_target(fake_raw, cfg_local)
            pred_fake_for_g = discriminator(batch_x, fake_phys).squeeze(-1)
            adv_loss = 0.5 * torch.mean((pred_fake_for_g - 1.0) ** 2)
            # Supervised loss reuses the masked compute_losses (lambda_phys=0).
            sup_loss, _ = compute_losses(
                fake_raw, batch_x, batch_y, cfg_local, bce, aep_op,
                clean_x=clean_x, label_mask=label_mask, phys_weight_scale=1.0,
            )
            # L1 only over labeled samples.
            l1_reg = (torch.abs(fake_phys - real_phys).mean(dim=1) * lab).sum() / n_lab
            loss_g = sup_loss + cfg_local.gan_adv_lambda * adv_loss + cfg_local.gan_l1_lambda * l1_reg
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg_local.grad_clip_norm)
            opt_g.step()
            count += bs
            sum_g += float(loss_g.detach().cpu()) * bs

        # ---- Validation: supervised loss of the generator mean prediction ----
        generator.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    batch_x, batch_y, clean_x, label_mask = batch
                    clean_x = clean_x.to(device, non_blocking=True)
                    label_mask = label_mask.to(device, non_blocking=True)
                else:
                    batch_x, batch_y = batch
                    clean_x, label_mask = None, None
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                bs = batch_x.size(0)
                z = torch.randn(bs, cfg_local.gan_noise_dim, device=device)
                fake_raw = generator(batch_x, z)
                val_loss, _ = compute_losses(
                    fake_raw, batch_x, batch_y, cfg_local, bce, aep_op,
                    clean_x=clean_x, label_mask=label_mask, phys_weight_scale=1.0,
                )
                val_count += bs
                val_sum += float(val_loss.detach().cpu()) * bs
        train_g = sum_g / max(count, 1)
        val_g = val_sum / max(val_count, 1)
        history.append({"epoch": ep, "train_total": train_g, "val_total": val_g})
        if val_g < best_val - 1e-7:
            best_val = val_g
            best_state_g = copy.deepcopy(generator.state_dict())
            best_epoch = ep
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1 or ep == cfg_local.epochs:
            print(f"adapted_cgan epoch {ep:4d} | train {train_g:.5f} | val {val_g:.5f}")
        if bad >= cfg_local.patience:
            print(f"adapted_cgan early stopping at epoch {ep}")
            break
    train_time = time.perf_counter() - t0
    if best_state_g is not None:
        generator.load_state_dict(best_state_g)

    ensure_dir(out_dir)
    pd.DataFrame(history).to_csv(out_dir / "history_adapted_cgan.csv", index=False, encoding="utf-8-sig")
    torch.save(
        {"epoch": best_epoch, "best_val_loss": best_val,
         "generator_state_dict": generator.state_dict(),
         "discriminator_state_dict": discriminator.state_dict(),
         "config": asdict(cfg_local), "model_type": "adapted_cgan"},
        out_dir / "best_checkpoint_adapted_cgan.pt",
    )
    summary = {"tag": "adapted_cgan", "best_epoch": best_epoch, "best_val_loss": best_val,
               "train_time_sec": train_time,
               "n_params": int(sum(p.numel() for p in generator.parameters())),
               "history": history}
    return generator, summary


@torch.no_grad()
def predict_batches_cgan(
    generator: CGANGenerator,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    avg_seeds: int = 8,
) -> Dict[str, np.ndarray]:
    """Inference for the cGAN baseline through the SAME prediction/metric path:
    the raw output is averaged over `avg_seeds` noise draws and decoded with the
    SAME `decode_prediction` as every other method."""
    generator.eval()
    amp_list, phase_list, prob_list, bin_list = [], [], [], []
    true_amp_list, true_phase_list, true_bin_list = [], [], []
    for batch in loader:
        batch_x, batch_y = batch[0], batch[1]
        batch_x = batch_x.to(device, non_blocking=True)
        preds = []
        for _ in range(avg_seeds):
            z = torch.randn(batch_x.size(0), cfg.gan_noise_dim, device=device)
            preds.append(generator(batch_x, z))
        raw_pred = torch.stack(preds, dim=0).mean(dim=0)
        pred_amp, pred_cos, pred_sin, fault_logits, _ = decode_prediction(raw_pred, cfg)
        prob = torch.sigmoid(fault_logits)
        amp_list.append(pred_amp.cpu().numpy())
        phase_list.append(torch.rad2deg(torch.atan2(pred_sin, pred_cos)).cpu().numpy())
        prob_list.append(prob.cpu().numpy())
        bin_list.append((prob > cfg.fault_probability_threshold).int().cpu().numpy())
        ty = batch_y.numpy()
        true_amp_list.append(ty[:, : cfg.N])
        true_cos = ty[:, cfg.N : 2 * cfg.N]
        true_sin = ty[:, 2 * cfg.N : 3 * cfg.N]
        true_phase_list.append(np.rad2deg(np.arctan2(true_sin, true_cos)))
        true_bin_list.append(ty[:, 3 * cfg.N :].astype(np.int32))
    return {
        "pred_amp": np.concatenate(amp_list, 0),
        "pred_phase_deg": np.concatenate(phase_list, 0),
        "pred_prob": np.concatenate(prob_list, 0),
        "pred_binary_head": np.concatenate(bin_list, 0),
        "true_amp": np.concatenate(true_amp_list, 0),
        "true_phase_deg": np.concatenate(true_phase_list, 0),
        "true_binary": np.concatenate(true_bin_list, 0),
        "phys_residual": np.full(np.concatenate(amp_list, 0).shape[0], np.nan, dtype=np.float64),
    }


# ----------------------------- CS baselines ---------------------------------

def build_cs_system(
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    delta_E_theta: np.ndarray,
    delta_E_phi: np.ndarray,
    idx_obs: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:

    A_cs_th = A_Etheta[idx_obs, :].astype(np.complex128)
    A_cs_ph = A_Ephi[idx_obs, :].astype(np.complex128)
    M = len(idx_obs)
    A_real = np.zeros((4 * M, 2 * cfg.N), dtype=np.float64)
    A_real[0:M, 0:cfg.N] = np.real(A_cs_th)
    A_real[0:M, cfg.N:] = -np.imag(A_cs_th)
    A_real[M:2 * M, 0:cfg.N] = np.imag(A_cs_th)
    A_real[M:2 * M, cfg.N:] = np.real(A_cs_th)
    A_real[2 * M:3 * M, 0:cfg.N] = np.real(A_cs_ph)
    A_real[2 * M:3 * M, cfg.N:] = -np.imag(A_cs_ph)
    A_real[3 * M:4 * M, 0:cfg.N] = np.imag(A_cs_ph)
    A_real[3 * M:4 * M, cfg.N:] = np.real(A_cs_ph)
    y = np.concatenate([
        np.real(delta_E_theta[idx_obs]), np.imag(delta_E_theta[idx_obs]),
        np.real(delta_E_phi[idx_obs]), np.imag(delta_E_phi[idx_obs]),
    ]).astype(np.float64)
    return A_real, y


def fista_solver(H: np.ndarray, c: np.ndarray, L: float, lmbda: float, max_iter: int = 500, tol: float = 1e-5) -> np.ndarray:
    x = np.zeros(c.shape[0], dtype=np.float64)
    z = np.zeros_like(x)
    t = 1.0
    for _ in range(max_iter):
        x_old = x.copy()
        grad = H @ z - c
        v = z - grad / L
        thresh = lmbda / L
        x = np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)
        if np.max(np.abs(x - x_old)) < tol:
            break
        t_old = t
        t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x + ((t_old - 1.0) / t) * (x - x_old)
    return x


def recover_excitation_cs(method: str, A_matrix: np.ndarray, y_target: np.ndarray, cfg: Config, omp_k: int, lasso_alpha: float) -> np.ndarray:
    """Return the recovered complex excitation w_rec (length N)."""
    if method == "OMP":
        from sklearn.linear_model import OrthogonalMatchingPursuit
        solver = OrthogonalMatchingPursuit(n_nonzero_coefs=omp_k, fit_intercept=False)
        solver.fit(A_matrix, y_target)
        coef = solver.coef_.astype(np.float64)
    elif method == "Lasso":
        from sklearn.linear_model import Lasso
        from sklearn.exceptions import ConvergenceWarning
        solver = Lasso(alpha=lasso_alpha, fit_intercept=False,
                       max_iter=cfg.lasso_max_iter, tol=cfg.lasso_tol)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            solver.fit(A_matrix, y_target)
        coef = solver.coef_.astype(np.float64)
    elif method == "FISTA":
        H = A_matrix.T @ A_matrix
        c = A_matrix.T @ y_target
        L_val = float(np.linalg.norm(A_matrix, ord=2) ** 2)
        coef = fista_solver(H, c, L_val, lmbda=lasso_alpha * A_matrix.shape[0],
                            max_iter=cfg.fista_max_iter, tol=cfg.fista_tol)
    else:
        raise ValueError(f"Unknown CS method: {method}")
    return np.ones(cfg.N, dtype=np.complex128) + coef[:cfg.N] + 1j * coef[cfg.N:]


def evaluate_cs_method(
    method: str,
    cfg: Config,
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    M_obs: int,
    omp_k: int,
    lasso_alpha: float,
    base_seed: int,
    eval_samples: Optional[int] = None,
) -> Dict[str, Any]:

    n = cfg.eval_samples if eval_samples is None else int(eval_samples)
    rng_obs = np.random.default_rng(base_seed)
    if M_obs >= K:
        idx_obs = np.arange(K)
    else:
        idx_obs = np.sort(rng_obs.choice(K, size=M_obs, replace=False))

    A_real_full = None
    if method in ("FISTA",):
        A_real_full = None  # built per sample below (depends only on idx_obs, reused)

    w_ideal = np.ones((cfg.N, 1), dtype=np.complex64)
    E_th_ideal = (A_Etheta @ w_ideal).reshape(-1)
    E_ph_ideal = (A_Ephi @ w_ideal).reshape(-1)

    rng = np.random.default_rng(base_seed + 12345)
    true_bin_list, pred_bin_list = [], []
    true_amp_list, true_phase_list, pred_amp_list, pred_phase_list = [], [], [], []
    A_real_cached = None
    t0 = time.perf_counter()
    for _ in range(n):
        s = generate_sample(cfg, rng, A_Etheta, A_Ephi, K, scenario="mixed",
                            sampler=cfg.eval_sampler, target_snr_db=cfg.nominal_eval_snr_db,
                            noise_kind="gaussian", return_meta=True)
        amp_true = s["amp_true"]
        phase_true_deg = s["phase_true_deg"]
        # Reconstruct the noisy complex fields from the stored real feature.
        feat = s["feature"]
        E_th_noisy = feat[0:K] + 1j * feat[K:2 * K]
        E_ph_noisy = feat[2 * K:3 * K] + 1j * feat[3 * K:4 * K]
        delta_th = E_th_noisy - E_th_ideal
        delta_ph = E_ph_noisy - E_ph_ideal
        A_real, y = build_cs_system(A_Etheta, A_Ephi, delta_th, delta_ph, idx_obs, cfg)
        w_rec = recover_excitation_cs(method, A_real, y, cfg, omp_k, lasso_alpha)
        pred_amp = np.abs(w_rec)
        pred_phase_deg = np.rad2deg(np.angle(w_rec))
        true_bin_list.append(diagnose_rule_np(amp_true, phase_true_deg, cfg.amp_fault_threshold, cfg.phase_fault_threshold_deg))
        pred_bin_list.append(diagnose_rule_np(pred_amp, pred_phase_deg, cfg.amp_fault_threshold, cfg.phase_fault_threshold_deg))
        true_amp_list.append(amp_true)
        true_phase_list.append(phase_true_deg)
        pred_amp_list.append(pred_amp.astype(np.float32))
        pred_phase_list.append(pred_phase_deg.astype(np.float32))
    infer_time_ms = 1000.0 * (time.perf_counter() - t0) / max(n, 1)

    true_bin = np.stack(true_bin_list, 0)
    pred_bin = np.stack(pred_bin_list, 0)
    metrics = compute_metrics_from_arrays(
        true_bin, pred_bin,
        np.stack(true_amp_list, 0), np.stack(true_phase_list, 0),
        np.stack(pred_amp_list, 0), np.stack(pred_phase_list, 0),
        np.full(true_bin.shape[0], np.nan), cfg,
    )
    metrics["infer_time_ms_per_sample"] = infer_time_ms
    metrics["M_obs"] = int(len(idx_obs))
    return {"metrics": metrics, "idx_obs": idx_obs}




# =============================================================================
# 4. Decoding and losses
# =============================================================================


def decode_prediction(raw_pred: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_amp = raw_pred[:, : cfg.N]
    raw_cos = raw_pred[:, cfg.N : 2 * cfg.N]
    raw_sin = raw_pred[:, 2 * cfg.N : 3 * cfg.N]
    fault_logits = raw_pred[:, 3 * cfg.N :]

    pred_amp = cfg.amp_max_value * torch.sigmoid(raw_amp)

    # Stable phase-vector normalization, directly addressing Reviewer 2 Comment 3.
    phase_vec = torch.stack([raw_cos, raw_sin], dim=-1)  # [B, N, 2]
    phase_vec = F.normalize(phase_vec, p=2, dim=-1, eps=cfg.eps_phase_norm)
    pred_cos = phase_vec[..., 0]
    pred_sin = phase_vec[..., 1]
    raw_phase_norm = torch.sqrt(raw_cos ** 2 + raw_sin ** 2 + cfg.eps_loss)
    return pred_amp, pred_cos, pred_sin, fault_logits, raw_phase_norm


def weighted_mean(loss: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sum(weight * loss) / torch.clamp(torch.sum(weight), min=eps)


def refine_with_physics(
    aep_op: "AEPForwardOperator",
    init_amp: torch.Tensor,
    init_cos: torch.Tensor,
    init_sin: torch.Tensor,
    measured_feature: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    amp = init_amp.detach().clone().requires_grad_(True)
    phs = torch.atan2(init_sin, init_cos).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([amp, phs], lr=cfg.tta_lr)
    target = measured_feature.detach()
    with torch.enable_grad():
        for _ in range(cfg.tta_iters):
            optimizer.zero_grad(set_to_none=True)
            a = amp.clamp(0.0, cfg.amp_max_value) if cfg.tta_amp_clamp else amp
            pred_feat = aep_op.forward(a, torch.cos(phs), torch.sin(phs))
            loss = torch.mean((pred_feat - target) ** 2)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        a_final = amp.clamp(0.0, cfg.amp_max_value) if cfg.tta_amp_clamp else amp
        cos_final = torch.cos(phs)
        sin_final = torch.sin(phs)
    return a_final.detach(), cos_final.detach(), sin_final.detach()


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return weighted_mean(loss, weight, eps)


def compute_losses(
    raw_pred: torch.Tensor,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    cfg: Config,
    bce: nn.Module,
    aep_op: AEPForwardOperator,
    clean_x: Optional[torch.Tensor] = None,
    label_mask: Optional[torch.Tensor] = None,
    phys_weight_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    pred_amp, pred_cos, pred_sin, pred_fault_logits, raw_phase_norm = decode_prediction(raw_pred, cfg)

    true_amp = batch_y[:, : cfg.N]
    true_cos = batch_y[:, cfg.N : 2 * cfg.N]
    true_sin = batch_y[:, 2 * cfg.N : 3 * cfg.N]
    true_fault = batch_y[:, 3 * cfg.N :]

    true_phase_rad = torch.atan2(true_sin, true_cos)
    pred_phase_rad = torch.atan2(pred_sin, pred_cos)

    amp_fault_mask = (true_amp < cfg.amp_fault_threshold).float()
    phase_fault_mask = (torch.abs(true_phase_rad) > math.radians(cfg.phase_fault_threshold_deg)).float()
    severe_amp_mask = (true_amp < cfg.severe_amp_focus_threshold).float()

    amp_weight = 1.0 + cfg.amp_fault_weight * amp_fault_mask + cfg.severe_amp_extra_weight * severe_amp_mask

    phase_observable_mask = (true_amp >= cfg.phase_amp_mask_threshold).float()
    phase_weight = phase_observable_mask * (1.0 + cfg.phase_fault_weight * phase_fault_mask)

    # ---- Semi-supervised per-sample masking of the SUPERVISED data losses ----
    if label_mask is None:
        lab = torch.ones(batch_x.size(0), device=batch_x.device, dtype=batch_x.dtype)
    else:
        lab = label_mask.to(batch_x.device, dtype=batch_x.dtype)
    lab_col = lab.unsqueeze(1)  # [B,1] broadcast over N
    n_lab = lab.sum().clamp_min(1.0)

    # Weight the per-element supervised weights by the sample's labeled flag, so
    # unlabeled samples contribute zero to amp/phase/BCE.
    amp_weight = amp_weight * lab_col
    phase_weight = phase_weight * lab_col

    loss_amp = weighted_smooth_l1(pred_amp, true_amp, amp_weight, cfg.smooth_l1_beta_amp, cfg.eps_loss)
    loss_cos = weighted_smooth_l1(pred_cos, true_cos, phase_weight, cfg.smooth_l1_beta_phase, cfg.eps_loss)
    loss_sin = weighted_smooth_l1(pred_sin, true_sin, phase_weight, cfg.smooth_l1_beta_phase, cfg.eps_loss)

    phase_err = wrap_phase_rad(pred_phase_rad - true_phase_rad)
    loss_phase_angle = weighted_mean(1.0 - torch.cos(phase_err), phase_weight, cfg.eps_loss)

    # phase-norm regularizer applies to all samples (it only constrains the
    # network's own (cos,sin) output magnitude, needs no labels).
    loss_phase_norm = torch.mean((raw_phase_norm - 1.0) ** 2)

    # Direct complex-excitation consistency.  This loss is still supervised and
    # is masked for unlabeled samples, but it is more physically matched to the
    # AEP equation E = A w than independent amplitude / trigonometric losses.
    pred_w_re = pred_amp * pred_cos
    pred_w_im = pred_amp * pred_sin
    true_w_re = true_amp * true_cos
    true_w_im = true_amp * true_sin
    w_fault_mask = torch.clamp(amp_fault_mask + phase_fault_mask, max=1.0)
    w_weight = lab_col * (1.0 + 2.0 * w_fault_mask + 1.0 * severe_amp_mask)
    loss_w_re = weighted_smooth_l1(pred_w_re, true_w_re, w_weight, cfg.smooth_l1_beta_amp, cfg.eps_loss)
    loss_w_im = weighted_smooth_l1(pred_w_im, true_w_im, w_weight, cfg.smooth_l1_beta_amp, cfg.eps_loss)
    loss_complex_w = loss_w_re + loss_w_im

    # BCE only over labeled samples (per-sample, then masked-averaged).
    bce_per = F.binary_cross_entropy_with_logits(
        pred_fault_logits, true_fault,
        pos_weight=torch.full((cfg.N,), float(cfg.bce_pos_weight), device=batch_x.device),
        reduction="none",
    ).mean(dim=1)  # [B]
    loss_bce = (bce_per * lab).sum() / n_lab

    # ---- Physics loss ----
    target_feat = clean_x if (cfg.physics_anchor_clean and clean_x is not None) else None
    if cfg.lambda_phys > 0.0:
        phys_per = aep_op.physics_residual(pred_amp, pred_cos, pred_sin, batch_x, cfg.eps_loss, target_feat)  # [B]
        if cfg.physics_on_labeled:
            phys_mask = torch.ones_like(lab)            # all samples
        else:
            phys_mask = 1.0 - lab                       # unlabeled only
        denom = phys_mask.sum().clamp_min(1.0)
        loss_phys = (phys_per * phys_mask).sum() / denom
    else:
        with torch.no_grad():
            phys_per = aep_op.physics_residual(pred_amp, pred_cos, pred_sin, batch_x, cfg.eps_loss, target_feat)
            loss_phys = phys_per.mean()

    total_loss = (
        cfg.lambda_amp * loss_amp
        + cfg.lambda_phase_trig * (loss_cos + loss_sin)
        + cfg.lambda_phase_angle * loss_phase_angle
        + cfg.lambda_phase_norm * loss_phase_norm
        + cfg.lambda_complex_w * loss_complex_w
        + cfg.lambda_bce * loss_bce
        + (phys_weight_scale * cfg.lambda_phys) * loss_phys
    )

    stats = {
        "total": float(total_loss.detach().cpu()),
        "amp": float(loss_amp.detach().cpu()),
        "cos": float(loss_cos.detach().cpu()),
        "sin": float(loss_sin.detach().cpu()),
        "phase_angle": float(loss_phase_angle.detach().cpu()),
        "phase_norm": float(loss_phase_norm.detach().cpu()),
        "complex_w": float(loss_complex_w.detach().cpu()),
        "bce": float(loss_bce.detach().cpu()),
        "phys": float(loss_phys.detach().cpu()),
    }
    return total_loss, stats


# =============================================================================
# 5. Training and inference
# =============================================================================


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    cfg: Config,
    bce: nn.Module,
    device: torch.device,
    aep_op: Optional["AEPForwardOperator"] = None,
    phys_weight_scale: float = 1.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    physics = aep_op if aep_op is not None else getattr(model, "aep", None)
    sums = {k: 0.0 for k in ["total", "amp", "cos", "sin", "phase_angle", "phase_norm", "complex_w", "bce", "phys"]}
    count = 0
    n_labeled = 0
    n_total = 0

    for batch in loader:
        # Dataset may yield (x, y) [legacy/baseline] or
        # (x, y, clean_x, label_mask) [semi-supervised PI pipeline].
        if len(batch) == 4:
            batch_x, batch_y, clean_x, label_mask = batch
            clean_x = clean_x.to(device, dtype=torch.float32, non_blocking=True)
            label_mask = label_mask.to(device, dtype=torch.float32, non_blocking=True)
        else:
            batch_x, batch_y = batch
            clean_x, label_mask = None, None
        batch_x = batch_x.to(device, dtype=torch.float32, non_blocking=True)
        batch_y = batch_y.to(device, dtype=torch.float32, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            raw_pred = model(batch_x)
            loss, stats = compute_losses(
                raw_pred, batch_x, batch_y, cfg, bce, physics,
                clean_x=clean_x, label_mask=label_mask, phys_weight_scale=phys_weight_scale,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
        bs = batch_x.size(0)
        count += bs
        if label_mask is not None:
            n_labeled += int(label_mask.sum().item())
            n_total += bs
        for k in sums:
            sums[k] += stats[k] * bs
    out = {k: v / max(count, 1) for k, v in sums.items()}
    if n_total > 0:
        out["labeled_frac"] = n_labeled / n_total
    return out


def make_bce(cfg: Config, device: torch.device) -> nn.Module:
    """Single source of the (class-imbalance-weighted) BCE loss, shared by ALL
    learning-based methods so that the fault-classification objective is
    identical across the proposed model and every baseline."""
    pos_weight = torch.full((cfg.N,), float(cfg.bce_pos_weight), device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_model_variant(
    cfg: Config,
    variant_name: str,
    input_dim: int,
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
    out_dir: Path,
) -> Tuple[PI1DResNet, Dict[str, Any]]:
    seed_everything(cfg.seed, cfg.deterministic_cudnn)
    model = PI1DResNet(input_dim, cfg.N, A_Etheta, A_Ephi).to(device)

    train_ds = SyntheticPhasedArrayDataset(
        cfg,
        cfg.train_samples,
        base_seed=cfg.seed + 100_000,
        A_Etheta=A_Etheta,
        A_Ephi=A_Ephi,
        K=K,
        scenario="mixed",
        sampler=cfg.train_sampler,
        target_snr_db=cfg.train_snr_db,
        noise_kind="gaussian",
        unlabeled_fraction=cfg.unlabeled_fraction,
    )
    val_ds = SyntheticPhasedArrayDataset(
        cfg,
        cfg.val_samples,
        base_seed=cfg.seed + 200_000,
        A_Etheta=A_Etheta,
        A_Ephi=A_Ephi,
        K=K,
        scenario="mixed",
        sampler=cfg.train_sampler,
        target_snr_db=cfg.train_snr_db,
        noise_kind="gaussian",
        unlabeled_fraction=cfg.unlabeled_fraction,
    )
    train_loader = build_loader(cfg, train_ds, shuffle=True, device=device, seed=cfg.seed + 300_000)
    val_loader = build_loader(cfg, val_ds, shuffle=False, device=device, seed=cfg.seed + 400_000)

    bce = make_bce(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    bad = 0
    t0 = time.perf_counter()

    print(f"\n=== Training {variant_name} ===")
    print(f"lambda_phys={cfg.lambda_phys}, lambda_bce={cfg.lambda_bce}, train_samples={cfg.train_samples}, val_samples={cfg.val_samples}")
    for ep in range(1, cfg.epochs + 1):
        # Curriculum: ramp the physics weight from 0 to 1 over the warmup epochs
        # (learn the data fit first, then enforce physics consistency).
        if cfg.phys_warmup_epochs > 0:
            phys_scale = min(1.0, ep / float(cfg.phys_warmup_epochs))
        else:
            phys_scale = 1.0
        train_stats = run_one_epoch(model, train_loader, optimizer, cfg, bce, device, phys_weight_scale=phys_scale)
        val_stats = run_one_epoch(model, val_loader, None, cfg, bce, device, phys_weight_scale=phys_scale)
        scheduler.step()
        row = {"epoch": ep, "lr": scheduler.get_last_lr()[0]}
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"val_{k}": v for k, v in val_stats.items()})
        history.append(row)

        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            bad = 0
        else:
            bad += 1

        if ep == 1 or ep % 10 == 0 or ep == cfg.epochs:
            print(
                f"{variant_name:18s} epoch {ep:4d} | "
                f"train {train_stats['total']:.5f} | val {val_stats['total']:.5f} | "
                f"val_phys {val_stats['phys']:.5e} | lr {scheduler.get_last_lr()[0]:.2e}"
            )
        if bad >= cfg.patience:
            print(f"{variant_name}: early stopping at epoch {ep}")
            break

    if best_state is None:
        raise RuntimeError(f"No best_state recorded for {variant_name}")
    model.load_state_dict(best_state)
    train_time_sec = time.perf_counter() - t0

    pd.DataFrame(history).to_csv(out_dir / f"history_{variant_name}.csv", index=False, encoding="utf-8-sig")
    torch.save(
        {
            "variant_name": variant_name,
            "epoch": best_epoch,
            "best_val_loss": best_val,
            "train_time_sec": train_time_sec,
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
        },
        out_dir / f"checkpoint_{variant_name}.pt",
    )
    return model, {"variant": variant_name, "best_epoch": best_epoch, "best_val_loss": best_val, "train_time_sec": train_time_sec, "history": history}


@torch.no_grad()
def predict_batches(
    model: PI1DResNet,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    use_tta: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()
    pred_amp_list: List[np.ndarray] = []
    pred_phase_list: List[np.ndarray] = []
    pred_prob_list: List[np.ndarray] = []
    pred_binary_head_list: List[np.ndarray] = []
    phys_residual_list: List[np.ndarray] = []

    true_amp_list: List[np.ndarray] = []
    true_phase_list: List[np.ndarray] = []
    true_binary_list: List[np.ndarray] = []

    aep_op = getattr(model, "aep", None)
    do_tta = bool(use_tta and cfg.tta_enable and aep_op is not None)

    total_infer_sec = 0.0
    n_samples_timed = 0

    for batch in loader:
        batch_x, batch_y = batch[0], batch[1]
        batch_x = batch_x.to(device, dtype=torch.float32, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_infer0 = time.perf_counter()
        with torch.no_grad():
            raw_pred = model(batch_x)
            pred_amp_t, pred_cos_t, pred_sin_t, logits_t, _ = decode_prediction(raw_pred, cfg)

        # ---- Optional Test-Time Physics Refinement (proposed model only) ----
        if do_tta:
            pred_amp_t, pred_cos_t, pred_sin_t = refine_with_physics(
                aep_op, pred_amp_t, pred_cos_t, pred_sin_t, batch_x, cfg
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        # Inference timing covers the forward pass (and TTPR, which is part of
        # the proposed model's inference cost); it excludes host-side numpy
        # post-processing and metric assembly.
        total_infer_sec += time.perf_counter() - t_infer0
        n_samples_timed += batch_x.shape[0]

        if aep_op is not None:
            with torch.no_grad():
                residual_t = aep_op.physics_residual(pred_amp_t, pred_cos_t, pred_sin_t, batch_x, cfg.eps_loss)
                residual_np = residual_t.detach().cpu().numpy()
        else:
            residual_np = np.full(batch_x.shape[0], np.nan, dtype=np.float64)

        head_prob_t = torch.sigmoid(logits_t)
        if do_tta:
            # Let the AEP-refined physical amplitude/phase contribute to the
            # final fault probability; otherwise TTPR would improve only MAE/NRMSE
            # while leaving the fault map unchanged.
            final_prob_t = fuse_fault_probability(head_prob_t, pred_amp_t, pred_cos_t, pred_sin_t, cfg)
        else:
            final_prob_t = head_prob_t

        pred_amp = pred_amp_t.detach().cpu().numpy()
        pred_phase = np.rad2deg(np.arctan2(pred_sin_t.detach().cpu().numpy(), pred_cos_t.detach().cpu().numpy()))
        pred_prob = final_prob_t.detach().cpu().numpy()
        pred_binary_head = (pred_prob > cfg.fault_probability_threshold).astype(np.int32)

        by = batch_y.numpy()
        true_amp = by[:, : cfg.N]
        true_phase = np.rad2deg(np.arctan2(by[:, 2 * cfg.N : 3 * cfg.N], by[:, cfg.N : 2 * cfg.N]))
        true_binary = by[:, 3 * cfg.N :].astype(np.int32)

        pred_amp_list.append(pred_amp)
        pred_phase_list.append(pred_phase)
        pred_prob_list.append(pred_prob)
        pred_binary_head_list.append(pred_binary_head)
        phys_residual_list.append(residual_np)
        true_amp_list.append(true_amp)
        true_phase_list.append(true_phase)
        true_binary_list.append(true_binary)

    return {
        "pred_amp": np.concatenate(pred_amp_list, axis=0),
        "pred_phase_deg": np.concatenate(pred_phase_list, axis=0),
        "pred_fault_prob": np.concatenate(pred_prob_list, axis=0),
        "pred_binary_head": np.concatenate(pred_binary_head_list, axis=0),
        "phys_residual": np.concatenate(phys_residual_list, axis=0),
        "true_amp": np.concatenate(true_amp_list, axis=0),
        "true_phase_deg": np.concatenate(true_phase_list, axis=0),
        "true_binary": np.concatenate(true_binary_list, axis=0),
        "infer_time_ms_per_sample": 1000.0 * total_infer_sec / max(n_samples_timed, 1),
    }


def binary_from_prediction(pred: Dict[str, np.ndarray], cfg: Config, mode: str) -> np.ndarray:
    mode = mode.lower()
    head = pred["pred_binary_head"].astype(np.int32)
    reg = diagnose_rule_np(
        pred["pred_amp"],
        pred["pred_phase_deg"],
        cfg.amp_fault_threshold,
        cfg.phase_fault_threshold_deg,
    ).astype(np.int32)
    if mode == "fault_head":
        return head
    if mode == "reg_threshold":
        return reg
    if mode == "hybrid_or":
        return ((head == 1) | (reg == 1)).astype(np.int32)
    if mode == "hybrid_and":
        return ((head == 1) & (reg == 1)).astype(np.int32)
    raise ValueError(f"Unknown decision mode: {mode}")


def compute_metrics_from_arrays(
    true_binary: np.ndarray,
    pred_binary: np.ndarray,
    true_amp: np.ndarray,
    true_phase_deg: np.ndarray,
    pred_amp: np.ndarray,
    pred_phase_deg: np.ndarray,
    phys_residual: np.ndarray,
    cfg: Config,
    prefix: str = "",
) -> Dict[str, float]:
    true_flat = true_binary.reshape(-1).astype(np.int32)
    pred_flat = pred_binary.reshape(-1).astype(np.int32)
    precision = precision_score(true_flat, pred_flat, zero_division=0)
    recall = recall_score(true_flat, pred_flat, zero_division=0)
    f1 = f1_score(true_flat, pred_flat, zero_division=0)
    ema_mask = np.all(pred_binary.astype(np.int32) == true_binary.astype(np.int32), axis=1)
    ema = float(np.mean(ema_mask))

    fault_mask = true_binary.astype(bool)
    phase_observable_mask = (true_amp >= cfg.phase_amp_mask_threshold)
    phase_eval_mask = fault_mask & phase_observable_mask

    amp_mae_fault = float(np.mean(np.abs(pred_amp[fault_mask] - true_amp[fault_mask]))) if np.any(fault_mask) else float("nan")
    phase_err = np.abs(wrap_phase_deg(pred_phase_deg - true_phase_deg))
    phase_mae_masked = float(np.mean(phase_err[phase_eval_mask])) if np.any(phase_eval_mask) else float("nan")
    phase_mae_unmasked_fault = float(np.mean(phase_err[fault_mask])) if np.any(fault_mask) else float("nan")

    field_nrmse_db = float(np.mean(10.0 * np.log10(np.maximum(phys_residual, 1e-12))))
    r_phys_mean = float(np.mean(phys_residual))
    r_phys_correct = float(np.mean(phys_residual[ema_mask])) if np.any(ema_mask) else float("nan")
    r_phys_wrong = float(np.mean(phys_residual[~ema_mask])) if np.any(~ema_mask) else float("nan")

    cm = confusion_matrix(true_flat, pred_flat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    far = float(fp / max(fp + tn, 1))
    miss = float(fn / max(fn + tp, 1))

    out = {
        "precision_pct": 100.0 * precision,
        "recall_pct": 100.0 * recall,
        "f1_pct": 100.0 * f1,
        "ema_pct": 100.0 * ema,
        "amp_mae_fault": amp_mae_fault,
        "phase_mae_deg_masked": phase_mae_masked,
        "phase_mae_deg_unmasked_fault": phase_mae_unmasked_fault,
        "field_nrmse_db": field_nrmse_db,
        "r_phys_mean": r_phys_mean,
        "r_phys_ema_correct": r_phys_correct,
        "r_phys_ema_wrong": r_phys_wrong,
        "false_alarm_rate_pct": 100.0 * far,
        "miss_detection_rate_pct": 100.0 * miss,
    }
    if prefix:
        return {f"{prefix}{k}": v for k, v in out.items()}
    return out


def evaluate_model_on_dataset(
    model: PI1DResNet,
    cfg: Config,
    A_Etheta_eval: np.ndarray,
    A_Ephi_eval: np.ndarray,
    K: int,
    device: torch.device,
    scenario: str = "mixed",
    sampler: str = "uniform",
    target_snr_db: NumberOrRange = 25.0,
    noise_kind: str = "gaussian",
    decision_mode: str = "fault_head",
    eval_samples: Optional[int] = None,
    base_seed: Optional[int] = None,
    use_tta: bool = False,
) -> Dict[str, Any]:
    n = cfg.eval_samples if eval_samples is None else int(eval_samples)
    seed = cfg.seed + 500_000 if base_seed is None else int(base_seed)
    ds = SyntheticPhasedArrayDataset(
        cfg,
        n,
        base_seed=seed,
        A_Etheta=A_Etheta_eval,
        A_Ephi=A_Ephi_eval,
        K=K,
        scenario=scenario,
        sampler=sampler,
        target_snr_db=target_snr_db,
        noise_kind=noise_kind,
    )
    loader = build_loader(cfg, ds, shuffle=False, device=device, seed=seed + 99)
    pred = predict_batches(model, loader, cfg, device, use_tta=use_tta)
    pred_binary = binary_from_prediction(pred, cfg, decision_mode)
    metrics = compute_metrics_from_arrays(
        pred["true_binary"],
        pred_binary,
        pred["true_amp"],
        pred["true_phase_deg"],
        pred["pred_amp"],
        pred["pred_phase_deg"],
        pred["phys_residual"],
        cfg,
    )
    metrics.update(
        {
            "scenario": scenario,
            "sampler": sampler,
            "target_snr_db": target_snr_db if isinstance(target_snr_db, float) else str(tuple(target_snr_db)),
            "noise_kind": noise_kind,
            "decision_mode": decision_mode,
            "eval_samples": n,
            "infer_time_ms_per_sample": pred.get("infer_time_ms_per_sample", float("nan")),
        }
    )
    return {"metrics": metrics, "pred": pred, "pred_binary": pred_binary}


# =============================================================================
# 6. Reviewer-response analyses
# =============================================================================


def dataframe_to_latex_table(df: pd.DataFrame, caption: str, label: str, float_format: str = "%.2f") -> str:
    return df.to_latex(index=False, escape=False, caption=caption, label=label, float_format=lambda x: float_format % x)


def format_value(x: float, ndigits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.{ndigits}f}"


def run_physics_ablation_table(
    models: Dict[str, PI1DResNet],
    cfgs: Dict[str, Config],
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, str]]:

    specs = [
        ("data_driven_no_phys", r"$\lambda_{\mathrm{phys}}=0$ (data-driven)", False),
        ("proposed", rf"$\lambda_{{\mathrm{{phys}}}}={cfgs['proposed'].lambda_phys:g}$ (proposed, network only)", False),
        ("proposed", rf"$\lambda_{{\mathrm{{phys}}}}={cfgs['proposed'].lambda_phys:g}$ (proposed + TTPR-fusion)", True),
    ]
    rows = []
    metrics_list = []
    for name, label, use_tta in specs:
        res = evaluate_model_on_dataset(
            models[name],
            cfgs[name],
            A_Etheta,
            A_Ephi,
            K,
            device,
            scenario="mixed",
            sampler=cfgs[name].eval_sampler,
            target_snr_db=cfgs[name].nominal_eval_snr_db,
            noise_kind="gaussian",
            decision_mode="fault_head",
            base_seed=cfgs[name].seed + 600_000,
            use_tta=use_tta,
        )["metrics"]
        metrics_list.append(res)
        rows.append(
            {
                "Training objective": label,
                "S-E F1 (%)": res["f1_pct"],
                "EMA (%)": res["ema_pct"],
                "Amp. MAE": res["amp_mae_fault"],
                r"Phase MAE ($^\circ$)": res["phase_mae_deg_masked"],
                "NRMSE (dB)": res["field_nrmse_db"],
                "r_phys": res["r_phys_mean"],
            }
        )
    df = pd.DataFrame(rows)
    d0 = metrics_list[0]          # no_phys
    d_net = metrics_list[1]       # proposed network-only
    d_tta = metrics_list[2]       # proposed + TTPR (the reported proposed config)
    # Placeholders: row 0 = no_phys, row 1 = proposed (with TTPR, the final config).
    values = {
        "F10": format_value(float(d0["f1_pct"]), 2),
        "EMA0": format_value(float(d0["ema_pct"]), 2),
        "A0": format_value(float(d0["amp_mae_fault"]), 4),
        "P0": format_value(float(d0["phase_mae_deg_masked"]), 2),
        "N0": format_value(float(d0["field_nrmse_db"]), 2),
        "F11": format_value(float(d_tta["f1_pct"]), 2),
        "EMA1": format_value(float(d_tta["ema_pct"]), 2),
        "A1": format_value(float(d_tta["amp_mae_fault"]), 4),
        "P1": format_value(float(d_tta["phase_mae_deg_masked"]), 2),
        "N1": format_value(float(d_tta["field_nrmse_db"]), 2),
        "DeltaEMA": format_value(float(d_tta["ema_pct"] - d0["ema_pct"]), 2),
        "DeltaNRMSE": format_value(float(d_tta["field_nrmse_db"] - d0["field_nrmse_db"]), 2),
        # Extra: network-only proposed vs no_phys (training-only physics effect).
        "EMA1_netonly": format_value(float(d_net["ema_pct"]), 2),
        "A1_netonly": format_value(float(d_net["amp_mae_fault"]), 4),
        # TTPR inversion-accuracy gain over network-only.
        "TTPR_amp_gain_pct": format_value(100.0 * (d_net["amp_mae_fault"] - d_tta["amp_mae_fault"]) / max(d_net["amp_mae_fault"], 1e-9), 1),
        "TTPR_phase_gain_pct": format_value(100.0 * (d_net["phase_mae_deg_masked"] - d_tta["phase_mae_deg_masked"]) / max(d_net["phase_mae_deg_masked"], 1e-9), 1),
    }
    return df, values


def run_noise_distribution_table(
    model: PI1DResNet,
    cfg: Config,
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rows = []
    values: Dict[str, str] = {}
    for kind in cfg.noise_kinds:
        res = evaluate_model_on_dataset(
            model,
            cfg,
            A_Etheta,
            A_Ephi,
            K,
            device,
            scenario="mixed",
            sampler=cfg.eval_sampler,
            target_snr_db=cfg.nominal_eval_snr_db,
            noise_kind=kind,
            decision_mode="fault_head",
            base_seed=cfg.seed + 700_000,
            use_tta=True,
        )["metrics"]
        pretty = {"gaussian": "Gaussian", "uniform": "Uniform", "laplacian": "Laplacian"}[kind]
        rows.append({"Additive noise model": pretty, "S-E F1 (%)": res["f1_pct"], "EMA (%)": res["ema_pct"]})
        suffix = {"gaussian": "1", "uniform": "u", "laplacian": "l"}[kind]
        values[f"F1{suffix}"] = format_value(res["f1_pct"], 2)
        values[f"EMA{suffix}"] = format_value(res["ema_pct"], 2)
    return pd.DataFrame(rows), values


def run_snr_sweep_table(
    model: PI1DResNet,
    cfg: Config,
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for snr in cfg.snr_sweep_db:
        res = evaluate_model_on_dataset(
            model,
            cfg,
            A_Etheta,
            A_Ephi,
            K,
            device,
            scenario="mixed",
            sampler=cfg.eval_sampler,
            target_snr_db=float(snr),
            noise_kind="gaussian",
            decision_mode="fault_head",
            base_seed=cfg.seed + 800_000 + int(snr * 10),
            use_tta=True,
        )["metrics"]
        rows.append({"SNR (dB)": float(snr), "S-E F1 (%)": res["f1_pct"], "EMA (%)": res["ema_pct"], "NRMSE (dB)": res["field_nrmse_db"]})
    return pd.DataFrame(rows)


def run_fault_decision_table(
    models: Dict[str, PI1DResNet],
    cfgs: Dict[str, Config],
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    # Same predictions for proposed, different decision modes.  We report both
    # network-only and TTPR-assisted decisions so the role of the learned fault
    # head and the refined amplitude/phase rule are separated.
    for use_tta, tag in [(False, "network"), (True, "TTPR")]:
        for mode in ["fault_head", "reg_threshold", "hybrid_or", "hybrid_and"]:
            res = evaluate_model_on_dataset(
                models["proposed"],
                cfgs["proposed"],
                A_Etheta,
                A_Ephi,
                K,
                device,
                scenario="mixed",
                sampler=cfgs["proposed"].eval_sampler,
                target_snr_db=cfgs["proposed"].nominal_eval_snr_db,
                noise_kind="gaussian",
                decision_mode=mode,
                base_seed=cfgs["proposed"].seed + 900_000,
                use_tta=use_tta,
            )["metrics"]
            rows.append({"Model / decision": f"Proposed-{tag} / {mode}", "S-E F1 (%)": res["f1_pct"], "EMA (%)": res["ema_pct"], "False alarm (%)": res["false_alarm_rate_pct"], "Miss detection (%)": res["miss_detection_rate_pct"]})

    if "no_fault_head" in models:
        res = evaluate_model_on_dataset(
            models["no_fault_head"],
            cfgs["no_fault_head"],
            A_Etheta,
            A_Ephi,
            K,
            device,
            scenario="mixed",
            sampler=cfgs["no_fault_head"].eval_sampler,
            target_snr_db=cfgs["no_fault_head"].nominal_eval_snr_db,
            noise_kind="gaussian",
            decision_mode="reg_threshold",
            base_seed=cfgs["no_fault_head"].seed + 900_000,
            use_tta=True,
        )["metrics"]
        rows.append({"Model / decision": r"$\lambda_{\mathrm{BCE}}=0$ / TTPR+reg_threshold", "S-E F1 (%)": res["f1_pct"], "EMA (%)": res["ema_pct"], "False alarm (%)": res["false_alarm_rate_pct"], "Miss detection (%)": res["miss_detection_rate_pct"]})
    return pd.DataFrame(rows)

def run_threshold_sensitivity_table(
    model: PI1DResNet,
    cfg: Config,
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
) -> pd.DataFrame:
    # Use one prediction bank, then recompute labels/decisions over threshold grid.
    ds = SyntheticPhasedArrayDataset(
        cfg,
        cfg.eval_samples,
        base_seed=cfg.seed + 1_000_000,
        A_Etheta=A_Etheta,
        A_Ephi=A_Ephi,
        K=K,
        scenario="mixed",
        sampler=cfg.eval_sampler,
        target_snr_db=cfg.nominal_eval_snr_db,
        noise_kind="gaussian",
    )
    loader = build_loader(cfg, ds, shuffle=False, device=device, seed=cfg.seed + 1_000_100)
    pred = predict_batches(model, loader, cfg, device)
    rows = []
    for a_th in cfg.amp_threshold_grid:
        for p_th in cfg.phase_threshold_grid_deg:
            true_bin = diagnose_rule_np(pred["true_amp"], pred["true_phase_deg"], a_th, p_th)
            pred_bin = diagnose_rule_np(pred["pred_amp"], pred["pred_phase_deg"], a_th, p_th)
            m = compute_metrics_from_arrays(
                true_bin,
                pred_bin,
                pred["true_amp"],
                pred["true_phase_deg"],
                pred["pred_amp"],
                pred["pred_phase_deg"],
                pred["phys_residual"],
                cfg,
            )
            rows.append({"a_th": a_th, "phi_th_deg": p_th, "S-E F1 (%)": m["f1_pct"], "EMA (%)": m["ema_pct"], "False alarm (%)": m["false_alarm_rate_pct"], "Miss detection (%)": m["miss_detection_rate_pct"]})
    return pd.DataFrame(rows)


def run_aep_mismatch_table(
    model: PI1DResNet,
    cfg: Config,
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for amp_sigma, phase_sigma in cfg.aep_mismatch_levels:
        rng = np.random.default_rng(cfg.seed + 1_100_000 + int(amp_sigma * 1000) + int(phase_sigma * 10))
        A_th_m = perturb_aep(A_Etheta, rng, amp_sigma, phase_sigma)
        A_ph_m = perturb_aep(A_Ephi, rng, amp_sigma, phase_sigma)
        res = evaluate_model_on_dataset(
            model,
            cfg,
            A_th_m,
            A_ph_m,
            K,
            device,
            scenario="mixed",
            sampler=cfg.eval_sampler,
            target_snr_db=cfg.nominal_eval_snr_db,
            noise_kind="gaussian",
            decision_mode="fault_head",
            base_seed=cfg.seed + 1_200_000,
            use_tta=True,
        )["metrics"]
        rows.append({"AEP amp. mismatch std": amp_sigma, "AEP phase mismatch std (deg)": phase_sigma, "S-E F1 (%)": res["f1_pct"], "EMA (%)": res["ema_pct"], "NRMSE (dB)": res["field_nrmse_db"]})
    return pd.DataFrame(rows)


def run_scenario_table(
    model: PI1DResNet,
    cfg: Config,
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for scenario in ["amp_only", "phase_only", "mixed"]:
        res = evaluate_model_on_dataset(
            model,
            cfg,
            A_Etheta,
            A_Ephi,
            K,
            device,
            scenario=scenario,
            sampler=cfg.eval_sampler,
            target_snr_db=cfg.nominal_eval_snr_db,
            noise_kind="gaussian",
            decision_mode="fault_head",
            base_seed=cfg.seed + 1_300_000 + {"amp_only": 1, "phase_only": 2, "mixed": 3}[scenario],
            use_tta=True,
        )["metrics"]
        rows.append({"Scenario": scenario, "S-E Precision (%)": res["precision_pct"], "S-E Recall (%)": res["recall_pct"], "S-E F1 (%)": res["f1_pct"], "EMA (%)": res["ema_pct"]})
    return pd.DataFrame(rows)


def run_phase_mask_report(
    model: PI1DResNet,
    cfg: Config,
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
) -> pd.DataFrame:
    res = evaluate_model_on_dataset(
        model,
        cfg,
        A_Etheta,
        A_Ephi,
        K,
        device,
        scenario="mixed",
        sampler=cfg.eval_sampler,
        target_snr_db=cfg.nominal_eval_snr_db,
        noise_kind="gaussian",
        decision_mode="fault_head",
        base_seed=cfg.seed + 1_400_000,
        use_tta=True,
    )["metrics"]
    return pd.DataFrame([
        {"Statistic": f"Phase MAE over all true faulty elements", "Value": res["phase_mae_deg_unmasked_fault"], "Unit": "deg"},
        {"Statistic": f"Phase MAE after masking true_amp < {cfg.phase_amp_mask_threshold}", "Value": res["phase_mae_deg_masked"], "Unit": "deg"},
        {"Statistic": "Amplitude MAE over true faulty elements", "Value": res["amp_mae_fault"], "Unit": "linear"},
    ])


def make_label_free_noise_landscape(
    cfg: Config,
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    out_dir: Path,
) -> Dict[str, float]:

    rng = np.random.default_rng(cfg.seed)
    N = cfg.N
    A_th = A_Etheta.astype(np.complex128)
    A_ph = A_Ephi.astype(np.complex128)
    amp = rng.uniform(cfg.healthy_amp_min, cfg.healthy_amp_max, N)
    ph_deg = rng.uniform(cfg.healthy_phase_min_deg, cfg.healthy_phase_max_deg, N)
    amp[6] = 0.35
    ph_deg[12] = 95.0
    amp[18], ph_deg[18] = 0.60, -140.0
    w_true = amp * np.exp(1j * np.deg2rad(ph_deg))

    E_th_clean = A_th @ w_true
    E_ph_clean = A_ph @ w_true
    feat_clean = build_feature(E_th_clean, E_ph_clean).astype(np.float64)
    sig_pow = float(np.mean(feat_clean ** 2))
    sigma = math.sqrt(sig_pow / (10.0 ** (cfg.nominal_eval_snr_db / 10.0)))
    noise_floor = 1.0 / (10.0 ** (cfg.nominal_eval_snr_db / 10.0) + 1.0)

    a_grid = np.linspace(0.0, cfg.amp_max_value, 221)
    p_grid = np.linspace(-180.0, 180.0, 361)
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    truth_residuals: Dict[str, float] = {}

    def residual_from_fields(Eth: np.ndarray, Eph: np.ndarray, X: np.ndarray, Xpow: float) -> np.ndarray:
        # Eth/Eph can be [M,K] or [K]. Return residual per row.
        if Eth.ndim == 1:
            f = build_feature(Eth, Eph).astype(np.float64)
            return np.array([float(np.mean((f - X) ** 2) / max(Xpow, 1e-12))])
        feats = np.concatenate([Eth.real, Eth.imag, Eph.real, Eph.imag], axis=1).astype(np.float64)
        return np.mean((feats - X[None, :]) ** 2, axis=1) / max(Xpow, 1e-12)

    for i, kind in enumerate(cfg.noise_kinds):
        rng_k = np.random.default_rng(cfg.seed + 10_000 * (i + 1))
        X = feat_clean + draw_real_noise(kind, rng_k, feat_clean.shape, sigma)
        Xpow = float(np.mean(X ** 2))
        truth_residuals[kind] = float(residual_from_fields(E_th_clean, E_ph_clean, X, Xpow)[0])

        # Amplitude sweep of element index 6.
        w7_values = a_grid * np.exp(1j * np.deg2rad(ph_deg[6]))
        delta7 = w7_values - w_true[6]
        Eth_amp = E_th_clean[None, :] + delta7[:, None] * A_th[:, 6][None, :]
        Eph_amp = E_ph_clean[None, :] + delta7[:, None] * A_ph[:, 6][None, :]
        amp_curve = residual_from_fields(Eth_amp, Eph_amp, X, Xpow)

        # Phase sweep of element index 12.
        w13_values = amp[12] * np.exp(1j * np.deg2rad(p_grid))
        delta13 = w13_values - w_true[12]
        Eth_phase = E_th_clean[None, :] + delta13[:, None] * A_th[:, 12][None, :]
        Eph_phase = E_ph_clean[None, :] + delta13[:, None] * A_ph[:, 12][None, :]
        phase_curve = residual_from_fields(Eth_phase, Eph_phase, X, Xpow)
        curves[kind] = {"amp": amp_curve, "phase": phase_curve}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for kind in cfg.noise_kinds:
        label = kind.capitalize()
        axes[0].plot(a_grid, curves[kind]["amp"], label=label)
        axes[1].plot(p_grid, curves[kind]["phase"], label=label)
    axes[0].axvline(amp[6], linestyle=":")
    axes[1].axvline(ph_deg[12], linestyle=":")
    axes[0].axhline(noise_floor, linestyle=":")
    axes[1].axhline(noise_floor, linestyle=":")
    axes[0].set_xlabel(r"swept amplitude $a_7$ of element 7")
    axes[1].set_xlabel(r"swept phase $\varphi_{13}$ of element 13 (deg)")
    axes[0].set_ylabel(r"$r_{\mathrm{phys}}$")
    axes[0].set_title("(a) amplitude sweep")
    axes[1].set_title("(b) phase sweep")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "Fig_R1_2_complete.pdf", dpi=300)
    fig.savefig(out_dir / "Fig_R1_2_complete.png", dpi=300)
    plt.close(fig)

    amp_stack = np.stack([curves[k]["amp"] for k in cfg.noise_kinds], axis=0)
    phase_stack = np.stack([curves[k]["phase"] for k in cfg.noise_kinds], axis=0)
    return {
        "noise_floor": noise_floor,
        "truth_residual_gaussian": truth_residuals.get("gaussian", float("nan")),
        "truth_residual_uniform": truth_residuals.get("uniform", float("nan")),
        "truth_residual_laplacian": truth_residuals.get("laplacian", float("nan")),
        "amp_curve_max_relative_spread_pct": float(100.0 * np.max(np.ptp(amp_stack, axis=0) / np.maximum(np.mean(amp_stack, axis=0), 1e-12))),
        "phase_curve_max_relative_spread_pct": float(100.0 * np.max(np.ptp(phase_stack, axis=0) / np.maximum(np.mean(phase_stack, axis=0), 1e-12))),
    }


# =============================================================================
# 7. Output helpers
# =============================================================================


def save_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str, label: str) -> None:
    df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    df.to_excel(out_dir / f"{name}.xlsx", index=False)
    (out_dir / f"{name}.tex").write_text(
        dataframe_to_latex_table(df, caption=caption, label=label),
        encoding="utf-8",
    )


def save_all_results_excel(tables: Dict[str, pd.DataFrame], out_path: Path) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)


def make_response_replacement_file(values: Dict[str, str], out_dir: Path) -> None:
    lines = [
        "% Auto-generated by unified_pipeline.py",
        "% Copy these values into Response_R1_C1_C2.tex placeholders.",
    ]
    for k in sorted(values):
        lines.append(f"\\newcommand{{\\{k}}}{{{values[k]}}}")
    (out_dir / "response_placeholder_values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


    plh_map = [
        ("Table R1-I (physics ablation, lambda_phys = 0 row)", [
            (r"\PLH{F1$_0$}", values.get("F10", "?")),
            (r"\PLH{EMA$_0$}", values.get("EMA0", "?")),
            (r"\PLH{A$_0$}", values.get("A0", "?")),
            (r"\PLH{P$_0$}", values.get("P0", "?")),
            (r"\PLH{N$_0$}", values.get("N0", "?")),
        ]),
        ("Table R1-I (physics ablation, lambda_phys = 1 row)", [
            (r"\PLH{F1$_1$}", values.get("F11", "?")),
            (r"\PLH{EMA$_1$}", values.get("EMA1", "?")),
            (r"\PLH{A$_1$}", values.get("A1", "?")),
            (r"\PLH{P$_1$}", values.get("P1", "?")),
            (r"\PLH{N$_1$}", values.get("N1", "?")),
        ]),
        ("Response 1 Part (6) summary sentence", [
            (r"\PLH{$\Delta$EMA}", values.get("DeltaEMA", "?")),
            (r"\PLH{$\Delta$NRMSE}", values.get("DeltaNRMSE", "?")),
        ]),
        ("Table R1-II (noise distribution)", [
            (r"\PLH{F1$_1$} (Gaussian)", values.get("F11", "?")),
            (r"\PLH{EMA$_1$} (Gaussian)", values.get("EMA1", "?")),
            (r"\PLH{F1$_u$} (Uniform)", values.get("F1u", "?")),
            (r"\PLH{EMA$_u$} (Uniform)", values.get("EMAu", "?")),
            (r"\PLH{F1$_l$} (Laplacian)", values.get("F1l", "?")),
            (r"\PLH{EMA$_l$} (Laplacian)", values.get("EMAl", "?")),
        ]),
        ("Response 2 Part (5) label-free landscape (Fig. R1-2 text)", [
            ("r_phys(Gaussian)", values.get("RphysGaussian", "?")),
            ("r_phys(Uniform)", values.get("RphysUniform", "?")),
            ("r_phys(Laplacian)", values.get("RphysLaplacian", "?")),
            ("AWGN noise floor", values.get("NoiseFloor", "?")),
            ("amplitude-sweep max spread (%)", values.get("AmpSpread", "?")),
            ("phase-sweep max spread (%)", values.get("PhaseSpread", "?")),
        ]),
    ]
    md = ["# Response-letter placeholder fill-in map",
          "",
          "Replace each `\\PLH{...}` in Response_R1_C1_C2.tex with the value below.",
          "(Values are from the full unified run; see the cited tables for context.)",
          ""]
    for section, items in plh_map:
        md.append(f"## {section}")
        for ph, val in items:
            md.append(f"- `{ph}`  ->  **{val}**")
        md.append("")
    (out_dir / "response_placeholder_fill_map.md").write_text("\n".join(md), encoding="utf-8")

    (out_dir / "response_placeholder_values.json").write_text(
        json.dumps(values, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def run_unified_benchmark_table(
    models: Dict[str, Any],
    cfgs: Dict[str, Config],
    cgan_generator: Optional[Any],
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
    training_times: Dict[str, float],
    cnn_cfar_model: Optional[Any] = None,
) -> pd.DataFrame:
    """Table II: accuracy + efficiency of all methods at the nominal SNR."""
    cfg = cfgs["proposed"]
    snr = cfg.nominal_eval_snr_db
    rows: List[Dict[str, Any]] = []

    # Learning baselines: use the SAME evaluation bank for every method.
    common_eval_seed = cfg.seed + 510_000
    if "ann_mlp" in models:
        rows.append(_learning_row("ANN-MLP", "ML", models["ann_mlp"], cfgs["ann_mlp"], A_Etheta, A_Ephi, K, device, snr, common_eval_seed, training_times.get("ann_mlp", float("nan"))))
    if "dnn_mlp" in models:
        rows.append(_learning_row("DNN-MLP", "ML", models["dnn_mlp"], cfgs["dnn_mlp"], A_Etheta, A_Ephi, K, device, snr, common_eval_seed, training_times.get("dnn_mlp", float("nan"))))
    if cgan_generator is not None:
        rows.append(_cgan_row("Adapted-cGAN", cgan_generator, cfgs["proposed"], A_Etheta, A_Ephi, K, device, snr, common_eval_seed, training_times.get("adapted_cgan", float("nan"))))
    # SOTA: adapted CNN-inversion + CFAR (Schenone et al., IEEE TAP 2025).
    if cnn_cfar_model is not None and "_CNN_CFAR_MODULE" in globals():
        cc = globals()["_CNN_CFAR_MODULE"]
        res = cc.evaluate_cnn_cfar_method(cnn_cfar_model, cfg, A_Etheta, A_Ephi, K, device,
                                          target_snr_db=snr, gamma=cfg.cnn_cfar_gamma, base_seed=common_eval_seed)
        m = res["metrics"]
        rows.append({"Method": "CNN-CFAR [TAP'25]", "Category": "SOTA",
                     "Train Time (s)": training_times.get("cnn_cfar", float("nan")),
                     "Infer (ms/sample)": m.get("Infer (ms/sample)", float("nan")),
                     "S-E F1 (%)": m["f1_pct"], "EMA (%)": m["ema_pct"]})

    # CS baselines at each observation setting
    for method in cfg.cs_methods:
        for M_obs in cfg.cs_M_obs_list:
            res = evaluate_cs_method(method, cfg, A_Etheta, A_Ephi, K, M_obs=M_obs,
                                     omp_k=cfg.cs_omp_k, lasso_alpha=cfg.cs_lasso_alpha,
                                     base_seed=common_eval_seed)
            m = res["metrics"]
            tag = f"{method} (M={m['M_obs']})"
            rows.append({"Method": tag, "Category": "CS", "Train Time (s)": float("nan"),
                         "Infer (ms/sample)": m["infer_time_ms_per_sample"],
                         "S-E F1 (%)": m["f1_pct"], "EMA (%)": m["ema_pct"]})

    # Proposed variants
    if "data_driven_no_phys" in models:
        rows.append(_learning_row("1D-ResNet (no phys)", "PI-ML", models["data_driven_no_phys"], cfgs["data_driven_no_phys"], A_Etheta, A_Ephi, K, device, snr, common_eval_seed, training_times.get("data_driven_no_phys", float("nan"))))
    rows.append(_learning_row("PI-1D-ResNet (proposed)", "PI-ML", models["proposed"], cfgs["proposed"], A_Etheta, A_Ephi, K, device, snr, common_eval_seed, training_times.get("proposed", float("nan")), use_tta=True))

    return pd.DataFrame(rows)


def _learning_row(label, category, model, cfg, A_Etheta, A_Ephi, K, device, snr, base_seed, ttime, use_tta=False):
    m = evaluate_model_on_dataset(
        model, cfg, A_Etheta, A_Ephi, K, device, scenario="mixed", sampler=cfg.eval_sampler,
        target_snr_db=float(snr), noise_kind="gaussian", decision_mode="fault_head", base_seed=base_seed,
        use_tta=use_tta,
    )["metrics"]
    return {"Method": label, "Category": category, "Train Time (s)": ttime,
            "Infer (ms/sample)": m.get("infer_time_ms_per_sample", float("nan")),
            "S-E F1 (%)": m["f1_pct"], "EMA (%)": m["ema_pct"]}


def _cgan_row(label, generator, cfg, A_Etheta, A_Ephi, K, device, snr, base_seed, ttime):
    ds = SyntheticPhasedArrayDataset(
        cfg, cfg.eval_samples, base_seed=base_seed, A_Etheta=A_Etheta, A_Ephi=A_Ephi, K=K,
        scenario="mixed", sampler=cfg.eval_sampler, target_snr_db=float(snr), noise_kind="gaussian",
    )
    loader = build_loader(cfg, ds, shuffle=False, device=device, seed=base_seed + 7)
    t0 = time.perf_counter()
    pred = predict_batches_cgan(generator, loader, cfg, device, avg_seeds=cfg.cgan_infer_avg_seeds)
    infer_ms = 1000.0 * (time.perf_counter() - t0) / max(pred["pred_amp"].shape[0], 1)
    m = compute_metrics_from_arrays(
        pred["true_binary"], pred["pred_binary_head"], pred["true_amp"], pred["true_phase_deg"],
        pred["pred_amp"], pred["pred_phase_deg"], pred["phys_residual"], cfg,
    )
    return {"Method": label, "Category": "ML", "Train Time (s)": ttime,
            "Infer (ms/sample)": infer_ms, "S-E F1 (%)": m["f1_pct"], "EMA (%)": m["ema_pct"]}


def run_unified_snr_sweep(
    models: Dict[str, Any],
    cfgs: Dict[str, Config],
    cgan_generator: Optional[Any],
    A_Etheta: np.ndarray,
    A_Ephi: np.ndarray,
    K: int,
    device: torch.device,
    cnn_cfar_model: Optional[Any] = None,
) -> pd.DataFrame:

    cfg = cfgs["proposed"]
    rows: List[Dict[str, Any]] = []
    for snr in cfg.snr_sweep_db:
        row: Dict[str, Any] = {"SNR (dB)": float(snr)}
        common_seed = cfg.seed + 610_000 + int(snr * 10)
        if "ann_mlp" in models:
            row["ANN-MLP"] = _learning_row("ANN-MLP", "ML", models["ann_mlp"], cfgs["ann_mlp"], A_Etheta, A_Ephi, K, device, snr, common_seed, float("nan"))["EMA (%)"]
        if "dnn_mlp" in models:
            row["DNN-MLP"] = _learning_row("DNN-MLP", "ML", models["dnn_mlp"], cfgs["dnn_mlp"], A_Etheta, A_Ephi, K, device, snr, common_seed, float("nan"))["EMA (%)"]
        if cgan_generator is not None:
            row["Adapted-cGAN"] = _cgan_row("Adapted-cGAN", cgan_generator, cfg, A_Etheta, A_Ephi, K, device, snr, common_seed, float("nan"))["EMA (%)"]
        if cnn_cfar_model is not None and "_CNN_CFAR_MODULE" in globals():
            cc = globals()["_CNN_CFAR_MODULE"]
            res_cc = cc.evaluate_cnn_cfar_method(cnn_cfar_model, cfg, A_Etheta, A_Ephi, K, device,
                                                 target_snr_db=snr, gamma=cfg.cnn_cfar_gamma,
                                                 base_seed=common_seed,
                                                 eval_samples=max(1000, cfg.eval_samples // 5))
            row["CNN-CFAR [TAP'25]"] = res_cc["metrics"]["ema_pct"]
        for method in cfg.cs_methods:
            res = evaluate_cs_method(method, cfg, A_Etheta, A_Ephi, K, M_obs=min(50, K),
                                     omp_k=cfg.cs_omp_k, lasso_alpha=cfg.cs_lasso_alpha,
                                     base_seed=common_seed,
                                     eval_samples=max(1000, cfg.eval_samples // 5))
            row[method] = res["metrics"]["ema_pct"]
        if "data_driven_no_phys" in models:
            row["1D-ResNet (no phys)"] = _learning_row("x", "PI-ML", models["data_driven_no_phys"], cfgs["data_driven_no_phys"], A_Etheta, A_Ephi, K, device, snr, common_seed, float("nan"))["EMA (%)"]
        row["PI-1D-ResNet"] = _learning_row("x", "PI-ML", models["proposed"], cfgs["proposed"], A_Etheta, A_Ephi, K, device, snr, common_seed, float("nan"), use_tta=True)["EMA (%)"]
        rows.append(row)
    return pd.DataFrame(rows)




# =============================================================================
# 8. Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete PI-1D-ResNet reviewer-response pipeline")
    parser.add_argument("--aep_mat_path", type=str, default=None, help="Path to AEP_Matrix.mat")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Small sanity-check run; not for manuscript numbers")
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--val_samples", type=int, default=None)
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--skip_no_bce_ablation", action="store_true", help="Skip lambda_BCE=0 ablation to save time")
    return parser.parse_args()


def main() -> None:

    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass
    args = parse_args()
    cfg = Config()
    if args.aep_mat_path is not None:
        cfg.aep_mat_path = args.aep_mat_path
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.quick:
        cfg.train_samples = 512
        cfg.val_samples = 128
        cfg.eval_samples = 128
        cfg.epochs = 2
        cfg.patience = 2
        cfg.batch_size = 64
        cfg.num_workers = 0
        cfg.persistent_workers = False
        cfg.train_no_bce_ablation = False
        cfg.snr_sweep_db = (25.0, 5.0, 0.0)
        cfg.aep_mismatch_levels = ((0.0, 0.0), (0.05, 5.0))
        cfg.cgan_infer_avg_seeds = 2
        cfg.tta_iters = 3
        cfg.run_cnn_cfar_baseline = False
    if args.train_samples is not None:
        cfg.train_samples = args.train_samples
    if args.val_samples is not None:
        cfg.val_samples = args.val_samples
    if args.eval_samples is not None:
        cfg.eval_samples = args.eval_samples
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.skip_no_bce_ablation:
        cfg.train_no_bce_ablation = False

    out_dir = Path(cfg.save_dir)
    ensure_dir(out_dir)
    seed_everything(cfg.seed, cfg.deterministic_cudnn)

    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.use_gpu_if_available and not args.cpu) else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(cfg.cpu_num_threads)))
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    A_Etheta, A_Ephi, K = load_aep_matrix(cfg.aep_mat_path, cfg)
    input_dim = 4 * K
    print(f"Loaded AEP matrix: K={K}, N={cfg.N}, input_dim={input_dim}")


    cfg.cs_M_obs_list = (min(50, K), K)
    print(f"CS observation settings (M_obs): {cfg.cs_M_obs_list}")

    # Save exact config.
    (out_dir / "config_complete.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    # Variant configs. Same initialization/data/schedule; only specified loss weights change.
    cfg_proposed = copy.deepcopy(cfg)
    cfg_proposed.lambda_phys = cfg.lambda_phys
    cfg_proposed.lambda_bce = cfg.lambda_bce

    cfg_no_phys = copy.deepcopy(cfg)
    cfg_no_phys.lambda_phys = 0.0
    cfg_no_phys.lambda_bce = cfg.lambda_bce

    cfg_no_bce = copy.deepcopy(cfg)
    cfg_no_bce.lambda_phys = cfg.lambda_phys
    cfg_no_bce.lambda_bce = 0.0

    variants: List[Tuple[str, Config]] = [("data_driven_no_phys", cfg_no_phys), ("proposed", cfg_proposed)]
    if cfg.train_no_bce_ablation:
        variants.append(("no_fault_head", cfg_no_bce))

    models: Dict[str, PI1DResNet] = {}
    cfgs: Dict[str, Config] = {}
    training_summaries: List[Dict[str, Any]] = []

    for name, c in variants:
        # Same initial seed for a clean controlled comparison.
        model, summary = train_model_variant(c, name, input_dim, A_Etheta, A_Ephi, K, device, out_dir)
        models[name] = model
        cfgs[name] = c
        training_summaries.append({k: v for k, v in summary.items() if k != "history"})

    # ----- Unified, strictly fair learning baselines (R2-C4 / R3-C1) -----
    training_times: Dict[str, float] = {s["variant"]: s["train_time_sec"] for s in training_summaries}
    cgan_generator = None
    if cfg.run_learning_baselines:
        ann = ANNMLP(input_dim, cfg.N, hidden=cfg.ann_hidden)
        ann, s_ann = train_standard_model(ann, cfg, "ann_mlp", input_dim, A_Etheta, A_Ephi, K, device, out_dir)
        models["ann_mlp"], cfgs["ann_mlp"] = ann, replace(cfg, lambda_phys=0.0)
        training_summaries.append({k: v for k, v in s_ann.items() if k != "history"})
        training_times["ann_mlp"] = s_ann["train_time_sec"]

        dnn = DNNMLP(input_dim, cfg.N, hidden=cfg.dnn_hidden, depth=cfg.dnn_depth, dropout=cfg.dnn_dropout)
        dnn, s_dnn = train_standard_model(dnn, cfg, "dnn_mlp", input_dim, A_Etheta, A_Ephi, K, device, out_dir)
        models["dnn_mlp"], cfgs["dnn_mlp"] = dnn, replace(cfg, lambda_phys=0.0)
        training_summaries.append({k: v for k, v in s_dnn.items() if k != "history"})
        training_times["dnn_mlp"] = s_dnn["train_time_sec"]

        cgan_generator, s_cgan = train_adapted_cgan(cfg, input_dim, A_Etheta, A_Ephi, K, device, out_dir)
        training_summaries.append({k: v for k, v in s_cgan.items() if k != "history"})
        training_times["adapted_cgan"] = s_cgan["train_time_sec"]


    cnn_cfar_model = None
    if cfg.run_learning_baselines and cfg.run_cnn_cfar_baseline:
        try:
            import importlib
            cc = importlib.import_module("CNN_CFAR_unified_final622")
            cnn_cfar_model, s_cc = cc.train_cnn_cfar(cfg, input_dim, A_Etheta, A_Ephi, K, device, out_dir)
            training_summaries.append({k: v for k, v in s_cc.items() if k != "history"})
            training_times["cnn_cfar"] = s_cc["train_time_sec"]
            globals()["_CNN_CFAR_MODULE"] = cc
        except Exception as exc:  # pragma: no cover
            print(f"[CNN-CFAR baseline skipped] {type(exc).__name__}: {exc}")
            cnn_cfar_model = None

    pd.DataFrame(training_summaries).to_csv(out_dir / "training_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== Running reviewer-response analyses ===")
    tables: Dict[str, pd.DataFrame] = {}
    response_values: Dict[str, str] = {}

    table_physics, values_physics = run_physics_ablation_table(models, cfgs, A_Etheta, A_Ephi, K, device)
    tables["R1I_physics_ablation"] = table_physics
    response_values.update(values_physics)
    save_table(
        table_physics,
        out_dir,
        "Table_R1_I_physics_ablation",
        caption="Isolated contribution of the AEP-based physics consistency loss under mixed impairments.",
        label="tab:R1_I_physics_ablation",
    )

    table_noise, values_noise = run_noise_distribution_table(models["proposed"], cfgs["proposed"], A_Etheta, A_Ephi, K, device)
    tables["R1II_noise_distribution"] = table_noise
    response_values.update(values_noise)
    save_table(
        table_noise,
        out_dir,
        "Table_R1_II_noise_distribution",
        caption="Trained-network sensitivity to additive-noise distribution shape at matched 25 dB SNR.",
        label="tab:R1_II_noise_distribution",
    )

    table_snr = run_snr_sweep_table(models["proposed"], cfgs["proposed"], A_Etheta, A_Ephi, K, device)
    tables["R1III_snr_sweep"] = table_snr
    save_table(table_snr, out_dir, "Table_R1_III_snr_sweep", caption="Low-SNR robustness sweep of the proposed PI-1D-ResNet.", label="tab:R1_III_snr_sweep")

    # ----- Unified 8-method benchmark (Table II) and SNR sweep (Fig. 3) -----
    if cfg.run_learning_baselines and cfg.run_cs_baselines:
        table_bench = run_unified_benchmark_table(models, cfgs, cgan_generator, A_Etheta, A_Ephi, K, device, training_times, cnn_cfar_model=cnn_cfar_model)
        tables["TableII_unified_benchmark"] = table_bench
        save_table(table_bench, out_dir, "Table_II_unified_benchmark",
                   caption="Unified, equal-condition comparison of diagnostic accuracy and efficiency under mixed impairments. All learning methods share identical training data, SNR distribution, decoding, and fault threshold; CS is reported at both the conventional (M=50) and equal-observation (M=K) settings.",
                   label="tab:unified_benchmark")
        table_snr8 = run_unified_snr_sweep(models, cfgs, cgan_generator, A_Etheta, A_Ephi, K, device, cnn_cfar_model=cnn_cfar_model)
        tables["Fig3_unified_snr_sweep"] = table_snr8
        save_table(table_snr8, out_dir, "Fig_3_unified_snr_sweep",
                   caption="EMA versus SNR for all methods under mixed impairments (unified pipeline).",
                   label="tab:unified_snr_sweep")

    table_fault = run_fault_decision_table(models, cfgs, A_Etheta, A_Ephi, K, device)
    tables["R2_fault_decision"] = table_fault
    save_table(table_fault, out_dir, "Table_R2_fault_decision", caption="Fault-head and regression-threshold decision comparison.", label="tab:R2_fault_decision")

    table_threshold = run_threshold_sensitivity_table(models["proposed"], cfgs["proposed"], A_Etheta, A_Ephi, K, device)
    tables["R2_threshold_sensitivity"] = table_threshold
    save_table(table_threshold, out_dir, "Table_R2_threshold_sensitivity", caption="Engineering-threshold sensitivity of regression-derived fault decisions.", label="tab:R2_threshold_sensitivity")

    table_mismatch = run_aep_mismatch_table(models["proposed"], cfgs["proposed"], A_Etheta, A_Ephi, K, device)
    tables["R2_AEP_mismatch"] = table_mismatch
    save_table(table_mismatch, out_dir, "Table_R2_AEP_mismatch", caption="AEP-mismatch robustness of the proposed PI-1D-ResNet.", label="tab:R2_AEP_mismatch")

    table_scenarios = run_scenario_table(models["proposed"], cfgs["proposed"], A_Etheta, A_Ephi, K, device)
    tables["Scenario_performance"] = table_scenarios
    save_table(table_scenarios, out_dir, "Table_scenario_performance", caption="Diagnosis performance for different impairment scenarios.", label="tab:scenario_performance")

    table_phase_mask = run_phase_mask_report(models["proposed"], cfgs["proposed"], A_Etheta, A_Ephi, K, device)
    tables["R2_phase_mask_report"] = table_phase_mask
    save_table(table_phase_mask, out_dir, "Table_R2_phase_mask_report", caption="Effect of excluding weakly observable low-amplitude elements from phase-error statistics.", label="tab:R2_phase_mask_report")

    label_free = make_label_free_noise_landscape(cfgs["proposed"], A_Etheta, A_Ephi, out_dir)
    response_values.update({
        "RphysGaussian": format_value(label_free["truth_residual_gaussian"], 4),
        "RphysUniform": format_value(label_free["truth_residual_uniform"], 4),
        "RphysLaplacian": format_value(label_free["truth_residual_laplacian"], 4),
        "NoiseFloor": format_value(label_free["noise_floor"], 4),
        "AmpSpread": format_value(label_free["amp_curve_max_relative_spread_pct"], 2),
        "PhaseSpread": format_value(label_free["phase_curve_max_relative_spread_pct"], 2),
    })

    save_all_results_excel(tables, out_dir / "PI_complete_all_tables.xlsx")
    make_response_replacement_file(response_values, out_dir)
    sio.savemat(
        out_dir / "PI_complete_results.mat",
        {name: df.select_dtypes(include=[np.number]).to_numpy() for name, df in tables.items()},
    )

    print("\n=== Finished ===")
    print(f"Outputs saved to: {out_dir.resolve()}")
    print("Key files:")
    for fn in [
        "Table_R1_I_physics_ablation.tex",
        "Table_R1_II_noise_distribution.tex",
        "Table_R1_III_snr_sweep.tex",
        "Table_R2_fault_decision.tex",
        "Table_R2_threshold_sensitivity.tex",
        "Table_R2_AEP_mismatch.tex",
        "response_placeholder_values.tex",
        "response_placeholder_values.json",
        "PI_complete_all_tables.xlsx",
        "Fig_R1_2_complete.pdf",
    ]:
        print("  -", out_dir / fn)


if __name__ == "__main__":
    main()
