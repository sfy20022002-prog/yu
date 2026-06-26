"""
Adapted-cGAN baseline -- UNIFIED launcher (resubmission of AWPL-04-26-1519).

Imports everything from `unified_pipeline.py`, so the adapted-cGAN baseline is
trained on EXACTLY the same data distribution, per-sample training SNR,
decoding, class-imbalance-weighted BCE, optimizer/scheduler/early-stopping rule,
seed, and initialization as the proposed PI-1D-ResNet and the other baselines.

Two fairness-relevant points are handled inside `unified_pipeline.train_adapted_cgan`:
  (1) The discriminator compares real vs. generated targets in the SAME
      physical (decoded) representation [amplitude, cos, sin, fault-probability],
      removing the input-scale mismatch that previously crippled this baseline
      (real target was a physical label while the generated target was an
      unbounded raw output).
  (2) The generator's supervised objective reuses the SAME `compute_losses`
      (with lambda_phys = 0) as every other learning method.

For the full jointly fair benchmark (Table II + Fig. 3), run
`unified_pipeline.py` directly. This launcher trains/inspects the cGAN baseline
in isolation.

Usage:
    python Adapted_cGAN.py --aep_mat_path AEP_Matrix.mat --save_dir ./out_cgan
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from unified_pipeline_final622 import (
    Config,
    build_loader,
    compute_metrics_from_arrays,
    ensure_dir,
    load_aep_matrix,
    predict_batches_cgan,
    seed_everything,
    train_adapted_cgan,
    SyntheticPhasedArrayDataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Adapted-cGAN baseline launcher")
    parser.add_argument("--aep_mat_path", type=str, default="AEP_Matrix.mat")
    parser.add_argument("--save_dir", type=str, default="./out_cgan")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Tiny run for syntax/runtime checks only")
    args = parser.parse_args()

    cfg = Config()
    cfg.aep_mat_path = args.aep_mat_path
    cfg.save_dir = args.save_dir
    if args.quick:
        cfg.train_samples, cfg.val_samples, cfg.eval_samples = 512, 128, 128
        cfg.epochs, cfg.patience, cfg.batch_size = 2, 2, 64
        cfg.num_workers, cfg.persistent_workers = 0, False
        cfg.cgan_infer_avg_seeds = 2

    out_dir = Path(cfg.save_dir)
    ensure_dir(out_dir)
    seed_everything(cfg.seed, cfg.deterministic_cudnn)
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.use_gpu_if_available and not args.cpu) else "cpu")
    print("Using device:", device)

    A_Etheta, A_Ephi, K = load_aep_matrix(cfg.aep_mat_path, cfg)
    input_dim = 4 * K
    print(f"Loaded AEP matrix: K={K}, N={cfg.N}, input_dim={input_dim}")

    generator, summary = train_adapted_cgan(cfg, input_dim, A_Etheta, A_Ephi, K, device, out_dir)

    ds = SyntheticPhasedArrayDataset(
        cfg, cfg.eval_samples, base_seed=cfg.seed + 530_000, A_Etheta=A_Etheta, A_Ephi=A_Ephi, K=K,
        scenario="mixed", sampler=cfg.eval_sampler, target_snr_db=cfg.nominal_eval_snr_db, noise_kind="gaussian",
    )
    loader = build_loader(cfg, ds, shuffle=False, device=device, seed=cfg.seed + 530_007)
    pred = predict_batches_cgan(generator, loader, cfg, device, avg_seeds=cfg.cgan_infer_avg_seeds)
    res = compute_metrics_from_arrays(
        pred["true_binary"], pred["pred_binary_head"], pred["true_amp"], pred["true_phase_deg"],
        pred["pred_amp"], pred["pred_phase_deg"], pred["phys_residual"], cfg,
    )

    pd.DataFrame([{ "best_epoch": summary["best_epoch"], "best_val_loss": summary["best_val_loss"],
                    "train_time_sec": summary["train_time_sec"], "S-E F1 (%)": res["f1_pct"],
                    "EMA (%)": res["ema_pct"] }]).to_csv(out_dir / "summary_cgan.csv", index=False)
    print(f"Adapted-cGAN @ {cfg.nominal_eval_snr_db} dB: F1={res['f1_pct']:.2f}%, EMA={res['ema_pct']:.2f}%")
    print("Saved to:", out_dir / "best_checkpoint_adapted_cgan.pt")


if __name__ == "__main__":
    main()
