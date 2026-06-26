"""
ANN-MLP baseline -- UNIFIED launcher (resubmission of AWPL-04-26-1519).

This script no longer owns its own data generation, SNR setting, decoding, or
loss. It imports everything from `unified_pipeline.py`, so the ANN-MLP baseline
is trained on EXACTLY the same data distribution, per-sample training SNR,
bounded-amplitude/stable-phase decoding, class-imbalance-weighted BCE, optimizer
/scheduler/early-stopping rule, seed, and weight initialization as the proposed
PI-1D-ResNet and every other learning baseline. The ONLY difference is the
network architecture (a shallow MLP). This directly removes the
training-pipeline confounds raised by Reviewer 2 (Comment 4) and Reviewer 3
(Comment 1).

For the full, jointly fair benchmark (Table II + Fig. 3, all 8 methods compared
under identical conditions), run `unified_pipeline.py` directly; it trains every
method and produces the comparison tables. This standalone launcher is provided
for users who want to train/inspect the ANN-MLP baseline in isolation.

Usage:
    python ANN-MLP.py --aep_mat_path AEP_Matrix.mat --save_dir ./out_ann_mlp
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from unified_pipeline_final622 import (
    ANNMLP,
    Config,
    ensure_dir,
    evaluate_model_on_dataset,
    load_aep_matrix,
    seed_everything,
    train_standard_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ANN-MLP baseline launcher")
    parser.add_argument("--aep_mat_path", type=str, default="AEP_Matrix.mat")
    parser.add_argument("--save_dir", type=str, default="./out_ann_mlp")
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

    out_dir = Path(cfg.save_dir)
    ensure_dir(out_dir)
    seed_everything(cfg.seed, cfg.deterministic_cudnn)
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.use_gpu_if_available and not args.cpu) else "cpu")
    print("Using device:", device)

    A_Etheta, A_Ephi, K = load_aep_matrix(cfg.aep_mat_path, cfg)
    input_dim = 4 * K
    print(f"Loaded AEP matrix: K={K}, N={cfg.N}, input_dim={input_dim}")

    model = ANNMLP(input_dim=input_dim, num_elements=cfg.N, hidden=cfg.ann_hidden)
    model, summary = train_standard_model(model, cfg, "ann_mlp", input_dim, A_Etheta, A_Ephi, K, device, out_dir)

    res = evaluate_model_on_dataset(
        model, cfg, A_Etheta, A_Ephi, K, device, scenario="mixed", sampler=cfg.eval_sampler,
        target_snr_db=cfg.nominal_eval_snr_db, noise_kind="gaussian", decision_mode="fault_head",
        base_seed=cfg.seed + 510_000,
    )["metrics"]

    pd.DataFrame([{ "best_epoch": summary["best_epoch"], "best_val_loss": summary["best_val_loss"],
                    "train_time_sec": summary["train_time_sec"], "S-E F1 (%)": res["f1_pct"],
                    "EMA (%)": res["ema_pct"] }]).to_csv(out_dir / "summary_ann_mlp.csv", index=False)
    print(f"ANN-MLP @ {cfg.nominal_eval_snr_db} dB: F1={res['f1_pct']:.2f}%, EMA={res['ema_pct']:.2f}%")
    print("Saved to:", out_dir / "best_checkpoint_ann_mlp.pt")


if __name__ == "__main__":
    main()
