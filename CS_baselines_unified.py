"""
Compressed-sensing (CS) baselines -- UNIFIED launcher (resubmission of
AWPL-04-26-1519).

Evaluates the OMP / Lasso / FISTA deviation-recovery baselines on EXACTLY the
same uniform mixed-impairment test distribution and nominal SNR used for the
learning methods (all drawn from `unified_pipeline.generate_sample`).

Reviewer 3 (Comment 1) and Reviewer 2 (Comment 4) asked for a fair comparison
under an identical number of observation points. This script therefore reports
each CS method at BOTH:
  * M = 50  : the conventional sparse array-diagnosis setting, and
  * M = K   : the equal-observation setting (same complete two-cut far field
              that the learning methods use).
Note (disclosed in the response letter): the CS baselines additionally use the
ideal healthy reference field (deviation formulation E_meas - A*1), which the
learning methods do not require.

Usage:
    python CS_baselines.py --aep_mat_path AEP_Matrix.mat --save_dir ./out_cs
"""
import argparse
from pathlib import Path

import pandas as pd

from unified_pipeline_final622 import Config, ensure_dir, evaluate_cs_method, load_aep_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified CS baselines launcher")
    parser.add_argument("--aep_mat_path", type=str, default="AEP_Matrix.mat")
    parser.add_argument("--save_dir", type=str, default="./out_cs")
    parser.add_argument("--quick", action="store_true", help="Fewer test samples for a fast check")
    args = parser.parse_args()

    cfg = Config()
    cfg.aep_mat_path = args.aep_mat_path
    cfg.save_dir = args.save_dir
    if args.quick:
        cfg.eval_samples = 128

    out_dir = Path(cfg.save_dir)
    ensure_dir(out_dir)
    A_Etheta, A_Ephi, K = load_aep_matrix(cfg.aep_mat_path, cfg)
    print(f"Loaded AEP matrix: K={K}, N={cfg.N}")

    rows = []
    for method in cfg.cs_methods:
        for M_obs in cfg.cs_M_obs_list:
            res = evaluate_cs_method(
                method, cfg, A_Etheta, A_Ephi, K, M_obs=M_obs,
                omp_k=cfg.cs_omp_k, lasso_alpha=cfg.cs_lasso_alpha, base_seed=cfg.seed + 540_000,
            )
            m = res["metrics"]
            rows.append({
                "Method": method, "M_obs": m["M_obs"],
                "S-E Precision (%)": m["precision_pct"], "S-E Recall (%)": m["recall_pct"],
                "S-E F1 (%)": m["f1_pct"], "EMA (%)": m["ema_pct"],
                "Infer (ms/sample)": m["infer_time_ms_per_sample"],
            })
            print(f"{method:6s} M={m['M_obs']:4d} | F1={m['f1_pct']:.2f}% | EMA={m['ema_pct']:.2f}% | {m['infer_time_ms_per_sample']:.2f} ms")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary_cs_baselines.csv", index=False, encoding="utf-8-sig")
    print("Saved to:", out_dir / "summary_cs_baselines.csv")


if __name__ == "__main__":
    main()
