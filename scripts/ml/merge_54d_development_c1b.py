from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEAT = PROJECT_ROOT / "results" / "features"

PATH_SE = FEAT / "participants_features_subject_v0.3_clean.csv"
PATH_CG = FEAT / "connectivity_graph_subject_v0.5.csv"
PATH_D = FEAT / "dynamic_subject_v0.6.csv"
OUT = FEAT / "features_54d_development_c1b.csv"

SE = [
    "alpha_rel",
    "beta_rel",
    "theta_rel",
    "spectral_entropy_mean",
    "log_theta_alpha",
    "log_theta_beta",
]
C = [
    "plv_mean_theta",
    "plv_mean_alpha",
    "plv_mean_beta",
    "plv_mean_gamma",
    "plv_median_theta",
    "plv_median_alpha",
    "plv_median_beta",
    "plv_median_gamma",
]
G = [
    f"{m}_{b}"
    for m in [
        "mean_degree",
        "degree_cv",
        "clustering",
        "global_efficiency",
        "mean_path_length",
        "laplacian_entropy",
    ]
    for b in ["theta", "alpha", "beta", "gamma"]
]
D = [
    f"{m}_{b}"
    for m in [
        "mean_delta",
        "cv_delta",
        "mean_degree_cv",
        "temporal_cv_degree_cv",
    ]
    for b in ["theta", "alpha", "beta", "gamma"]
]
FEATURES_54 = SE + C + G + D
assert len(FEATURES_54) == 54

META = ["participant_id", "age", "sex", "group", "qc_flag", "n_runs"]

def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns ({len(missing)}): {missing}")

def main() -> None:
    print("=" * 72)
    print("C1b — merge 54-D development table")
    print("=" * 72)

    se = pd.read_csv(PATH_SE)
    cg = pd.read_csv(PATH_CG)
    dyn = pd.read_csv(PATH_D)

    print(f"SE clean : {len(se)} rows")
    print(f"C+G      : {len(cg)} rows")
    print(f"D        : {len(dyn)} rows")

    _require(se, ["participant_id", "age", "group"] + SE, "SE")
    meta_se = [c for c in META if c in se.columns]
    _require(cg, ["participant_id"] + C + G, "C+G")
    _require(dyn, ["participant_id"] + D, "D")

    left = se[meta_se + SE].copy()
    mid = cg[["participant_id"] + C + G].copy()
    right = dyn[["participant_id"] + D].copy()

    m = left.merge(mid, on="participant_id", how="inner", validate="one_to_one")
    m = m.merge(right, on="participant_id", how="inner", validate="one_to_one")

    print(f"After inner join: {len(m)} subjects")
    print(m["group"].value_counts(dropna=False).to_string())

    ordered_meta = [c for c in META if c in m.columns]
    out = m[ordered_meta + FEATURES_54].copy()

    if out["participant_id"].duplicated().any():
        raise ValueError("duplicate participant_id after merge")
    n_na = out[FEATURES_54].isna().sum().sum()
    if n_na:
        print(f"WARN: feature NaN cells = {n_na}")
    else:
        print("PASS: 0 feature NaN")

    missing_f = [c for c in FEATURES_54 if c not in out.columns]
    if missing_f:
        raise KeyError(f"output missing features: {missing_f}")

    out.to_csv(OUT, index=False)
    print(f"Saved → {OUT}")
    print(f"shape: {out.shape}")
    print("=" * 72)
    print("Next: python scripts/ml/ml_dataset_audit_c1.py --csv results/features/features_54d_development_c1b.csv")
    print("=" * 72)

if __name__ == "__main__":
    main()