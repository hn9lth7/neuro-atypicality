from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
B4_CSV = PROJECT_ROOT / "results" / "validation" / "b4_block_contribution.csv"
OUT_DIR = PROJECT_ROOT / "results" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_COLS = ["C_SE", "C_C", "C_G", "C_D"]
ID_COL = "participant_id"
GROUP_COL = "group"
NAI_COL = "NAI"

BIN_EDGES = [0.0, 0.30, 0.40, 0.50, 1.01]
BIN_LABELS = [
    "maxC < 0.30",
    "0.30 ≤ maxC < 0.40",
    "0.40 ≤ maxC < 0.50",
    "maxC ≥ 0.50",
]

def profile_entropy(row: pd.Series) -> float:
    p = row[C_COLS].to_numpy(dtype=float)
    p = np.clip(p, 1e-15, None)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))

def dist_stats(s: pd.Series) -> dict:
    s = s.astype(float)
    return {
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
    }

def main() -> None:
    print("=" * 72)
    print("B5 — Heterogeneity of contribution profiles (frozen B4)")
    print("=" * 72)

    if not B4_CSV.exists():
        raise FileNotFoundError(f"Missing B4 input: {B4_CSV}")

    df = pd.read_csv(B4_CSV)

    required = [ID_COL, GROUP_COL, NAI_COL, *C_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in B4 CSV: {missing}")

    row_sum = df[C_COLS].sum(axis=1)
    max_sum_err = float(np.max(np.abs(row_sum - 1.0)))
    if max_sum_err > 1e-6:
        raise ValueError(f"Contribution rows do not sum to 1 (max |Δ|={max_sum_err})")

    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate participant_id")

    out = df[[ID_COL, GROUP_COL, NAI_COL, *C_COLS]].copy()
    if "max_contribution" in df.columns:
        out["max_contribution"] = df["max_contribution"].astype(float)
    else:
        out["max_contribution"] = out[C_COLS].max(axis=1)

    if "dominant_block" in df.columns:
        out["dominant_block"] = df["dominant_block"].astype(str)
    else:
        out["dominant_block"] = out[C_COLS].idxmax(axis=1).str.replace("C_", "", regex=False)

    out["profile_entropy"] = out.apply(profile_entropy, axis=1)

    out["concentration_bin"] = pd.cut(
        out["max_contribution"],
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        right=False,
        include_lowest=True,
    )

    q75 = float(out[NAI_COL].quantile(0.75))
    high = out.loc[out[NAI_COL] >= q75].sort_values(NAI_COL, ascending=False)

    bin_counts = (
        out["concentration_bin"]
        .value_counts()
        .reindex(BIN_LABELS, fill_value=0)
        .astype(int)
        .to_dict()
    )

    summary = {
        "n_total": int(len(out)),
        "groups": out[GROUP_COL].value_counts(dropna=False).to_dict(),
        "contribution_sum_max_abs_error": max_sum_err,
        "block_contribution_distributions": {
            c: dist_stats(out[c]) for c in C_COLS
        },
        "max_contribution_distribution": dist_stats(out["max_contribution"]),
        "profile_entropy_distribution": dist_stats(out["profile_entropy"]),
        "concentration_bins": bin_counts,
        "dominant_block_counts": out["dominant_block"].value_counts().to_dict(),
        "high_nai_definition": {
            "rule": "top quartile of NAI on canonical cohort (all subjects)",
            "NAI_P75": q75,
            "n_high": int(len(high)),
        },
        "high_nai_mean_contributions": {
            c: float(high[c].mean()) for c in C_COLS
        }
        if len(high)
        else {},
        "high_nai_median_contributions": {
            c: float(high[c].median()) for c in C_COLS
        }
        if len(high)
        else {},
        "high_nai_mean_max_contribution": float(high["max_contribution"].mean())
        if len(high)
        else None,
        "high_nai_mean_profile_entropy": float(high["profile_entropy"].mean())
        if len(high)
        else None,
        "all_mean_contributions": {c: float(out[c].mean()) for c in C_COLS},
        "all_mean_max_contribution": float(out["max_contribution"].mean()),
        "all_mean_profile_entropy": float(out["profile_entropy"].mean()),
        "notes": [
            "Descriptive analysis on frozen B4 contributions only.",
            "No model refit; no ASD vs TD hypothesis tests.",
            "High-NAI = top 25% of NAI within canonical n=41.",
            "Profile entropy uses natural log; max = log(4) ≈ 1.386 for uniform C.",
            "Not external validation.",
        ],
    }

    path_profiles = OUT_DIR / "b5_subject_profiles.csv"
    path_high = OUT_DIR / "b5_high_nai_profiles.csv"
    path_json = OUT_DIR / "b5_heterogeneity_summary.json"

    out.to_csv(path_profiles, index=False)
    high.to_csv(path_high, index=False)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Loaded {len(out)} subjects from B4")
    print(f"Contribution sum max |error|: {max_sum_err:.3e}")
    print("-" * 72)
    print("Mean C_B:", {k: round(v, 3) for k, v in summary["all_mean_contributions"].items()})
    print("max_contribution:", dist_stats(out["max_contribution"]))
    print("profile_entropy :", dist_stats(out["profile_entropy"]))
    print("Concentration bins:")
    for k, v in bin_counts.items():
        print(f"  {k}: {v}")
    print(f"High-NAI (NAI ≥ P75={q75:.3f}): n={len(high)}")
    if len(high):
        print(
            "  mean C:",
            {k: round(v, 3) for k, v in summary["high_nai_mean_contributions"].items()},
        )
    print("-" * 72)
    print(f"Saved → {path_profiles}")
    print(f"Saved → {path_high}")
    print(f"Saved → {path_json}")
    print("=" * 72)

if __name__ == "__main__":
    main()