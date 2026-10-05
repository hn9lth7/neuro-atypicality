from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NAI_CSV = PROJECT_ROOT / "results" / "normative" / "nai_v10.csv"
OUT_DIR = PROJECT_ROOT / "results" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLOCKS = ["D_SE", "D_C", "D_G", "D_D"]
NAI_COL = "NAI_v10"  
ID_COL = "participant_id"
GROUP_COL = "group"

OPTIONAL = ["age", "sex", "qc_flag"]

def _resolve_nai_column(df: pd.DataFrame) -> str:
    for c in (NAI_COL, "NAI", "NAI_v1.0", "nai"):
        if c in df.columns:
            return c
    raise KeyError(
        f"NAI column not found. Available: {list(df.columns)}"
    )

def load_canonical() -> pd.DataFrame:
    if not NAI_CSV.exists():
        raise FileNotFoundError(f"Missing: {NAI_CSV}")

    df = pd.read_csv(NAI_CSV)
    nai_col = _resolve_nai_column(df)

    required = [ID_COL, GROUP_COL, *BLOCKS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    keep = required + [nai_col] + [c for c in OPTIONAL if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={nai_col: "NAI"})

    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate participant_id in nai_v10.csv")
    if not np.isfinite(df[BLOCKS + ["NAI"]].to_numpy()).all():
        raise ValueError("Non-finite values in block distances or NAI")

    recon = df[BLOCKS].mean(axis=1)
    max_delta = float(np.max(np.abs(recon - df["NAI"])))
    if max_delta > 1e-6:
        raise ValueError(
            f"NAI is not mean of blocks (max |Δ|={max_delta:.3e}). "
            "B4 refuses to proceed on inconsistent frozen table."
        )

    return df

def add_contributions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    total = out[BLOCKS].sum(axis=1)
    for b in BLOCKS:
        name = "C_" + b.replace("D_", "")
        out[name] = out[b] / total

    c_cols = ["C_SE", "C_C", "C_G", "C_D"]
    out["dominant_block"] = out[c_cols].idxmax(axis=1).str.replace("C_", "", regex=False)
    out["max_contribution"] = out[c_cols].max(axis=1)
    return out

def summarize(df: pd.DataFrame) -> dict:
    c_cols = ["C_SE", "C_C", "C_G", "C_D"]
    block_df = df[BLOCKS]
    nai = df["NAI"]

    corr_blocks = block_df.corr(method="spearman").round(4)
    corr_with_nai = {
        b: float(block_df[b].corr(nai, method="spearman")) for b in BLOCKS
    }

    def dist_stats(s: pd.Series) -> dict:
        return {
            "n": int(s.shape[0]),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
        }

    by_group = {}
    for g, sub in df.groupby(GROUP_COL, dropna=False):
        key = str(g)
        by_group[key] = {
            "n": int(len(sub)),
            "blocks": {b: dist_stats(sub[b]) for b in BLOCKS},
            "NAI": dist_stats(sub["NAI"]),
            "mean_contributions": {
                c: float(sub[c].mean()) for c in c_cols
            },
            "dominant_block_counts": sub["dominant_block"]
            .value_counts()
            .to_dict(),
        }

    concentrated = df.loc[
        df["max_contribution"] >= 0.40,
        [ID_COL, GROUP_COL, "NAI", *BLOCKS, *c_cols, "dominant_block", "max_contribution"],
    ].sort_values("max_contribution", ascending=False)

    summary = {
        "n_total": int(len(df)),
        "groups": df[GROUP_COL].value_counts(dropna=False).to_dict(),
        "formula_check_max_abs_delta": float(
            np.max(np.abs(df[BLOCKS].mean(axis=1) - df["NAI"]))
        ),
        "block_distributions_all": {b: dist_stats(df[b]) for b in BLOCKS},
        "NAI_distribution_all": dist_stats(df["NAI"]),
        "mean_contributions_all": {c: float(df[c].mean()) for c in c_cols},
        "spearman_block_correlation": corr_blocks.to_dict(),
        "spearman_block_vs_NAI": corr_with_nai,
        "dominant_block_counts_all": df["dominant_block"]
        .value_counts()
        .to_dict(),
        "n_concentrated_maxC_ge_0.40": int(len(concentrated)),
        "by_group": by_group,
        "notes": [
            "Analysis uses frozen distances only; model not recomputed.",
            "Contributions are descriptive; equal weights remain frozen.",
            "n_ASD is small; group comparisons are descriptive only.",
            "Not external validation; discovery/canonical cohort only.",
        ],
    }
    return summary, concentrated

def main() -> None:
    print("=" * 70)
    print("B4 — Block Contribution Analysis (frozen distances)")
    print("=" * 70)

    df = load_canonical()
    print(f"Loaded {len(df)} subjects from {NAI_CSV}")
    print(df[GROUP_COL].value_counts(dropna=False).to_string())

    df = add_contributions(df)
    summary, concentrated = summarize(df)

    out_csv = OUT_DIR / "b4_block_contribution.csv"
    out_json = OUT_DIR / "b4_block_summary.json"
    out_conc = OUT_DIR / "b4_concentrated_subjects.csv"

    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    concentrated.to_csv(out_conc, index=False)

    print("-" * 70)
    print("Mean contributions (all):")
    for k, v in summary["mean_contributions_all"].items():
        print(f"  {k}: {v:.3f}")
    print("Spearman block vs NAI:")
    for k, v in summary["spearman_block_vs_NAI"].items():
        print(f"  {k}: {v:.3f}")
    print("Dominant block counts:")
    for k, v in summary["dominant_block_counts_all"].items():
        print(f"  {k}: {v}")
    print(f"Concentrated (max C ≥ 0.40): {summary['n_concentrated_maxC_ge_0.40']}")
    print("-" * 70)
    print(f"Saved → {out_csv}")
    print(f"Saved → {out_json}")
    print(f"Saved → {out_conc}")
    print("=" * 70)

if __name__ == "__main__":
    main()