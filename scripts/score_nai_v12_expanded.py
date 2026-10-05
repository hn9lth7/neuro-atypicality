from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.blocks import (
    C_FEATURES,
    D_FEATURES,
    G_FEATURES,
    SE_FEATURES,
)
from nai.features.extractor import score_nai_from_dataframe

FEAT = PROJECT_ROOT / "results" / "features"
OUT = PROJECT_ROOT / "results" / "normative"
OUT.mkdir(parents=True, exist_ok=True)

LAMBDA = 0.10

def load_merged_expanded() -> pd.DataFrame:
    se = pd.read_csv(FEAT / "participants_features_subject_v0.3_clean.csv")
    conn = pd.read_csv(FEAT / "connectivity_graph_subject_v0.5.csv")
    dyn = pd.read_csv(FEAT / "dynamic_subject_v0.6.csv")

    for extra in (conn, dyn):
        drop = [c for c in ("age", "sex", "group", "n_runs") if c in extra.columns]
        extra.drop(columns=drop, inplace=True, errors="ignore")

    df = se.merge(conn, on="participant_id", how="inner").merge(
        dyn, on="participant_id", how="inner"
    )
    df = df[df["group"].isin(["TD", "ASD"])].copy()
    df = df.dropna(subset=["age"]).reset_index(drop=True)

    missing = [
        c
        for c in SE_FEATURES + C_FEATURES + G_FEATURES + D_FEATURES
        if c not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:20]}")

    n = len(df)
    n_u = df["participant_id"].nunique()
    if n != n_u:
        raise ValueError(f"Duplicate participant_id: {n} rows / {n_u} ids")

    return df

def main() -> None:
    print("=" * 72)
    print("NAI v1.2 — Expanded cohort (frozen v1.0 formula)")
    print("=" * 72)

    df = load_merged_expanded()
    n_td = int((df["group"] == "TD").sum())
    n_asd = int((df["group"] == "ASD").sum())
    print(f"Merged subjects: {len(df)}  (TD={n_td}, ASD={n_asd})")

    scored = score_nai_from_dataframe(df, lam=LAMBDA)

    if "qc_flag" in df.columns:
        scored = scored.merge(
            df[["participant_id", "qc_flag"]],
            on="participant_id",
            how="left",
        )

    out_csv = OUT / "nai_v12_expanded.csv"
    scored.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    print("\n--- NAI by group ---")
    print(
        scored.groupby("group")["NAI"].agg(
            ["count", "mean", "std", "min", "median", "max"]
        )
    )

    td_nai = scored.loc[scored["group"] == "TD", "NAI"].values
    asd = scored[scored["group"] == "ASD"].copy()
    asd["pct_vs_TD"] = [
        100.0 * (td_nai < x).mean() for x in asd["NAI"].values
    ]
    print("\n--- ASD vs TD empirical rank (in-sample TD scores) ---")
    print(
        f"  ASD median NAI={asd['NAI'].median():.3f}  "
        f"median rank vs TD={asd['pct_vs_TD'].median():.1f}th"
    )
    print(
        f"  ASD > TD P95: "
        f"{(asd['NAI'] > np.percentile(td_nai, 95)).sum()} / {len(asd)}"
    )
    print(
        f"  ASD > TD P99: "
        f"{(asd['NAI'] > np.percentile(td_nai, 99)).sum()} / {len(asd)}"
    )

    print("\nTop 10 NAI (all groups):")
    cols = ["participant_id", "group", "age", "NAI", "D_SE", "D_C", "D_G", "D_D"]
    if "qc_flag" in scored.columns:
        cols.insert(2, "qc_flag")
    print(scored.nlargest(10, "NAI")[cols].to_string(index=False))

    summary = {
        "version": "v1.2_expanded",
        "lambda": LAMBDA,
        "n_td_fit": n_td,
        "n_asd_scored": n_asd,
        "n_total": len(scored),
        "formula": "NAI = mean(D_SE, D_C, D_G, D_D)",
        "note": "Same math as v1.0; expanded ASD scoring; no ASD in covariance fit",
    }
    with open(OUT / "model_summary_v12.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {OUT / 'model_summary_v12.json'}")
    print("=" * 72)

if __name__ == "__main__":
    main()