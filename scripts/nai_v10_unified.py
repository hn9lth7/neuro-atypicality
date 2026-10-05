from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.blocks import ALL_BLOCKS, BLOCK_DIMS
from nai.normative import (
    fit_age_models,
    compute_residuals,
    empirical_covariance,
    regularize_covariance,
    covariance_diagnostics,
    mahalanobis_distance,
    mahalanobis_batch,
    loo_mahalanobis,
    lambda_sensitivity,
    empirical_percentile,
)
from nai.nai.composite import compute_nai, compute_nai_batch
from nai.nai import compute_nai

FEATURES_DIR = PROJECT_ROOT / "results" / "features"
NORM_DIR = PROJECT_ROOT / "results" / "normative"
NORM_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA = 0.10
LAMBDAS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

def _assert_unique_subjects(df: pd.DataFrame, name: str) -> None:
    n = len(df)
    n_unique = df["participant_id"].nunique()
    if n != n_unique:
        raise ValueError(
            f"{name}: expected 1 row per subject, got {n} rows / {n_unique} ids"
        )
    
def load_merged() -> pd.DataFrame:
    spectral = pd.read_csv(FEATURES_DIR / "features_v032.csv")
    conn = pd.read_csv(FEATURES_DIR / "connectivity_graph_subject_v0.5.csv")
    dyn = pd.read_csv(FEATURES_DIR / "dynamic_subject_v0.6.csv")

    _assert_unique_subjects(spectral, "features_v032")
    _assert_unique_subjects(conn, "connectivity_graph_subject_v0.5")
    _assert_unique_subjects(dyn, "dynamic_subject_v0.6")

    df = spectral.merge(
        conn.drop(columns=["age", "sex", "group"], errors="ignore"),
        on="participant_id",
        how="inner",
    )
    df = df.merge(
        dyn.drop(columns=["age", "sex", "group", "n_runs"], errors="ignore"),
        on="participant_id",
        how="inner",
    )

    if "qc_flag" in df.columns:
        df = df[df["qc_flag"].isin(["ok", "very_low_alpha"])].copy()
    df = df[df["participant_id"] != "sub-10777"].copy()
    df = df.dropna(subset=["age", "group"]).copy()

    _assert_unique_subjects(df, "merged canonical")
    return df.reset_index(drop=True)

def main():
    print("=" * 72)
    print("NAI v1.0 — Unified Four-Block Scoring")
    print("=" * 72)

    df = load_merged()
    td = df[df["group"] == "TD"].copy().reset_index(drop=True)
    asd = df[df["group"] == "ASD"].copy().reset_index(drop=True)

    print(f"Canonical : {len(df)}  (TD={len(td)}, ASD={len(asd)})")
    print(f"Block dims: {BLOCK_DIMS}")

    for name, feats in ALL_BLOCKS.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            raise KeyError(f"[{name}] missing columns: {missing}")
        print(f"  [{name}] {len(feats)} features OK")

    models = {}
    residuals_td = {}
    Sigma_reg = {}
    diag = {}

    print("\n" + "-" * 72)
    print(f"BLOCK DIAGNOSTICS (TD, λ={LAMBDA})")
    print("-" * 72)

    for name, feats in ALL_BLOCKS.items():
        X = td[feats].values.astype(float)
        age = td["age"].values
        models[name] = fit_age_models(X, age)
        R = compute_residuals(X, age, models[name])
        residuals_td[name] = R
        Sigma = empirical_covariance(R)
        S_reg = regularize_covariance(Sigma, LAMBDA)
        Sigma_reg[name] = S_reg
        d = covariance_diagnostics(Sigma, LAMBDA)
        diag[name] = d
        print(
            f"  [{name:2s}] n={d['n_features']:2d}  rank={d['rank_reg']:2d}  "
            f"cond_raw={d['cond_raw']:.2e}  cond_reg={d['cond_reg']:.2e}"
        )

    rows = []
    for _, row in df.iterrows():
        age = float(row["age"])
        rec = {
            "participant_id": row["participant_id"],
            "age": age,
            "sex": row.get("sex"),
            "group": row["group"],
            "qc_flag": row.get("qc_flag", "ok"),
        }
        D = {}
        for name, feats in ALL_BLOCKS.items():
            x = row[feats].values.astype(float)
            r = compute_residuals(x.reshape(1, -1), np.array([age]), models[name]).ravel()
            D[name] = mahalanobis_distance(r, Sigma_reg[name])
            rec[f"D_{name}"] = D[name]
        rec["NAI_v10"] = compute_nai(D)
        rows.append(rec)

    res = pd.DataFrame(rows)

    print("\n" + "-" * 72)
    print("LEAVE-ONE-OUT (TD)")
    print("-" * 72)

    loo_D = {}
    for name, feats in ALL_BLOCKS.items():
        X = td[feats].values.astype(float)
        age = td["age"].values
        loo_D[name] = loo_mahalanobis(X, age, LAMBDA)
        print(
            f"  D_{name}: mean={loo_D[name].mean():.3f}  "
            f"median={np.median(loo_D[name]):.3f}  "
            f"max={loo_D[name].max():.3f}"
        )

    loo_nai = (
        loo_D["SE"] + loo_D["C"] + loo_D["G"] + loo_D["D"]
    ) / 4.0
    print(
        f"  NAI  : mean={loo_nai.mean():.3f}  "
        f"median={np.median(loo_nai):.3f}  "
        f"max={loo_nai.max():.3f}"
    )
    print(f"  NAI P95={np.percentile(loo_nai, 95):.3f}  "
          f"P99={np.percentile(loo_nai, 99):.3f}")

    print("\n" + "-" * 72)
    print("ASD vs LOO TD (λ=0.10)")
    print("-" * 72)

    for _, row in res[res["group"] == "ASD"].iterrows():
        nai = row["NAI_v10"]
        pct = empirical_percentile(loo_nai, nai)
        print(
            f"  {row['participant_id']}  "
            f"D_SE={row['D_SE']:.3f}  D_C={row['D_C']:.3f}  "
            f"D_G={row['D_G']:.3f}  D_D={row['D_D']:.3f}  "
            f"NAI={nai:.3f}  →  {pct:.1f}-th percentile"
        )

    print("\n" + "-" * 72)
    print("λ SENSITIVITY (mean NAI TD / ASD values)")
    print("-" * 72)

    age_all = df["age"].values
    X_blocks = {name: df[feats].values.astype(float) for name, feats in ALL_BLOCKS.items()}
    X_td_blocks = {name: td[feats].values.astype(float) for name, feats in ALL_BLOCKS.items()}
    age_td = td["age"].values

    for lam in LAMBDAS:
        D_by_block = {}
        for name in ALL_BLOCKS:
            sens = lambda_sensitivity(
                X_td_blocks[name],
                age_td,
                X_blocks[name],
                age_all,
                lambdas=[lam],
            )
            D_by_block[name] = sens[float(lam)]

        nai_all = compute_nai_batch(D_by_block)
        mask_td = (df["group"] == "TD").values
        mask_asd = (df["group"] == "ASD").values

        nai_td = nai_all[mask_td]
        asd_ids = df.loc[mask_asd, "participant_id"].values
        asd_nai = nai_all[mask_asd]
        asd_str = "  ".join(
            f"{pid}={val:.3f}" for pid, val in zip(asd_ids, asd_nai)
        )
        print(f"  λ={lam:.2f}  TD mean={nai_td.mean():.3f}  {asd_str}")

    print("\n" + "-" * 72)
    print("NAI_v10 SUMMARY")
    print("-" * 72)
    print(
        res.groupby("group")[["D_SE", "D_C", "D_G", "D_D", "NAI_v10"]]
        .describe()
        .round(3)
        .to_string()
    )

    print("\nTop 5 NAI_v10:")
    print(
        res.nlargest(5, "NAI_v10")[
            ["participant_id", "group", "qc_flag", "D_SE", "D_C", "D_G", "D_D", "NAI_v10"]
        ]
        .round(3)
        .to_string(index=False)
    )

    out_csv = NORM_DIR / "nai_v10.csv"
    res.to_csv(out_csv, index=False)

    for name, S in Sigma_reg.items():
        np.save(NORM_DIR / f"covariance_{name}_v10.npy", S)

    loo_df = td[["participant_id", "age"]].copy()
    for name in ALL_BLOCKS:
        loo_df[f"D_{name}_LOO"] = loo_D[name]
    loo_df["NAI_LOO"] = loo_nai
    loo_df.to_csv(NORM_DIR / "loo_nai_v10.csv", index=False)

    summary = {
        "version": "v1.0",
        "n_td": len(td),
        "n_asd": len(asd),
        "lambda": LAMBDA,
        "block_dims": BLOCK_DIMS,
        "diagnostics": diag,
        "weights": {k: 0.25 for k in ALL_BLOCKS},
        "loo_nai_mean": float(loo_nai.mean()),
        "loo_nai_p95": float(np.percentile(loo_nai, 95)),
        "loo_nai_p99": float(np.percentile(loo_nai, 99)),
    }
    with open(NORM_DIR / "model_summary_v10.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved → {out_csv}")
    print(f"Saved → {NORM_DIR / 'loo_nai_v10.csv'}")
    print(f"Saved → {NORM_DIR / 'model_summary_v10.json'}")
    print("=" * 72)

if __name__ == "__main__":
    main()