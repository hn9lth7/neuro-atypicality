from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.blocks import (
    ALL_BLOCKS,
    C_FEATURES,
    D_FEATURES,
    G_FEATURES,
    SE_FEATURES,
)
from nai.nai.composite import compute_nai
from nai.normative.covariance import regularize_covariance
from nai.normative.mahalanobis import mahalanobis_distance
from nai.normative.regression import compute_residuals, fit_age_models

FEAT = PROJECT_ROOT / "results" / "features"
OUT = PROJECT_ROOT / "results" / "normative"
OUT.mkdir(parents=True, exist_ok=True)

LAMBDA = 0.10
BLOCK_FEATS = {
    "SE": SE_FEATURES,
    "C": C_FEATURES,
    "G": G_FEATURES,
    "D": D_FEATURES,
}

def load_merged() -> pd.DataFrame:
    se = pd.read_csv(FEAT / "participants_features_subject_v0.3_clean.csv")
    conn = pd.read_csv(FEAT / "connectivity_graph_subject_v0.5.csv")
    dyn = pd.read_csv(FEAT / "dynamic_subject_v0.6.csv")
    for extra in (conn, dyn):
        drop = [c for c in ("age", "sex", "group", "n_runs") if c in extra.columns]
        extra.drop(columns=drop, inplace=True, errors="ignore")
    df = se.merge(conn, on="participant_id", how="inner").merge(
        dyn, on="participant_id", how="inner"
    )
    df = df[df["group"].isin(["TD", "ASD"])].dropna(subset=["age"]).reset_index(drop=True)
    return df

def score_blocks(
    X_blocks: dict[str, np.ndarray],
    age: np.ndarray,
    models: dict,
    covs: dict,
) -> dict[str, np.ndarray]:
    out = {}
    for name, feats in BLOCK_FEATS.items():
        R = compute_residuals(X_blocks[name], age, models[name])
        d = np.array(
            [mahalanobis_distance(R[i], covs[name]) for i in range(R.shape[0])]
        )
        out[name] = d
    return out

def fit_blocks(df_td: pd.DataFrame) -> tuple[dict, dict]:
    age = df_td["age"].to_numpy(dtype=float)
    models, covs = {}, {}
    for name, feats in BLOCK_FEATS.items():
        X = df_td[feats].to_numpy(dtype=float)
        models[name] = fit_age_models(X, age)
        R = compute_residuals(X, age, models[name])
        Sigma = np.cov(R, rowvar=False)
        if Sigma.ndim == 0:
            Sigma = np.array([[float(Sigma)]])
        covs[name] = regularize_covariance(Sigma, LAMBDA)
    return models, covs

def main() -> None:
    print("=" * 72)
    print("NAI v1.2 — LOO TD validation")
    print("=" * 72)

    df = load_merged()
    td = df[df["group"] == "TD"].reset_index(drop=True)
    asd = df[df["group"] == "ASD"].reset_index(drop=True)
    n_td = len(td)
    print(f"TD={n_td}  ASD={len(asd)}  λ={LAMBDA}")

    loo_rows = []
    for i in range(n_td):
        train = td.drop(index=i).reset_index(drop=True)
        test = td.iloc[[i]]
        models, covs = fit_blocks(train)

        Xb = {name: test[feats].to_numpy(float) for name, feats in BLOCK_FEATS.items()}
        age_t = test["age"].to_numpy(float)
        D = score_blocks(Xb, age_t, models, covs)
        nai = float(compute_nai({k: float(D[k][0]) for k in BLOCK_FEATS}))

        row = {
            "participant_id": test["participant_id"].iloc[0],
            "age": float(test["age"].iloc[0]),
            "D_SE_LOO": float(D["SE"][0]),
            "D_C_LOO": float(D["C"][0]),
            "D_G_LOO": float(D["G"][0]),
            "D_D_LOO": float(D["D"][0]),
            "NAI_LOO": nai,
        }
        loo_rows.append(row)
        if (i + 1) % 10 == 0 or i + 1 == n_td:
            print(f"  LOO processed {i+1}/{n_td}")

    loo = pd.DataFrame(loo_rows)
    loo_path = OUT / "loo_nai_v12_td.csv"
    loo.to_csv(loo_path, index=False)
    print(f"Saved → {loo_path}")

    p95 = float(np.percentile(loo["NAI_LOO"], 95))
    p99 = float(np.percentile(loo["NAI_LOO"], 99))
    print("\nLOO TD NAI:")
    print(
        f"  mean={loo['NAI_LOO'].mean():.3f}  median={loo['NAI_LOO'].median():.3f}  "
        f"max={loo['NAI_LOO'].max():.3f}"
    )
    print(f"  P95={p95:.3f}  P99={p99:.3f}")

    models_full, covs_full = fit_blocks(td)
    X_asd = {name: asd[feats].to_numpy(float) for name, feats in BLOCK_FEATS.items()}
    D_asd = score_blocks(X_asd, asd["age"].to_numpy(float), models_full, covs_full)
    asd_out = asd[["participant_id", "age", "group"]].copy()
    for name in BLOCK_FEATS:
        asd_out[f"D_{name}"] = D_asd[name]
    asd_out["NAI"] = [
        compute_nai({k: float(D_asd[k][i]) for k in BLOCK_FEATS})
        for i in range(len(asd))
    ]
    asd_out["pct_vs_LOO_TD"] = [
        100.0 * (loo["NAI_LOO"].values < x).mean() for x in asd_out["NAI"].values
    ]
    asd_out["above_LOO_P95"] = asd_out["NAI"] > p95
    asd_out["above_LOO_P99"] = asd_out["NAI"] > p99

    asd_path = OUT / "asd_scores_vs_loo_v12.csv"
    asd_out.to_csv(asd_path, index=False)
    print(f"Saved → {asd_path}")

    n95 = int(asd_out["above_LOO_P95"].sum())
    n99 = int(asd_out["above_LOO_P99"].sum())
    print("\nASD vs LOO TD thresholds:")
    print(f"  n ASD              = {len(asd_out)}")
    print(f"  median NAI         = {asd_out['NAI'].median():.3f}")
    print(f"  median rank vs LOO = {asd_out['pct_vs_LOO_TD'].median():.1f}th")
    print(f"  above LOO P95      = {n95} / {len(asd_out)} ({100*n95/len(asd_out):.1f}%)")
    print(f"  above LOO P99      = {n99} / {len(asd_out)} ({100*n99/len(asd_out):.1f}%)")

    drop = {"sub-11936", "sub-2713"}
    sens = asd_out[~asd_out["participant_id"].isin(drop)]
    print("\nSensitivity (exclude 11936, 2713):")
    print(
        f"  above LOO P95 = {int(sens['above_LOO_P95'].sum())} / {len(sens)} "
        f"({100*sens['above_LOO_P95'].mean():.1f}%)"
    )

    summary = {
        "lambda": LAMBDA,
        "n_td": n_td,
        "n_asd": len(asd_out),
        "loo_nai_mean": float(loo["NAI_LOO"].mean()),
        "loo_nai_p95": p95,
        "loo_nai_p99": p99,
        "asd_above_loo_p95": n95,
        "asd_above_loo_p99": n99,
        "asd_median_nai": float(asd_out["NAI"].median()),
        "asd_median_rank_vs_loo": float(asd_out["pct_vs_LOO_TD"].median()),
    }
    with open(OUT / "loo_summary_v12.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {OUT / 'loo_summary_v12.json'}")
    print("=" * 72)

if __name__ == "__main__":
    main()