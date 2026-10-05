from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.blocks import ALL_BLOCKS, BLOCK_DIMS
from nai.features.blocks import SE_FEATURES, C_FEATURES, G_FEATURES, D_FEATURES
from nai.nai.composite import compute_nai, compute_nai_batch
from nai.normative.regression import (
    fit_age_models,
    fit_age_models_quadratic,
    compute_residuals,
)
from nai.normative.covariance import regularize_covariance
from nai.normative.mahalanobis import mahalanobis_distance

FEATURES_DIR = PROJECT_ROOT / "results" / "features"
OUT_DIR = PROJECT_ROOT / "results" / "normative" / "robustness_age_v11"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA = 0.10

FEATURE_MAP = {
    "SE": SE_FEATURES,
    "C": C_FEATURES,
    "G": G_FEATURES,
    "D": D_FEATURES,
}

def load_merged() -> pd.DataFrame:
    spectral = pd.read_csv(FEATURES_DIR / "features_v032.csv")
    conn = pd.read_csv(FEATURES_DIR / "connectivity_graph_subject_v0.5.csv")
    dyn = pd.read_csv(FEATURES_DIR / "dynamic_subject_v0.6.csv")

    for extra in (conn, dyn):
        drop = [c for c in ("age", "sex", "group") if c in extra.columns and c in spectral.columns]
        extra.drop(columns=drop, inplace=True, errors="ignore")

    df = spectral.merge(conn, on="participant_id", how="inner")
    df = df.merge(dyn, on="participant_id", how="inner")

    if "qc_flag" in df.columns:
        df = df[~df["qc_flag"].isin(["extreme_artifact", "missing_metadata"])].copy()
    df = df[df["group"].isin(["TD", "ASD"])].copy()
    df = df.dropna(subset=["age"]).copy()

    if df["participant_id"].nunique() != len(df):
        raise ValueError("Duplicate participant_id after merge")
    return df.reset_index(drop=True)

def fit_fn(mode: str):
    if mode == "linear":
        return fit_age_models
    if mode == "quadratic":
        return fit_age_models_quadratic
    raise ValueError(mode)

def block_distances(
    X_all: np.ndarray,
    age_all: np.ndarray,
    mask_td: np.ndarray,
    mode: str,
) -> np.ndarray:
    X_td = X_all[mask_td]
    age_td = age_all[mask_td]
    models = fit_fn(mode)(X_td, age_td)
    R_td = compute_residuals(X_td, age_td, models)
    R_all = compute_residuals(X_all, age_all, models)
    S = np.cov(R_td, rowvar=False)
    Sigma = regularize_covariance(S, lam=LAMBDA)
    return np.array(
        [mahalanobis_distance(R_all[i], Sigma) for i in range(R_all.shape[0])],
        dtype=float,
    )

def loo_nai_td(
    X_blocks: dict[str, np.ndarray],
    age: np.ndarray,
    mask_td: np.ndarray,
    mode: str,
) -> np.ndarray:
    td_idx = np.where(mask_td)[0]
    scores = np.zeros(len(td_idx), dtype=float)
    for k, i in enumerate(td_idx):
        mask_fit = mask_td.copy()
        mask_fit[i] = False
        D = {}
        for name, X in X_blocks.items():
            d = block_distances(X, age, mask_fit, mode)
            D[name] = d[i]
        scores[k] = compute_nai(D)
    return scores

def main() -> None:
    print("=" * 72)
    print("NAI v1.1 — Age robustness (linear vs quadratic)")
    print("=" * 72)

    df = load_merged()
    mask_td = (df["group"] == "TD").values
    mask_asd = (df["group"] == "ASD").values
    age = df["age"].values.astype(float)

    print(f"Subjects: {len(df)}  TD={mask_td.sum()}  ASD={mask_asd.sum()}")
    print(f"Block dims: {BLOCK_DIMS}")
    print(f"Covariance: λ={LAMBDA} (fixed)")

    X_blocks: dict[str, np.ndarray] = {}
    for name, cols in FEATURE_MAP.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name}: missing {missing[:5]}")
        X_blocks[name] = df[cols].values.astype(float)
        print(f"  [{name}] {X_blocks[name].shape[1]} features OK")

    results = {}
    for mode in ("linear", "quadratic"):
        print(f"\n--- Age model: {mode} ---")
        D_by_block = {}
        for name, X in X_blocks.items():
            d = block_distances(X, age, mask_td, mode)
            D_by_block[name] = d
            print(
                f"  D_{name}: mean_TD={d[mask_td].mean():.3f}  max_TD={d[mask_td].max():.3f}"
            )

        nai = compute_nai_batch(D_by_block)
        out = df[["participant_id", "age", "group"]].copy()
        if "qc_flag" in df.columns:
            out["qc_flag"] = df["qc_flag"].values
        for name in ALL_BLOCKS:
            out[f"D_{name}"] = D_by_block[name]
        out["NAI"] = nai
        out.to_csv(OUT_DIR / f"scores_{mode}.csv", index=False)
        results[mode] = out
        print(f"  NAI mean TD={nai[mask_td].mean():.3f}  ASD={list(np.round(nai[mask_asd], 3))}")

    a = results["linear"].set_index("participant_id")
    b = results["quadratic"].set_index("participant_id")
    common = a.index.intersection(b.index)

    rows = []
    for col in ["D_SE", "D_C", "D_G", "D_D", "NAI"]:
        x = a.loc[common, col].values.astype(float)
        y = b.loc[common, col].values.astype(float)
        rho, p = spearmanr(x, y)
        rows.append(
            {
                "metric": col,
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "mean_abs_diff": float(np.mean(np.abs(x - y))),
                "max_abs_diff": float(np.max(np.abs(x - y))),
            }
        )
    rank_df = pd.DataFrame(rows)
    rank_df.to_csv(OUT_DIR / "rank_correlations.csv", index=False)

    print("\n--- Rank stability (Spearman) ---")
    print(rank_df.to_string(index=False))

    print("\n--- LOO TD NAI ---")
    loo_lin = loo_nai_td(X_blocks, age, mask_td, "linear")
    loo_quad = loo_nai_td(X_blocks, age, mask_td, "quadratic")
    rho_loo, p_loo = spearmanr(loo_lin, loo_quad)
    print(f"  linear     mean={loo_lin.mean():.3f}  P95={np.percentile(loo_lin, 95):.3f}")
    print(f"  quadratic  mean={loo_quad.mean():.3f}  P95={np.percentile(loo_quad, 95):.3f}")
    print(f"  Spearman LOO NAI: ρ={rho_loo:.3f}  p={p_loo:.3g}")

    def empirical_pct(score: float, ref: np.ndarray) -> float:
        return float(100.0 * (ref < score).mean())

    asd_rows = []
    for pid in df.loc[mask_asd, "participant_id"]:
        nai_l = float(a.loc[pid, "NAI"])
        nai_q = float(b.loc[pid, "NAI"])
        asd_rows.append(
            {
                "participant_id": pid,
                "NAI_linear": nai_l,
                "NAI_quadratic": nai_q,
                "pct_LOO_linear": empirical_pct(nai_l, loo_lin),
                "pct_LOO_quadratic": empirical_pct(nai_q, loo_quad),
            }
        )
        print(
            f"  {pid}: NAI_lin={nai_l:.3f} ({asd_rows[-1]['pct_LOO_linear']:.1f}th)  "
            f"NAI_quad={nai_q:.3f} ({asd_rows[-1]['pct_LOO_quadratic']:.1f}th)"
        )

    summary = {
        "n_total": int(len(df)),
        "n_td": int(mask_td.sum()),
        "n_asd": int(mask_asd.sum()),
        "lambda": LAMBDA,
        "rank_correlations": rank_df.to_dict(orient="records"),
        "loo": {
            "linear_mean": float(loo_lin.mean()),
            "linear_p95": float(np.percentile(loo_lin, 95)),
            "quadratic_mean": float(loo_quad.mean()),
            "quadratic_p95": float(np.percentile(loo_quad, 95)),
            "spearman_rho": float(rho_loo),
            "spearman_p": float(p_loo),
        },
        "asd": asd_rows,
        "note": "v1.0 default remains linear age; quadratic is robustness only.",
    }
    with open(OUT_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved → {OUT_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()