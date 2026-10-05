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
from nai.normative.regression import fit_age_models, compute_residuals
from nai.normative.covariance import regularize_covariance
from nai.normative.mahalanobis import mahalanobis_distance
from nai.normative.robustness import ledoit_wolf_covariance

FEATURES_DIR = PROJECT_ROOT / "results" / "features"
OUT_DIR = PROJECT_ROOT / "results" / "normative" / "robustness_v11"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_REF = 0.10

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

    n = len(df)
    if df["participant_id"].nunique() != n:
        raise ValueError("Duplicate participant_id after merge")
    return df.reset_index(drop=True)

def block_distances(
    X_all: np.ndarray,
    age_all: np.ndarray,
    mask_td: np.ndarray,
    estimator: str,
) -> np.ndarray:
    X_td = X_all[mask_td]
    age_td = age_all[mask_td]

    models = fit_age_models(X_td, age_td)
    R_td = compute_residuals(X_td, age_td, models)
    R_all = compute_residuals(X_all, age_all, models)

    if estimator == "lambda":
        S = np.cov(R_td, rowvar=False)
        Sigma = regularize_covariance(S, lam=LAMBDA_REF)
        meta = {"shrinkage": LAMBDA_REF}
    elif estimator == "lw":
        Sigma, sh = ledoit_wolf_covariance(R_td, assume_centered=True)
        meta = {"shrinkage": sh}
    else:
        raise ValueError(estimator)

    d = np.array(
        [mahalanobis_distance(R_all[i], Sigma) for i in range(R_all.shape[0])],
        dtype=float,
    )
    return d, meta

def loo_nai_td(
    X_blocks: dict[str, np.ndarray],
    age: np.ndarray,
    mask_td: np.ndarray,
    estimator: str,
) -> np.ndarray:
    td_idx = np.where(mask_td)[0]
    scores = np.zeros(len(td_idx), dtype=float)

    for k, i in enumerate(td_idx):
        mask_fit = mask_td.copy()
        mask_fit[i] = False

        D = {}
        for name, X in X_blocks.items():
            d, _ = block_distances(X, age, mask_fit, estimator)
            D[name] = d[i]
        scores[k] = compute_nai(D)
    return scores

def main() -> None:
    print("=" * 72)
    print("NAI v1.1 — Covariance robustness (λ=0.10 vs Ledoit–Wolf)")
    print("=" * 72)

    df = load_merged()
    mask_td = (df["group"] == "TD").values
    mask_asd = (df["group"] == "ASD").values
    age = df["age"].values.astype(float)

    print(f"Subjects: {len(df)}  TD={mask_td.sum()}  ASD={mask_asd.sum()}")
    print(f"Block dims: {BLOCK_DIMS}")

    X_blocks: dict[str, np.ndarray] = {}
    for name, cols in FEATURE_MAP.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name}: missing {missing[:5]}")
        X_blocks[name] = df[cols].values.astype(float)
        print(f"  [{name}] {X_blocks[name].shape[1]} features OK")

    results = {}
    shrinkage_info = {}

    for estimator, tag in [("lambda", "lambda"), ("lw", "lw")]:
        print(f"\n--- Estimator: {estimator} ---")
        D_by_block = {}
        for name, X in X_blocks.items():
            d, meta = block_distances(X, age, mask_td, estimator)
            D_by_block[name] = d
            shrinkage_info[f"{tag}_{name}"] = meta["shrinkage"]
            print(
                f"  D_{name}: mean_TD={d[mask_td].mean():.3f}  "
                f"max_TD={d[mask_td].max():.3f}  shrinkage={meta['shrinkage']:.4f}"
            )

        nai = compute_nai_batch(D_by_block)
        out = df[["participant_id", "age", "group"]].copy()
        if "qc_flag" in df.columns:
            out["qc_flag"] = df["qc_flag"].values
        for name in ALL_BLOCKS:
            out[f"D_{name}"] = D_by_block[name]
        out["NAI"] = nai
        out.to_csv(OUT_DIR / f"scores_{tag}.csv", index=False)
        results[tag] = out
        print(f"  NAI mean TD={nai[mask_td].mean():.3f}  ASD={list(nai[mask_asd].round(3))}")

    a = results["lambda"].set_index("participant_id")
    b = results["lw"].set_index("participant_id")
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

    print("\n--- Rank stability (Spearman, all subjects) ---")
    print(rank_df.to_string(index=False))

    print("\n--- LOO TD NAI ---")
    loo_lam = loo_nai_td(X_blocks, age, mask_td, "lambda")
    loo_lw = loo_nai_td(X_blocks, age, mask_td, "lw")
    rho_loo, p_loo = spearmanr(loo_lam, loo_lw)
    print(f"  λ=0.10  mean={loo_lam.mean():.3f}  P95={np.percentile(loo_lam, 95):.3f}")
    print(f"  LW      mean={loo_lw.mean():.3f}  P95={np.percentile(loo_lw, 95):.3f}")
    print(f"  Spearman LOO NAI: ρ={rho_loo:.3f}  p={p_loo:.3g}")

    def empirical_pct(score: float, ref: np.ndarray) -> float:
        return float(100.0 * (ref < score).mean())

    asd_rows = []
    for pid in df.loc[mask_asd, "participant_id"]:
        nai_l = float(a.loc[pid, "NAI"])
        nai_w = float(b.loc[pid, "NAI"])
        asd_rows.append(
            {
                "participant_id": pid,
                "NAI_lambda": nai_l,
                "NAI_lw": nai_w,
                "pct_LOO_lambda": empirical_pct(nai_l, loo_lam),
                "pct_LOO_lw": empirical_pct(nai_w, loo_lw),
            }
        )
        print(
            f"  {pid}: NAI_λ={nai_l:.3f} ({asd_rows[-1]['pct_LOO_lambda']:.1f}th)  "
            f"NAI_LW={nai_w:.3f} ({asd_rows[-1]['pct_LOO_lw']:.1f}th)"
        )

    summary = {
        "n_total": int(len(df)),
        "n_td": int(mask_td.sum()),
        "n_asd": int(mask_asd.sum()),
        "lambda_ref": LAMBDA_REF,
        "shrinkage": shrinkage_info,
        "rank_correlations": rank_df.to_dict(orient="records"),
        "loo": {
            "lambda_mean": float(loo_lam.mean()),
            "lambda_p95": float(np.percentile(loo_lam, 95)),
            "lw_mean": float(loo_lw.mean()),
            "lw_p95": float(np.percentile(loo_lw, 95)),
            "spearman_rho": float(rho_loo),
            "spearman_p": float(p_loo),
        },
        "asd": asd_rows,
        "note": "v1.0 default remains λ=0.10; LW is robustness comparison only.",
    }
    with open(OUT_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved → {OUT_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()