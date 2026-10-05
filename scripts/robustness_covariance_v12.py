from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.blocks import C_FEATURES, D_FEATURES, G_FEATURES, SE_FEATURES
from nai.features.extractor import score_nai_from_dataframe

FEAT = PROJECT_ROOT / "results" / "features"
OUT = PROJECT_ROOT / "results" / "normative" / "robustness_v12"
OUT.mkdir(parents=True, exist_ok=True)

LAMBDAS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
BASE_LAM = 0.10
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
    return (
        df[df["group"].isin(["TD", "ASD"])]
        .dropna(subset=["age"])
        .reset_index(drop=True)
    )

def main() -> None:
    print("=" * 72)
    print("NAI v1.2 — Covariance λ sensitivity")
    print("=" * 72)

    df = load_merged()
    print(f"Subjects: {len(df)}  TD={(df.group=='TD').sum()}  ASD={(df.group=='ASD').sum()}")

    scores_by_lam: dict[float, pd.DataFrame] = {}
    for lam in LAMBDAS:
        print(f"  scoring λ={lam:.2f} ...")
        sc = score_nai_from_dataframe(df, lam=lam)
        sc = sc.rename(columns={"NAI": f"NAI_l{lam}"})
        for b in ("SE", "C", "G", "D"):
            sc = sc.rename(columns={f"D_{b}": f"D_{b}_l{lam}"})
        scores_by_lam[lam] = sc

    base = scores_by_lam[BASE_LAM][
        ["participant_id", "group", "age", "NAI_l0.1", "D_SE_l0.1", "D_C_l0.1", "D_G_l0.1", "D_D_l0.1"]
    ].copy()
    base = base.rename(
        columns={
            "NAI_l0.1": "NAI_base",
            "D_SE_l0.1": "D_SE_base",
            "D_C_l0.1": "D_C_base",
            "D_G_l0.1": "D_G_base",
            "D_D_l0.1": "D_D_base",
        }
    )

    wide = base[["participant_id", "group", "age", "NAI_base"]].copy()
    for lam in LAMBDAS:
        col = f"NAI_l{lam}"
        wide = wide.merge(
            scores_by_lam[lam][["participant_id", col]],
            on="participant_id",
            how="left",
        )

    wide_path = OUT / "covariance_sensitivity_nai.csv"
    wide.to_csv(wide_path, index=False)
    print(f"Saved → {wide_path}")

    rows = []
    asd_mask = wide["group"] == "ASD"
    td_mask = wide["group"] == "TD"
    for lam in LAMBDAS:
        col = f"NAI_l{lam}"
        rho_all, _ = spearmanr(wide["NAI_base"], wide[col])
        rho_asd, _ = spearmanr(wide.loc[asd_mask, "NAI_base"], wide.loc[asd_mask, col])
        rho_td, _ = spearmanr(wide.loc[td_mask, "NAI_base"], wide.loc[td_mask, col])
        rows.append(
            {
                "lambda": lam,
                "spearman_all": float(rho_all),
                "spearman_TD": float(rho_td),
                "spearman_ASD": float(rho_asd),
                "NAI_mean_TD": float(wide.loc[td_mask, col].mean()),
                "NAI_mean_ASD": float(wide.loc[asd_mask, col].mean()),
                "NAI_median_ASD": float(wide.loc[asd_mask, col].median()),
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = OUT / "covariance_sensitivity.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved → {summary_path}")
    print(summary.to_string(index=False))

    try:
        from sklearn.covariance import LedoitWolf
        from nai.normative.regression import fit_age_models, compute_residuals
        from nai.normative.mahalanobis import mahalanobis_distance
        from nai.nai.composite import compute_nai

        td = df[df["group"] == "TD"]
        age_td = td["age"].to_numpy(float)
        models, covs = {}, {}
        for name, feats in BLOCK_FEATS.items():
            X = td[feats].to_numpy(float)
            models[name] = fit_age_models(X, age_td)
            R = compute_residuals(X, age_td, models[name])
            covs[name] = LedoitWolf().fit(R).covariance_

        nai_lw = []
        for _, row in df.iterrows():
            d = {}
            for name, feats in BLOCK_FEATS.items():
                x = row[feats].to_numpy(float).reshape(1, -1)
                r = compute_residuals(x, np.array([row["age"]]), models[name])[0]
                d[name] = float(mahalanobis_distance(r, covs[name]))
            nai_lw.append(compute_nai({k: d[k] for k in ("SE", "C", "G", "D")}))
        wide["NAI_LedoitWolf"] = nai_lw
        rho_lw, _ = spearmanr(wide["NAI_base"], wide["NAI_LedoitWolf"])
        print(f"\nLedoit–Wolf vs λ=0.10 Spearman (all): {rho_lw:.4f}")
        wide.to_csv(wide_path, index=False)
    except Exception as e:
        print(f"\nLedoit–Wolf skipped: {e}")

    meta = {
        "baseline_lambda": BASE_LAM,
        "lambdas": LAMBDAS,
        "n_subjects": len(df),
        "note": "Same NAI formula; only residual covariance regularization varies",
    }
    with open(OUT / "robustness_summary_cov.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=" * 72)

if __name__ == "__main__":
    main()