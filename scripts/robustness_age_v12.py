from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.blocks import C_FEATURES, D_FEATURES, G_FEATURES, SE_FEATURES
from nai.nai.composite import compute_nai
from nai.normative.covariance import regularize_covariance
from nai.normative.mahalanobis import mahalanobis_distance

FEAT = PROJECT_ROOT / "results" / "features"
OUT = PROJECT_ROOT / "results" / "normative" / "robustness_v12"
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
    return (
        df[df["group"].isin(["TD", "ASD"])]
        .dropna(subset=["age"])
        .reset_index(drop=True)
    )

def fit_age_models_poly(X: np.ndarray, age: np.ndarray, degree: int) -> list:
    age = np.asarray(age, dtype=float).reshape(-1, 1)
    if degree == 1:
        A = age
    elif degree == 2:
        A = np.hstack([age, age ** 2])
    else:
        raise ValueError("degree must be 1 or 2")
    models = []
    for j in range(X.shape[1]):
        m = LinearRegression()
        m.fit(A, X[:, j])
        models.append(m)
    return models

def residuals_poly(X: np.ndarray, age: np.ndarray, models: list, degree: int) -> np.ndarray:
    age = np.asarray(age, dtype=float).reshape(-1, 1)
    if degree == 1:
        A = age
    else:
        A = np.hstack([age, age ** 2])
    pred = np.column_stack([m.predict(A) for m in models])
    return X - pred

def score_cohort(df: pd.DataFrame, degree: int) -> pd.DataFrame:
    td = df[df["group"] == "TD"]
    age_td = td["age"].to_numpy(float)
    models, covs = {}, {}
    for name, feats in BLOCK_FEATS.items():
        X = td[feats].to_numpy(float)
        models[name] = fit_age_models_poly(X, age_td, degree)
        R = residuals_poly(X, age_td, models[name], degree)
        Sigma = np.cov(R, rowvar=False)
        if Sigma.ndim == 0:
            Sigma = np.array([[float(Sigma)]])
        covs[name] = regularize_covariance(Sigma, LAMBDA)

    rows = []
    for _, row in df.iterrows():
        d = {}
        age = np.array([float(row["age"])])
        for name, feats in BLOCK_FEATS.items():
            x = row[feats].to_numpy(float).reshape(1, -1)
            r = residuals_poly(x, age, models[name], degree)[0]
            d[name] = float(mahalanobis_distance(r, covs[name]))
        nai = float(compute_nai({k: d[k] for k in ("SE", "C", "G", "D")}))
        rows.append(
            {
                "participant_id": row["participant_id"],
                "group": row["group"],
                "age": float(row["age"]),
                "D_SE": d["SE"],
                "D_C": d["C"],
                "D_G": d["G"],
                "D_D": d["D"],
                "NAI": nai,
            }
        )
    return pd.DataFrame(rows)

def main() -> None:
    print("=" * 72)
    print("NAI v1.2 — Age-model sensitivity (linear vs quadratic)")
    print("=" * 72)

    df = load_merged()
    print(f"Subjects: {len(df)}")

    print("  linear (degree=1) ...")
    lin = score_cohort(df, degree=1)
    print("  quadratic (degree=2) ...")
    quad = score_cohort(df, degree=2)

    m = lin.merge(
        quad,
        on=["participant_id", "group", "age"],
        suffixes=("_lin", "_quad"),
    )
    out_csv = OUT / "age_model_sensitivity.csv"
    m.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    rho_all, _ = spearmanr(m["NAI_lin"], m["NAI_quad"])
    rho_td, _ = spearmanr(
        m.loc[m.group == "TD", "NAI_lin"], m.loc[m.group == "TD", "NAI_quad"]
    )
    rho_asd, _ = spearmanr(
        m.loc[m.group == "ASD", "NAI_lin"], m.loc[m.group == "ASD", "NAI_quad"]
    )
    print(f"\nSpearman NAI linear↔quadratic:")
    print(f"  all={rho_all:.4f}  TD={rho_td:.4f}  ASD={rho_asd:.4f}")

    td_max = float(m.loc[m.group == "TD", "age"].max())
    old = m[m["age"] > td_max]
    print(f"\nTD age max = {td_max:.1f}")
    if len(old):
        print("Subjects with age > TD max (extrapolation):")
        print(
            old[
                [
                    "participant_id",
                    "group",
                    "age",
                    "NAI_lin",
                    "NAI_quad",
                    "D_SE_lin",
                    "D_SE_quad",
                ]
            ].to_string(index=False)
        )
    else:
        print("No subjects beyond TD age max.")

    m["rank_lin"] = m["NAI_lin"].rank(ascending=False)
    m["rank_quad"] = m["NAI_quad"].rank(ascending=False)
    m["rank_shift"] = (m["rank_lin"] - m["rank_quad"]).abs()
    print("\nLargest |rank| shifts (top 5):")
    print(
        m.nlargest(5, "rank_shift")[
            ["participant_id", "group", "age", "NAI_lin", "NAI_quad", "rank_shift"]
        ].to_string(index=False)
    )

    summary = {
        "lambda": LAMBDA,
        "spearman_all": float(rho_all),
        "spearman_TD": float(rho_td),
        "spearman_ASD": float(rho_asd),
        "td_age_max": td_max,
        "n_extrapolated": int(len(old)),
    }
    with open(OUT / "age_model_sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {OUT / 'age_model_sensitivity_summary.json'}")
    print("=" * 72)

if __name__ == "__main__":
    main()