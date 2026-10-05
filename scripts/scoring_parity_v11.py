from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.blocks import SE_FEATURES, C_FEATURES, G_FEATURES, D_FEATURES, ALL_BLOCKS
from nai.normative.regression import fit_age_models, compute_residuals
from nai.normative.covariance import empirical_covariance, regularize_covariance
from nai.normative.mahalanobis import mahalanobis_distance
from nai.nai.composite import compute_nai

FEAT = PROJECT_ROOT / "results" / "features"
NORM = PROJECT_ROOT / "results" / "normative"
LAMBDA = 0.10

EXCLUDE = {"sub-10777", "sub-10950"} 

def load_merged() -> pd.DataFrame:
    se = pd.read_csv(FEAT / "features_v032.csv")
    cg = pd.read_csv(FEAT / "connectivity_graph_subject_v0.5.csv")
    dy = pd.read_csv(FEAT / "dynamic_subject_v0.6.csv")

    for name, df in [("SE", se), ("CG", cg), ("D", dy)]:
        n = df["participant_id"].nunique()
        if len(df) != n:
            raise ValueError(f"{name}: expected 1 row/subject, got {len(df)}/{n}")

    df = se.merge(cg, on="participant_id", how="inner", suffixes=("", "_cg"))
    df = df.merge(dy, on="participant_id", how="inner", suffixes=("", "_dy"))

    if "age" not in df.columns and "age_cg" in df.columns:
        df["age"] = df["age_cg"]
    if "group" not in df.columns and "group_cg" in df.columns:
        df["group"] = df["group_cg"]

    df = df[~df["participant_id"].astype(str).isin(EXCLUDE)].copy()
    df = df.dropna(subset=["age"])
    df = df[df["group"].isin(["TD", "ASD"])].copy()
    return df.reset_index(drop=True)

def block_distances(df: pd.DataFrame, features: list[str], lam: float) -> np.ndarray:
    X = df[features].to_numpy(dtype=float)
    age = df["age"].to_numpy(dtype=float)
    td = df["group"].to_numpy() == "TD"
    X_td, age_td = X[td], age[td]

    models = fit_age_models(X_td, age_td)
    R_td = compute_residuals(X_td, age_td, models)
    Sigma = regularize_covariance(empirical_covariance(R_td), lam)
    R_all = compute_residuals(X, age, models)

    return np.array([mahalanobis_distance(R_all[i], Sigma) for i in range(len(df))])

def main() -> None:
    print("=" * 72)
    print("v1.1 scoring parity  |  λ =", LAMBDA)
    print("=" * 72)

    df = load_merged()
    print(f"Subjects: {len(df)}  TD={(df.group=='TD').sum()}  ASD={(df.group=='ASD').sum()}")

    for name, feats in ALL_BLOCKS.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            raise KeyError(f"{name} missing: {missing}")

    d_se = block_distances(df, SE_FEATURES, LAMBDA)
    d_c = block_distances(df, C_FEATURES, LAMBDA)
    d_g = block_distances(df, G_FEATURES, LAMBDA)
    d_d = block_distances(df, D_FEATURES, LAMBDA)

    nai = np.array(
        [
            float(
                compute_nai(
                    {"SE": d_se[i], "C": d_c[i], "G": d_g[i], "D": d_d[i]}
                )
            )
            for i in range(len(df))
        ]
    )

    out = df[["participant_id", "age", "group"]].copy()
    out["D_SE"] = d_se
    out["D_C"] = d_c
    out["D_G"] = d_g
    out["D_D"] = d_d
    out["NAI_v11"] = nai

    frozen = pd.read_csv(NORM / "nai_v10.csv")
    nai_col = "NAI_v10" if "NAI_v10" in frozen.columns else "NAI"
    m = out.merge(
        frozen[
            ["participant_id", "D_SE", "D_C", "D_G", "D_D", nai_col]
        ].rename(
            columns={
                "D_SE": "D_SE_frz",
                "D_C": "D_C_frz",
                "D_G": "D_G_frz",
                "D_D": "D_D_frz",
                nai_col: "NAI_frz",
            }
        ),
        on="participant_id",
        how="inner",
    )

    print(f"Matched subjects: {len(m)}")
    print("-" * 72)

    cols = [
        ("D_SE", "D_SE_frz"),
        ("D_C", "D_C_frz"),
        ("D_G", "D_G_frz"),
        ("D_D", "D_D_frz"),
        ("NAI_v11", "NAI_frz"),
    ]
    for a, b in cols:
        ad = (m[a] - m[b]).abs()
        print(
            f"  {a:8s}  max|Δ|={ad.max():.3e}  median|Δ|={ad.median():.3e}  "
            f"mean|Δ|={ad.mean():.3e}"
        )

    tol = 1e-6
    bad = m[(m["NAI_v11"] - m["NAI_frz"]).abs() > tol]
    print("-" * 72)
    if bad.empty:
        print(f"SCORING PARITY PASS  (all |Δ NAI| ≤ {tol})")
    else:
        print(f"SCORING PARITY FAIL  {len(bad)} subjects |Δ NAI| > {tol}")
        print(
            bad[
                ["participant_id", "group", "NAI_v11", "NAI_frz"]
            ].to_string(index=False)
        )

    for sid in ("sub-11025", "sub-11038", "sub-10025"):
        r = m[m["participant_id"] == sid]
        if r.empty:
            continue
        r = r.iloc[0]
        print(
            f"  {sid}: NAI v11={r['NAI_v11']:.6f}  frz={r['NAI_frz']:.6f}  "
            f"|Δ|={abs(r['NAI_v11']-r['NAI_frz']):.3e}"
        )

    out_path = NORM / "scoring_parity_v11.csv"
    m.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    main()