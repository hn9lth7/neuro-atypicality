from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv, pinv, matrix_rank, cond

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
NORM_DIR = PROJECT_ROOT / "results" / "normative"
NORM_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_SE = [
    "alpha_rel", "beta_rel", "theta_rel",
    "spectral_entropy_mean",
    "log_theta_alpha", "log_theta_beta",
]

FEATURES_C = [
    "plv_mean_theta", "plv_mean_alpha", "plv_mean_beta", "plv_mean_gamma",
    "plv_median_theta", "plv_median_alpha", "plv_median_beta", "plv_median_gamma",
]

FEATURES_G = [
    f"{m}_{b}"
    for m in [
        "mean_degree", "degree_cv", "clustering",
        "global_efficiency", "mean_path_length", "laplacian_entropy"
    ]
    for b in ["theta", "alpha", "beta", "gamma"]
]

FEATURES_G_AUX = [f"lambda2_{b}" for b in ["theta", "alpha", "beta", "gamma"]]

LAMBDAS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
EPS = 1e-12

def fit_age_models(df: pd.DataFrame, features: list[str]) -> dict:
    models = {}
    age = df[["age"]].values
    for feat in features:
        y = df[feat].values
        reg = LinearRegression().fit(age, y)
        models[feat] = reg
    return models

def predict(models: dict, features: list[str], age: np.ndarray) -> np.ndarray:
    return np.column_stack([
        models[f].predict(age.reshape(-1, 1)) for f in features
    ])

def regularize(Sigma: np.ndarray, lam: float) -> np.ndarray:
    p = Sigma.shape[0]
    target = (np.trace(Sigma) / p) * np.eye(p)
    return (1 - lam) * Sigma + lam * target

def mahalanobis(r: np.ndarray, Sigma_inv: np.ndarray) -> float:
    d2 = float(r @ Sigma_inv @ r)
    return float(np.sqrt(max(d2, 0.0)))

def block_diagnostics(name: str, R: np.ndarray, lam: float = 0.10):
    Sigma = np.cov(R, rowvar=False)
    Sigma_reg = regularize(Sigma, lam)
    rank = matrix_rank(Sigma_reg, tol=1e-8)
    cond_raw = cond(Sigma)
    cond_reg = cond(Sigma_reg)
    print(f"\n[{name}]  n_features={R.shape[1]}  n_samples={R.shape[0]}")
    print(f"  Rank (reg)     : {rank}")
    print(f"  Cond raw       : {cond_raw:.2e}")
    print(f"  Cond reg (λ={lam}): {cond_reg:.2e}")
    return Sigma, Sigma_reg, rank, cond_raw, cond_reg

def main():
    print("=" * 72)
    print("NAI v0.5 — Hierarchical Normative Blocks")
    print("=" * 72)

    spectral = pd.read_csv(FEATURES_DIR / "features_v032.csv")
    conn = pd.read_csv(FEATURES_DIR / "connectivity_graph_subject_v0.5.csv")

    df = spectral.merge(
        conn.drop(columns=["age", "sex", "group"], errors="ignore"),
        on="participant_id",
        how="inner",
        suffixes=("", "_conn")
    )

    if "qc_flag" in df.columns:
        df = df[df["qc_flag"].isin(["ok", "very_low_alpha"])].copy()
    else:
        df = df.dropna(subset=["age", "group"]).copy()

    print(f"Subjects after QC filter: {len(df)}")
    print(df["group"].value_counts(dropna=False))

    td = df[df["group"] == "TD"].copy().reset_index(drop=True)
    print(f"TD for normative fit: {len(td)}")

    for block_name, feats in [("SE", FEATURES_SE), ("C", FEATURES_C), ("G", FEATURES_G)]:
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"WARNING [{block_name}] missing features: {missing}")
        else:
            print(f"[{block_name}] all {len(feats)} features present")

    blocks = {
        "SE": FEATURES_SE,
        "C": FEATURES_C,
        "G": FEATURES_G,
    }

    models = {}
    residuals_td = {}
    Sigma_reg = {}
    Sigma_inv = {}
    diag_info = {}

    for name, feats in blocks.items():
        models[name] = fit_age_models(td, feats)
        X = td[feats].values
        age = td["age"].values
        R = X - predict(models[name], feats, age)
        residuals_td[name] = R

        Sigma, S_reg, rank, c_raw, c_reg = block_diagnostics(name, R, lam=0.10)
        Sigma_reg[name] = S_reg
        try:
            Sigma_inv[name] = inv(S_reg)
        except np.linalg.LinAlgError:
            Sigma_inv[name] = pinv(S_reg)

        diag_info[name] = {
            "n_features": len(feats),
            "rank": int(rank),
            "cond_raw": float(c_raw),
            "cond_reg": float(c_reg),
        }

    results = []

    for _, row in df.iterrows():
        age = row["age"]
        rec = {
            "participant_id": row["participant_id"],
            "age": age,
            "sex": row.get("sex"),
            "group": row["group"],
            "qc_flag": row.get("qc_flag", "ok"),
        }

        D = {}
        for name, feats in blocks.items():
            x = row[feats].values.astype(float)
            x_pred = predict(models[name], feats, np.array([age])).ravel()
            r = x - x_pred
            d = mahalanobis(r, Sigma_inv[name])
            D[name] = d
            rec[f"D_{name}"] = d

            if name == "SE":
                sigma = np.sqrt(np.diag(Sigma_reg[name])) + EPS
                z = r / sigma
                for i, f in enumerate(feats):
                    rec[f"z_{f}"] = z[i]

        D_CG = 0.5 * (D["C"] + D["G"])
        NAI = 0.5 * (D["SE"] + D_CG)

        rec["D_CG"] = D_CG
        rec["NAI_v05"] = NAI
        results.append(rec)

    res_df = pd.DataFrame(results)

    print("\n" + "-" * 72)
    print("λ SENSITIVITY (mean NAI on TD / max NAI on ASD)")
    print("-" * 72)

    sens = []
    for lam in LAMBDAS:
        invs = {}
        for name in blocks:
            S = regularize(np.cov(residuals_td[name], rowvar=False), lam)
            try:
                invs[name] = inv(S)
            except np.linalg.LinAlgError:
                invs[name] = pinv(S)

        nai_td, nai_asd = [], []
        for _, row in df.iterrows():
            age = row["age"]
            Ds = {}
            for name, feats in blocks.items():
                x = row[feats].values.astype(float)
                r = x - predict(models[name], feats, np.array([age])).ravel()
                Ds[name] = mahalanobis(r, invs[name])
            nai = 0.5 * (Ds["SE"] + 0.5 * (Ds["C"] + Ds["G"]))
            if row["group"] == "TD":
                nai_td.append(nai)
            elif row["group"] == "ASD":
                nai_asd.append(nai)

        print(f"  λ={lam:4.2f}  TD mean={np.mean(nai_td):.3f}  "
              f"ASD = {[f'{v:.3f}' for v in nai_asd]}")
        sens.append({"lambda": lam, "td_mean": float(np.mean(nai_td)),
                     "asd": [float(v) for v in nai_asd]})

    res_df.to_csv(NORM_DIR / "normative_blocks_v05.csv", index=False)

    for name in blocks:
        np.save(NORM_DIR / f"covariance_{name}_v05.npy", Sigma_reg[name])

    summary = {
        "n_td": len(td),
        "blocks": diag_info,
        "features": {
            "SE": FEATURES_SE,
            "C": FEATURES_C,
            "G": FEATURES_G,
            "G_aux_lambda2": FEATURES_G_AUX,
        },
        "lambda_sensitivity": sens,
    }
    with open(NORM_DIR / "model_summary_v05.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "-" * 72)
    print("NAI_v05 SUMMARY")
    print("-" * 72)
    print(res_df.groupby("group")[["D_SE", "D_C", "D_G", "D_CG", "NAI_v05"]]
          .describe().round(3).to_string())

    print("\nTop 5 NAI_v05:")
    print(res_df.nlargest(5, "NAI_v05")[
        ["participant_id", "group", "qc_flag", "D_SE", "D_C", "D_G", "NAI_v05"]
    ].round(3).to_string(index=False))

    print("\nASD detail:")
    print(res_df[res_df["group"] == "ASD"][
        ["participant_id", "D_SE", "D_C", "D_G", "D_CG", "NAI_v05"]
    ].round(3).to_string(index=False))

    print(f"\nSaved → {NORM_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()