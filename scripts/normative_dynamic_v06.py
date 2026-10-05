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

FEATURES_D = [
    f"{m}_{b}"
    for m in ["mean_delta", "cv_delta", "mean_degree_cv", "temporal_cv_degree_cv"]
    for b in ["theta", "alpha", "beta", "gamma"]
]

BLOCKS = {
    "SE": FEATURES_SE,
    "C": FEATURES_C,
    "G": FEATURES_G,
    "D": FEATURES_D,
}

LAMBDAS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
LAMBDA_REF = 0.10
EPS = 1e-12

def fit_age_models(df, features):
    models = {}
    age = df[["age"]].values
    for f in features:
        models[f] = LinearRegression().fit(age, df[f].values)
    return models

def predict(models, features, age):
    return np.column_stack([
        models[f].predict(age.reshape(-1, 1)) for f in features
    ])

def regularize(Sigma, lam):
    p = Sigma.shape[0]
    target = (np.trace(Sigma) / p) * np.eye(p)
    return (1 - lam) * Sigma + lam * target

def mahalanobis(r, Sigma_inv):
    d2 = float(r @ Sigma_inv @ r)
    return float(np.sqrt(max(d2, 0.0)))

def main():
    print("=" * 72)
    print("NAI v0.6 — Dynamic Normative Block + Hierarchical NAI")
    print("=" * 72)

    spectral = pd.read_csv(FEATURES_DIR / "features_v032.csv")
    conn = pd.read_csv(FEATURES_DIR / "connectivity_graph_subject_v0.5.csv")
    dyn = pd.read_csv(FEATURES_DIR / "dynamic_subject_v0.6.csv")

    df = spectral.merge(
        conn.drop(columns=["age", "sex", "group"], errors="ignore"),
        on="participant_id", how="inner"
    )
    df = df.merge(
        dyn.drop(columns=["age", "sex", "group", "n_runs"], errors="ignore"),
        on="participant_id", how="inner"
    )

    if "qc_flag" in df.columns:
        df = df[df["qc_flag"].isin(["ok", "very_low_alpha"])].copy()
    df = df[df["participant_id"] != "sub-10777"].copy()
    df = df.dropna(subset=["age", "group"]).copy()

    td = df[df["group"] == "TD"].copy().reset_index(drop=True)
    asd = df[df["group"] == "ASD"].copy().reset_index(drop=True)

    print(f"Canonical : {len(df)}  (TD={len(td)}, ASD={len(asd)})")

    for name, feats in BLOCKS.items():
        miss = [f for f in feats if f not in df.columns]
        if miss:
            print(f"WARNING [{name}] missing: {miss}")
        else:
            print(f"[{name}] {len(feats)} features OK")

    models = {}
    residuals = {}
    Sigma_reg = {}
    Sigma_inv = {}

    print("\n" + "-" * 72)
    print("BLOCK DIAGNOSTICS (TD, λ=0.10)")
    print("-" * 72)

    for name, feats in BLOCKS.items():
        models[name] = fit_age_models(td, feats)
        X = td[feats].values.astype(float)
        age = td["age"].values
        R = X - predict(models[name], feats, age)
        residuals[name] = R

        Sigma = np.cov(R, rowvar=False)
        S_reg = regularize(Sigma, LAMBDA_REF)
        rank = matrix_rank(S_reg, tol=1e-8)
        c_raw = cond(Sigma)
        c_reg = cond(S_reg)

        print(f"  [{name:2s}] n={len(feats):2d}  rank={rank:2d}  "
              f"cond_raw={c_raw:.2e}  cond_reg={c_reg:.2e}")

        Sigma_reg[name] = S_reg
        try:
            Sigma_inv[name] = inv(S_reg)
        except np.linalg.LinAlgError:
            Sigma_inv[name] = pinv(S_reg)

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
        for name, feats in BLOCKS.items():
            x = row[feats].values.astype(float)
            r = x - predict(models[name], feats, np.array([age])).ravel()
            d = mahalanobis(r, Sigma_inv[name])
            D[name] = d
            rec[f"D_{name}"] = d

            if name in ("SE", "D"):
                sigma = np.sqrt(np.diag(Sigma_reg[name])) + EPS
                z = r / sigma
                for i, f in enumerate(feats):
                    rec[f"z_{f}"] = z[i]

        nai = 0.25 * (D["SE"] + D["C"] + D["G"] + D["D"])
        rec["NAI_v06"] = nai
        results.append(rec)

    res_df = pd.DataFrame(results)

    print("\n" + "-" * 72)
    print("λ SENSITIVITY")
    print("-" * 72)
    print(f"{'λ':>6s}  {'mean_DD_TD':>11s}  {'max_DD_TD':>10s}  "
          f"{'ASD_11025':>10s}  {'ASD_11038':>10s}  {'NAI_11025':>10s}")

    for lam in LAMBDAS:
        invs = {}
        for name in BLOCKS:
            S = regularize(np.cov(residuals[name], rowvar=False), lam)
            try:
                invs[name] = inv(S)
            except np.linalg.LinAlgError:
                invs[name] = pinv(S)

        dd_td = []
        for _, row in td.iterrows():
            x = row[FEATURES_D].values.astype(float)
            r = x - predict(models["D"], FEATURES_D, np.array([row["age"]])).ravel()
            dd_td.append(mahalanobis(r, invs["D"]))
        dd_td = np.array(dd_td)

        asd_vals = {}
        for _, row in asd.iterrows():
            Ds = {}
            for name, feats in BLOCKS.items():
                x = row[feats].values.astype(float)
                r = x - predict(models[name], feats, np.array([row["age"]])).ravel()
                Ds[name] = mahalanobis(r, invs[name])
            asd_vals[row["participant_id"]] = Ds

        nai_11025 = 0.25 * sum(asd_vals["sub-11025"][k] for k in BLOCKS)
        print(f"{lam:6.2f}  {dd_td.mean():11.3f}  {dd_td.max():10.3f}  "
              f"{asd_vals['sub-11025']['D']:10.3f}  "
              f"{asd_vals['sub-11038']['D']:10.3f}  "
              f"{nai_11025:10.3f}")

    print("\n" + "-" * 72)
    print("NAI_v06 SUMMARY")
    print("-" * 72)
    print(res_df.groupby("group")[["D_SE", "D_C", "D_G", "D_D", "NAI_v06"]]
          .describe().round(3).to_string())

    print("\nTop 5 NAI_v06:")
    print(res_df.nlargest(5, "NAI_v06")[
        ["participant_id", "group", "qc_flag", "D_SE", "D_C", "D_G", "D_D", "NAI_v06"]
    ].round(3).to_string(index=False))

    print("\nASD detail:")
    print(res_df[res_df["group"] == "ASD"][
        ["participant_id", "D_SE", "D_C", "D_G", "D_D", "NAI_v06"]
    ].round(3).to_string(index=False))

    res_df.to_csv(NORM_DIR / "normative_blocks_v06.csv", index=False)
    for name in BLOCKS:
        np.save(NORM_DIR / f"covariance_{name}_v06.npy", Sigma_reg[name])

    summary = {
        "n_td": len(td),
        "n_asd": len(asd),
        "lambda_ref": LAMBDA_REF,
        "blocks": {k: len(v) for k, v in BLOCKS.items()},
        "weights": {"SE": 0.25, "C": 0.25, "G": 0.25, "D": 0.25},
    }
    with open(NORM_DIR / "model_summary_v06.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved → {NORM_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()