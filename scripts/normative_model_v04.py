from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv, pinv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
NORM_DIR = PROJECT_ROOT / "results" / "normative"
NORM_DIR.mkdir(parents=True, exist_ok=True)

SRC = FEATURES_DIR / "features_v032.csv"

FEATURES = [
    "alpha_rel",
    "beta_rel",
    "theta_rel",
    "spectral_entropy_mean",
    "log_theta_alpha",
    "log_theta_beta",
]

EPS = 1e-12
LAMBDA = 0.1          

def fit_age_models(df_td: pd.DataFrame) -> dict:
    models = {}
    age = df_td[["age"]].values

    for feat in FEATURES:
        y = df_td[feat].values
        reg = LinearRegression()
        reg.fit(age, y)
        models[feat] = {
            "intercept": float(reg.intercept_),
            "slope": float(reg.coef_[0]),
            "model": reg
        }
    return models

def predict_features(models: dict, age: np.ndarray) -> np.ndarray:
    preds = []
    for feat in FEATURES:
        m = models[feat]["model"]
        preds.append(m.predict(age.reshape(-1, 1)))
    return np.column_stack(preds)

def main():
    print("=" * 70)
    print("NAI v0.4 — Residual-based Normative Model")
    print("=" * 70)

    df = pd.read_csv(SRC)
    print(f"Total subjects in features_v032: {len(df)}")

    td = df[df["group"] == "TD"].copy().reset_index(drop=True)
    print(f"TD subjects for normative model: {len(td)}")

    models = fit_age_models(td)

    print("\nAge models (TD):")
    for feat, m in models.items():
        print(f"  {feat:25s}  intercept={m['intercept']:.4f}  slope={m['slope']:.4f}")

    X_td = td[FEATURES].values
    age_td = td["age"].values
    X_pred_td = predict_features(models, age_td)
    R_td = X_td - X_pred_td        

    Sigma = np.cov(R_td, rowvar=False)

    target = np.mean(np.diag(Sigma)) * np.eye(Sigma.shape[0])
    Sigma_reg = (1 - LAMBDA) * Sigma + LAMBDA * target

    cond_raw = np.linalg.cond(Sigma)
    cond_reg = np.linalg.cond(Sigma_reg)
    print(f"\nCovariance condition number:")
    print(f"  Raw      : {cond_raw:.2e}")
    print(f"  Regularized (λ={LAMBDA}): {cond_reg:.2e}")

    try:
        Sigma_inv = inv(Sigma_reg)
    except np.linalg.LinAlgError:
        print("  Warning: using pseudo-inverse")
        Sigma_inv = pinv(Sigma_reg)

    results = []

    for _, row in df.iterrows():
        age = row["age"]
        x = row[FEATURES].values.astype(float)

        x_pred = predict_features(models, np.array([age])).ravel()

        r = x - x_pred

        z = r / (np.sqrt(np.diag(Sigma_reg)) + EPS)

        d2 = r @ Sigma_inv @ r
        d_m = np.sqrt(max(d2, 0.0))

        rec = {
            "participant_id": row["participant_id"],
            "age": age,
            "sex": row.get("sex"),
            "group": row["group"],
            "qc_flag": row.get("qc_flag", "ok"),
            "n_runs": row.get("n_runs"),
            "mahalanobis_distance": d_m,
        }

        for i, feat in enumerate(FEATURES):
            rec[f"feat_{feat}"] = x[i]
            rec[f"pred_{feat}"] = x_pred[i]
            rec[f"resid_{feat}"] = r[i]
            rec[f"z_{feat}"] = z[i]

        results.append(rec)

    res_df = pd.DataFrame(results)

    res_df.to_csv(NORM_DIR / "normative_all_v04.csv", index=False)
    res_df[res_df["group"] == "TD"].to_csv(NORM_DIR / "normative_td_v04.csv", index=False)

    np.save(NORM_DIR / "covariance_v04.npy", Sigma_reg)
    np.save(NORM_DIR / "covariance_inv_v04.npy", Sigma_inv)

    summary = {
        "n_td": len(td),
        "features": FEATURES,
        "lambda": LAMBDA,
        "condition_number_raw": float(cond_raw),
        "condition_number_reg": float(cond_reg),
        "age_models": {
            feat: {"intercept": m["intercept"], "slope": m["slope"]}
            for feat, m in models.items()
        }
    }
    with open(NORM_DIR / "model_summary_v04.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "-" * 70)
    print("MAHALANOBIS DISTANCE SUMMARY")
    print("-" * 70)
    print(res_df.groupby("group")["mahalanobis_distance"].describe().round(3))

    print("\nTop 5 highest NAI (Mahalanobis):")
    print(res_df.nlargest(5, "mahalanobis_distance")[
        ["participant_id", "age", "group", "qc_flag", "mahalanobis_distance"]
    ].to_string(index=False))

    print("\nASD subjects:")
    print(res_df[res_df["group"] == "ASD"][
        ["participant_id", "age", "qc_flag", "mahalanobis_distance"] +
        [f"z_{f}" for f in FEATURES]
    ].round(3).to_string(index=False))

    print(f"\nSaved results to: {NORM_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()