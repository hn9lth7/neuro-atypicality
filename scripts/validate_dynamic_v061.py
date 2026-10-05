from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv, pinv
from scipy import stats
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
NORM_DIR = PROJECT_ROOT / "results" / "normative"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "validation_v061"
FIG_DIR.mkdir(parents=True, exist_ok=True)
NORM_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_D = [
    f"{m}_{b}"
    for m in ["mean_delta", "cv_delta", "mean_degree_cv", "temporal_cv_degree_cv"]
    for b in ["theta", "alpha", "beta", "gamma"]
]

LAMBDAS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
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
    print("NAI v0.6.1 — LOO Validation of Dynamic Block D_D")
    print("=" * 72)

    spectral = pd.read_csv(FEATURES_DIR / "features_v032.csv")
    dyn = pd.read_csv(FEATURES_DIR / "dynamic_subject_v0.6.csv")

    df = spectral.merge(
        dyn.drop(columns=["age", "sex", "group", "n_runs"], errors="ignore"),
        on="participant_id", how="inner"
    )

    if "qc_flag" in df.columns:
        df = df[df["qc_flag"].isin(["ok", "very_low_alpha"])].copy()
    df = df[df["participant_id"] != "sub-10777"].copy()
    df = df.dropna(subset=["age", "group"]).copy()

    td = df[df["group"] == "TD"].copy().reset_index(drop=True)
    asd = df[df["group"] == "ASD"].copy().reset_index(drop=True)

    print(f"TD subjects : {len(td)}")
    print(f"ASD subjects: {len(asd)}")

    print("\n" + "-" * 72)
    print("1. LEAVE-ONE-OUT D_D (TD)")
    print("-" * 72)

    loo = {lam: [] for lam in LAMBDAS}
    n_td = len(td)

    for i in range(n_td):
        mask = np.ones(n_td, dtype=bool)
        mask[i] = False
        train = td.iloc[mask]

        models = fit_age_models(train, FEATURES_D)
        X = train[FEATURES_D].values.astype(float)
        age_train = train["age"].values
        R = X - predict(models, FEATURES_D, age_train)

        row = td.iloc[i]
        age_i = row["age"]
        x_i = row[FEATURES_D].values.astype(float)
        r_i = x_i - predict(models, FEATURES_D, np.array([age_i])).ravel()

        for lam in LAMBDAS:
            S = regularize(np.cov(R, rowvar=False), lam)
            try:
                S_inv = inv(S)
            except np.linalg.LinAlgError:
                S_inv = pinv(S)
            loo[lam].append(mahalanobis(r_i, S_inv))

        if (i + 1) % 10 == 0 or i == n_td - 1:
            print(f"  processed {i+1}/{n_td}")

    for lam in LAMBDAS:
        loo[lam] = np.array(loo[lam])

    print("\nLOO D_D (TD) summary:")
    print(f"{'λ':>6s}  {'mean':>8s}  {'median':>8s}  {'std':>8s}  {'max':>8s}")
    for lam in LAMBDAS:
        a = loo[lam]
        print(f"{lam:6.2f}  {a.mean():8.3f}  {np.median(a):8.3f}  "
              f"{a.std():8.3f}  {a.max():8.3f}")

    print("\n" + "-" * 72)
    print("2. ASD scoring (full TD model)")
    print("-" * 72)

    models_full = fit_age_models(td, FEATURES_D)
    X_full = td[FEATURES_D].values.astype(float)
    R_full = X_full - predict(models_full, FEATURES_D, td["age"].values)

    print(f"{'λ':>6s}  {'ASD_11025':>10s}  {'ASD_11038':>10s}")
    asd_scores = {lam: {} for lam in LAMBDAS}

    for lam in LAMBDAS:
        S = regularize(np.cov(R_full, rowvar=False), lam)
        try:
            S_inv = inv(S)
        except np.linalg.LinAlgError:
            S_inv = pinv(S)

        for _, row in asd.iterrows():
            x = row[FEATURES_D].values.astype(float)
            r = x - predict(models_full, FEATURES_D, np.array([row["age"]])).ravel()
            d = mahalanobis(r, S_inv)
            asd_scores[lam][row["participant_id"]] = d

        print(f"{lam:6.2f}  {asd_scores[lam]['sub-11025']:10.3f}  "
              f"{asd_scores[lam]['sub-11038']:10.3f}")

    print("\n" + "-" * 72)
    print("3. EMPIRICAL PERCENTILES (λ = 0.10, LOO TD reference)")
    print("-" * 72)

    lam_ref = 0.10
    dist = loo[lam_ref]
    print(f"  LOO TD  95% = {np.percentile(dist, 95):.3f}")
    print(f"  LOO TD  99% = {np.percentile(dist, 99):.3f}")

    for pid, d in asd_scores[lam_ref].items():
        pct = stats.percentileofscore(dist, d, kind="rank")
        print(f"  {pid}  D_D={d:.3f}  →  {pct:.1f}-th percentile")

    loo_df = td[["participant_id", "age"]].copy()
    for lam in LAMBDAS:
        loo_df[f"D_D_LOO_λ{lam}"] = loo[lam]
    loo_df.to_csv(NORM_DIR / "loo_dynamic_v061.csv", index=False)

    summary = {
        "n_td": len(td),
        "n_asd": len(asd),
        "loo_means": {str(lam): float(loo[lam].mean()) for lam in LAMBDAS},
        "loo_max": {str(lam): float(loo[lam].max()) for lam in LAMBDAS},
        "asd": {str(lam): asd_scores[lam] for lam in LAMBDAS},
        "p95_loo": float(np.percentile(dist, 95)),
        "p99_loo": float(np.percentile(dist, 99)),
    }
    with open(NORM_DIR / "validation_dynamic_v061.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dist, bins=12, color="steelblue", alpha=0.75, edgecolor="k",
            label="TD LOO")
    for pid, d in asd_scores[lam_ref].items():
        ax.axvline(d, lw=2, label=f"{pid} ({d:.2f})")
    ax.axvline(np.percentile(dist, 95), color="gray", ls="--", label="P95")
    ax.set_xlabel(r"$D_D$ (Mahalanobis)")
    ax.set_title(r"LOO $D_D$ distribution (λ = 0.10)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "loo_DD_distribution.png", dpi=140)
    plt.close()

    print(f"\nSaved LOO table → {NORM_DIR / 'loo_dynamic_v061.csv'}")
    print(f"Saved summary  → {NORM_DIR / 'validation_dynamic_v061.json'}")
    print(f"Figures        → {FIG_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()