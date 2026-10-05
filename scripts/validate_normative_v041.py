from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv, pinv, eigvalsh
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
NORM_DIR = PROJECT_ROOT / "results" / "normative"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "validation_v041"
FIG_DIR.mkdir(parents=True, exist_ok=True)
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

LAMBDAS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
EPS = 1e-12

def fit_age_models(df: pd.DataFrame) -> dict:
    models = {}
    age = df[["age"]].values
    for feat in FEATURES:
        reg = LinearRegression().fit(age, df[feat].values)
        models[feat] = reg
    return models

def predict(models: dict, age: np.ndarray) -> np.ndarray:
    preds = [models[f].predict(age.reshape(-1, 1)) for f in FEATURES]
    return np.column_stack(preds)

def regularize_cov(Sigma: np.ndarray, lam: float) -> np.ndarray:
    if lam <= 0:
        return Sigma.copy()
    target = np.mean(np.diag(Sigma)) * np.eye(Sigma.shape[0])
    return (1 - lam) * Sigma + lam * target

def mahalanobis(r: np.ndarray, Sigma_inv: np.ndarray) -> float:
    d2 = float(r @ Sigma_inv @ r)
    return np.sqrt(max(d2, 0.0))

def main():
    print("=" * 72)
    print("NAI v0.4.1 — LOO Validation + λ Sensitivity")
    print("=" * 72)

    df = pd.read_csv(SRC)
    td = df[df["group"] == "TD"].copy().reset_index(drop=True)
    asd = df[df["group"] == "ASD"].copy().reset_index(drop=True)

    print(f"TD subjects : {len(td)}")
    print(f"ASD subjects: {len(asd)}")

    n_feat = len(FEATURES)
    X_td = td[FEATURES].values
    age_td = td["age"].values

    print("\n" + "-" * 72)
    print("1. LEAVE-ONE-OUT MAHALANOBIS (TD)")
    print("-" * 72)

    loo_results = {lam: [] for lam in LAMBDAS}

    for i in range(len(td)):
        mask = np.ones(len(td), dtype=bool)
        mask[i] = False

        td_train = td.iloc[mask]
        models = fit_age_models(td_train)

        X_train = td_train[FEATURES].values
        age_train = td_train["age"].values
        R_train = X_train - predict(models, age_train)

        Sigma = np.cov(R_train, rowvar=False)

        x_i = X_td[i]
        age_i = age_td[i]
        r_i = x_i - predict(models, np.array([age_i])).ravel()

        for lam in LAMBDAS:
            Sigma_reg = regularize_cov(Sigma, lam)
            try:
                Sigma_inv = inv(Sigma_reg)
            except np.linalg.LinAlgError:
                Sigma_inv = pinv(Sigma_reg)

            d = mahalanobis(r_i, Sigma_inv)
            loo_results[lam].append(d)

    for lam in LAMBDAS:
        loo_results[lam] = np.array(loo_results[lam])

    print("LOO Mahalanobis (TD) — summary:")
    for lam in LAMBDAS:
        d = loo_results[lam]
        print(f"  λ={lam:4.2f}  mean={d.mean():.3f}  median={np.median(d):.3f}  "
              f"std={d.std():.3f}  max={d.max():.3f}")

    print("\n" + "-" * 72)
    print("2. ASD SCORING (full TD model) for each λ")
    print("-" * 72)

    models_full = fit_age_models(td)
    R_td_full = X_td - predict(models_full, age_td)
    Sigma_full = np.cov(R_td_full, rowvar=False)

    asd_dm = {lam: [] for lam in LAMBDAS}
    cond_numbers = {}
    eigvals = {}

    for lam in LAMBDAS:
        Sigma_reg = regularize_cov(Sigma_full, lam)
        cond_numbers[lam] = float(np.linalg.cond(Sigma_reg))
        eigvals[lam] = eigvalsh(Sigma_reg)

        try:
            Sigma_inv = inv(Sigma_reg)
        except np.linalg.LinAlgError:
            Sigma_inv = pinv(Sigma_reg)

        for _, row in asd.iterrows():
            x = row[FEATURES].values.astype(float)
            r = x - predict(models_full, np.array([row["age"]])).ravel()
            d = mahalanobis(r, Sigma_inv)
            asd_dm[lam].append(d)

        print(f"  λ={lam:4.2f}  cond={cond_numbers[lam]:.2e}  "
              f"ASD D_M = {[f'{d:.3f}' for d in asd_dm[lam]]}")

    print("\n" + "-" * 72)
    print("3. STABILITY ACROSS λ (Spearman correlation of LOO D_M)")
    print("-" * 72)

    loo_df = pd.DataFrame({f"λ={lam}": loo_results[lam] for lam in LAMBDAS})
    corr = loo_df.corr(method="spearman")
    print(corr.round(3).to_string())

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0.8,
                vmin=0.5, vmax=1.0, square=True)
    plt.title("Spearman correlation of LOO D_M across λ")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lambda_stability_corr.png", dpi=140)
    plt.close()

    print("\n" + "-" * 72)
    print("4. EMPIRICAL PERCENTILE CALIBRATION (λ = 0.10)")
    print("-" * 72)

    lam_ref = 0.10
    loo_ref = loo_results[lam_ref]

    chi2_95 = np.sqrt(stats.chi2.ppf(0.95, df=n_feat))
    chi2_99 = np.sqrt(stats.chi2.ppf(0.99, df=n_feat))

    print(f"Theoretical √χ²_6  95% = {chi2_95:.3f}")
    print(f"Theoretical √χ²_6  99% = {chi2_99:.3f}")
    print(f"Empirical LOO TD  95% = {np.percentile(loo_ref, 95):.3f}")
    print(f"Empirical LOO TD  99% = {np.percentile(loo_ref, 99):.3f}")

    print("\nASD empirical percentiles (relative to LOO TD distribution):")
    for i, (_, row) in enumerate(asd.iterrows()):
        d = asd_dm[lam_ref][i]
        pct = stats.percentileofscore(loo_ref, d, kind="rank")
        print(f"  {row['participant_id']}  D_M={d:.3f}  →  {pct:.1f}-th percentile of LOO TD")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(loo_ref, bins=12, color="steelblue", alpha=0.75, edgecolor="k",
            label="TD LOO D_M")
    for i, d in enumerate(asd_dm[lam_ref]):
        ax.axvline(d, color="crimson" if i == 0 else "orange", lw=2,
                   label=f"ASD {asd.iloc[i]['participant_id']}")
    ax.axvline(chi2_95, color="gray", ls="--", label=f"√χ²_6 95% = {chi2_95:.2f}")
    ax.set_xlabel("Mahalanobis distance")
    ax.set_ylabel("Count")
    ax.set_title("LOO TD distribution vs ASD (λ = 0.10)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "loo_distribution_calibration.png", dpi=140)
    plt.close()

    loo_out = td[["participant_id", "age"]].copy()
    for lam in LAMBDAS:
        loo_out[f"DM_LOO_λ{lam}"] = loo_results[lam]
    loo_out.to_csv(NORM_DIR / "loo_td_v041.csv", index=False)

    summary = {
        "n_td": len(td),
        "n_asd": len(asd),
        "features": FEATURES,
        "lambdas": LAMBDAS,
        "condition_numbers": cond_numbers,
        "loo_mean": {str(lam): float(loo_results[lam].mean()) for lam in LAMBDAS},
        "loo_max": {str(lam): float(loo_results[lam].max()) for lam in LAMBDAS},
        "asd_dm": {
            str(lam): [float(d) for d in asd_dm[lam]] for lam in LAMBDAS
        },
        "theoretical_chi2_sqrt_95": float(chi2_95),
        "empirical_loo_95": float(np.percentile(loo_ref, 95)),
    }
    with open(NORM_DIR / "validation_summary_v041.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved LOO table → {NORM_DIR / 'loo_td_v041.csv'}")
    print(f"Saved summary   → {NORM_DIR / 'validation_summary_v041.json'}")
    print(f"Figures         → {FIG_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()