from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import matrix_rank, cond

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "feateng_v032"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SRC = FEATURES_DIR / "participants_features_subject_v0.3_clean.csv"
OUT_FEATURES = FEATURES_DIR / "features_v032.csv"
OUT_TD = FEATURES_DIR / "features_v032_TD.csv"

EPS = 1e-12

def main():
    print("=" * 70)
    print("NAI v0.3.2 — Feature Engineering")
    print("=" * 70)

    df = pd.read_csv(SRC)
    print(f"Clean subjects: {len(df)}")
    print(df["group"].value_counts())

    df["log_theta_alpha"] = np.log((df["theta_abs"] + EPS) / (df["alpha_abs"] + EPS))
    df["log_theta_beta"]  = np.log((df["theta_abs"] + EPS) / (df["beta_abs"]  + EPS))
    df["log_alpha_beta"]  = np.log((df["alpha_abs"] + EPS) / (df["beta_abs"]  + EPS))

    feature_sets = {
        "full_rel": [
            "delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel",
            "spectral_entropy_mean",
            "log_theta_alpha", "log_theta_beta", "log_alpha_beta"
        ],
        "drop_delta": [
            "theta_rel", "alpha_rel", "beta_rel", "gamma_rel",
            "spectral_entropy_mean",
            "log_theta_alpha", "log_theta_beta", "log_alpha_beta"
        ],
        "drop_gamma": [
            "delta_rel", "theta_rel", "alpha_rel", "beta_rel",
            "spectral_entropy_mean",
            "log_theta_alpha", "log_theta_beta", "log_alpha_beta"
        ],
        "core": [
            "alpha_rel", "beta_rel", "theta_rel",
            "spectral_entropy_mean",
            "log_theta_alpha", "log_theta_beta"
        ],
    }

    print("\n" + "-" * 70)
    print("MULTICOLLINEARITY DIAGNOSTICS")
    print("-" * 70)

    results = []
    for name, cols in feature_sets.items():
        X = df[cols].dropna().values
        Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)
        rank = matrix_rank(Xz, tol=1e-8)
        condition = cond(Xz)
        n_features = len(cols)

        results.append({
            "set": name,
            "n_features": n_features,
            "rank": rank,
            "full_rank": rank == n_features,
            "condition_number": condition
        })

        print(f"\n[{name}]  n={n_features}")
        print(f"  Rank            : {rank}  {'✓ full rank' if rank == n_features else '✗ rank deficient'}")
        print(f"  Condition number: {condition:.2e}")

    res_df = pd.DataFrame(results)
    print("\nSummary:")
    print(res_df.to_string(index=False))

    preferred = "drop_gamma"   
    cols = feature_sets[preferred]

    corr = df[cols].corr(method="spearman")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title(f"Spearman correlation — feature set: {preferred}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"corr_{preferred}.png", dpi=150)
    plt.close()
    print(f"\nSaved correlation heatmap → {FIG_DIR / f'corr_{preferred}.png'}")

    keep_cols = ["participant_id", "age", "sex", "group", "qc_flag", "n_runs"] + feature_sets["full_rel"]
    df[keep_cols].to_csv(OUT_FEATURES, index=False)
    print(f"Saved all clean features → {OUT_FEATURES.name}")

    td = df[df["group"] == "TD"][keep_cols].copy()
    td.to_csv(OUT_TD, index=False)
    print(f"Saved TD only (n={len(td)}) → {OUT_TD.name}")

    print("\n" + "=" * 70)
    print("Feature engineering finished.")
    print("Recommended next: choose the best feature set (lowest condition number + full rank)")
    print("=" * 70)

if __name__ == "__main__":
    main()