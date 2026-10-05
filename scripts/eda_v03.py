from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "eda_v03"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV = FEATURES_DIR / "participants_features_subject_v0.2.csv"

def main():
    print("=" * 70)
    print("NAI v0.3 — Exploratory Analysis")
    print("=" * 70)

    df = pd.read_csv(CSV)
    print(f"Subjects: {len(df)}")
    print(df["group"].value_counts(dropna=False))

    feature_cols = [
        "delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel",
        "theta_alpha", "theta_beta", "alpha_beta",
        "spectral_entropy_mean", "spectral_entropy_std",
        "mean_abs_uv", "std_uv", "line_noise_60hz_ratio"
    ]

    print("\n" + "-" * 50)
    print("DESCRIPTIVE STATISTICS")
    print("-" * 50)
    print(df[feature_cols].describe().round(4).T)

    print("\n" + "-" * 50)
    print("AGE ↔ FEATURE CORRELATIONS")
    print("-" * 50)

    df_age = df.dropna(subset=["age"]).copy()
    print(f"Subjects used for age correlations: {len(df_age)} (dropped {df['age'].isna().sum()} with missing age)")

    age_corr = []
    for col in feature_cols:
        if df_age[col].notna().sum() < 5:
            continue
        r_p, p_p = stats.pearsonr(df_age["age"], df_age[col])
        r_s, p_s = stats.spearmanr(df_age["age"], df_age[col])
        age_corr.append({
            "feature": col,
            "pearson_r": round(r_p, 4),
            "pearson_p": round(p_p, 4),
            "spearman_r": round(r_s, 4),
            "spearman_p": round(p_s, 4)
        })

    age_df = pd.DataFrame(age_corr).sort_values("pearson_r", key=abs, ascending=False)
    print(age_df.to_string(index=False))

    print("\n" + "-" * 50)
    print("EXTREME RATIOS — top 5 by theta_beta")
    print("-" * 50)
    cols = ["participant_id", "age", "group", "theta_abs", "beta_abs", "theta_beta", "theta_alpha", "alpha_rel", "beta_rel"]
    print(df.nlargest(5, "theta_beta")[cols].round(4).to_string(index=False))

    print("\nEXTREME RATIOS — top 5 by theta_alpha")
    print(df.nlargest(5, "theta_alpha")[cols].round(4).to_string(index=False))

    print("\n" + "-" * 50)
    print("FEATURE CORRELATION MATRIX (saving heatmap)")
    print("-" * 50)

    corr = df[feature_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Matrix (n=43)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_matrix.png", dpi=150)
    plt.close()
    print(f"Saved → {FIG_DIR / 'correlation_matrix.png'}")

    print("\n" + "-" * 50)
    print("DISTRIBUTIONS")
    print("-" * 50)

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.ravel()

    for i, col in enumerate(feature_cols[:12]):
        ax = axes[i]
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(col)
        ax.axvline(df[col].median(), color="red", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_distributions.png", dpi=150)
    plt.close()
    print(f"Saved → {FIG_DIR / 'feature_distributions.png'}")

    key_features = ["alpha_rel", "theta_rel", "theta_alpha", "spectral_entropy_mean"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for i, col in enumerate(key_features):
        ax = axes[i]
        sns.regplot(data=df, x="age", y=col, ax=ax,
                    scatter_kws={"alpha": 0.7}, line_kws={"color": "crimson"})
        ax.set_title(f"{col} vs Age")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "age_trends.png", dpi=150)
    plt.close()
    print(f"Saved → {FIG_DIR / 'age_trends.png'}")

    print("\n" + "-" * 50)
    print("QC SUMMARY")
    print("-" * 50)
    print(df[["mean_abs_uv", "std_uv", "line_noise_60hz_ratio", "n_channels", "n_runs"]].describe().round(3))

    print("\nEDA finished.")
    print(f"Figures saved in: {FIG_DIR}")

if __name__ == "__main__":
    main()