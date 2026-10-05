from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.linalg import matrix_rank, cond
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "diagnostics_v06"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DYN = FEATURES_DIR / "dynamic_subject_v0.6.csv"
CLEAN = FEATURES_DIR / "participants_features_subject_v0.3_clean.csv"

CORE = [
    "mean_delta",
    "cv_delta",
    "mean_degree_cv",
    "temporal_cv_degree_cv",
]
BANDS = ["theta", "alpha", "beta", "gamma"]
FEATURES = [f"{m}_{b}" for m in CORE for b in BANDS]

def main():
    print("=" * 72)
    print("NAI v0.6.3 — Dynamic Feature Diagnostics")
    print("=" * 72)

    dyn = pd.read_csv(DYN)

    if CLEAN.exists():
        clean = pd.read_csv(CLEAN)
        keep = set(clean["participant_id"])
        keep.discard("sub-10777")
        df = dyn[dyn["participant_id"].isin(keep)].copy()
    else:
        df = dyn.dropna(subset=["age", "group"]).copy()
        df = df[df["participant_id"] != "sub-10777"]

    td = df[df["group"] == "TD"].copy()
    asd = df[df["group"] == "ASD"].copy()

    print(f"Canonical cohort : {len(df)}")
    print(f"  TD  : {len(td)}")
    print(f"  ASD : {len(asd)}")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print("MISSING:", missing)
        return
    print(f"Dynamic features : {len(FEATURES)}")

    X = td[FEATURES].values.astype(float)
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    td = td.iloc[mask]
    print(f"TD after NaN drop: {len(td)}")

    print("\n" + "-" * 72)
    print("MATRIX DIAGNOSTICS (TD)")
    print("-" * 72)

    Xz = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
    Sigma = np.cov(Xz, rowvar=False)

    rank = matrix_rank(Sigma, tol=1e-8)
    c_raw = cond(Sigma)

    p = Sigma.shape[0]
    target = (np.trace(Sigma) / p) * np.eye(p)
    Sigma_reg = 0.9 * Sigma + 0.1 * target
    c_reg = cond(Sigma_reg)

    print(f"  n_features     : {p}")
    print(f"  n_samples (TD) : {X.shape[0]}")
    print(f"  Rank           : {rank}  {'✓ full' if rank == p else '✗ deficient'}")
    print(f"  Cond raw       : {c_raw:.2e}")
    print(f"  Cond λ=0.1     : {c_reg:.2e}")

    print("\n" + "-" * 72)
    print("HIGHLY CORRELATED PAIRS (|ρ| > 0.85)")
    print("-" * 72)

    corr = pd.DataFrame(Xz, columns=FEATURES).corr(method="spearman")
    pairs = []
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            rho = corr.iloc[i, j]
            if abs(rho) > 0.85:
                pairs.append((FEATURES[i], FEATURES[j], rho))

    if not pairs:
        print("  None")
    else:
        for a, b, rho in sorted(pairs, key=lambda x: -abs(x[2])):
            print(f"  {a:30s}  ↔  {b:30s}  ρ = {rho:+.3f}")

    print("\n" + "-" * 72)
    print("AGE CORRELATIONS (Spearman, TD)")
    print("-" * 72)

    age = td["age"].values
    print(f"{'feature':32s}  {'ρ':>7s}  {'p':>8s}")
    for f in FEATURES:
        r, pval = stats.spearmanr(age, td[f].values)
        flag = " *" if pval < 0.05 else ""
        print(f"  {f:30s}  {r:+7.3f}  {pval:8.4f}{flag}")

    print("\n" + "-" * 72)
    print("FEATURE VARIANCE (TD, standardized scale ≈ 1)")
    print("-" * 72)
    var = Xz.var(axis=0, ddof=1)
    for f, v in zip(FEATURES, var):
        print(f"  {f:30s}  var = {v:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, square=True,
                xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.7})
    ax.set_title("Dynamic features — Spearman correlation (TD)")
    plt.xticks(fontsize=7, rotation=90)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dynamic_corr_matrix.png", dpi=140)
    plt.close()
    print(f"\nSaved heatmap → {FIG_DIR / 'dynamic_corr_matrix.png'}")

    # Sub-block suggestion
    print("\n" + "-" * 72)
    print("SUGGESTED SUB-BLOCKS")
    print("-" * 72)
    print("""
  D_Δ  (PLV transition dynamics)     — 8 features
       mean_delta_{theta,alpha,beta,gamma}
       cv_delta_{theta,alpha,beta,gamma}

  D_H  (Degree heterogeneity dyn.)   — 8 features
       mean_degree_cv_{...}
       temporal_cv_degree_cv_{...}

  If full 16D condition remains high after shrinkage,
  prefer separate D_Δ and D_H rather than one 16D covariance.
""")
    print("=" * 72)

if __name__ == "__main__":
    main()