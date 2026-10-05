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
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "validation_v052"
FIG_DIR.mkdir(parents=True, exist_ok=True)
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

BLOCKS = {
    "SE": FEATURES_SE,
    "C": FEATURES_C,
    "G": FEATURES_G,
}

LAMBDAS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
EPS = 1e-12

def fit_age_models(df: pd.DataFrame, features: list[str]) -> dict:
    models = {}
    age = df[["age"]].values
    for feat in features:
        models[feat] = LinearRegression().fit(age, df[feat].values)
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

def main():
    print("=" * 72)
    print("NAI v0.5.2 — LOO Validation of Hierarchical Blocks")
    print("=" * 72)

    spectral = pd.read_csv(FEATURES_DIR / "features_v032.csv")
    conn = pd.read_csv(FEATURES_DIR / "connectivity_graph_subject_v0.5.csv")

    df = spectral.merge(
        conn.drop(columns=["age", "sex", "group"], errors="ignore"),
        on="participant_id", how="inner"
    )

    if "qc_flag" in df.columns:
        df = df[df["qc_flag"].isin(["ok", "very_low_alpha"])].copy()
    else:
        df = df.dropna(subset=["age", "group"]).copy()

    td = df[df["group"] == "TD"].copy().reset_index(drop=True)
    asd = df[df["group"] == "ASD"].copy().reset_index(drop=True)

    print(f"TD subjects : {len(td)}")
    print(f"ASD subjects: {len(asd)}")

    print("\n" + "-" * 72)
    print("1. LEAVE-ONE-OUT (TD)")
    print("-" * 72)

    loo = {lam: {b: [] for b in BLOCKS} for lam in LAMBDAS}
    loo_nai = {lam: [] for lam in LAMBDAS}

    n_td = len(td)

    for i in range(n_td):
        mask = np.ones(n_td, dtype=bool)
        mask[i] = False
        td_train = td.iloc[mask]

        models = {}
        residuals = {}
        for name, feats in BLOCKS.items():
            models[name] = fit_age_models(td_train, feats)
            X = td_train[feats].values
            age = td_train["age"].values
            residuals[name] = X - predict(models[name], feats, age)

        row = td.iloc[i]
        age_i = row["age"]

        for lam in LAMBDAS:
            Ds = {}
            for name, feats in BLOCKS.items():
                Sigma = np.cov(residuals[name], rowvar=False)
                Sigma_reg = regularize(Sigma, lam)
                try:
                    S_inv = inv(Sigma_reg)
                except np.linalg.LinAlgError:
                    S_inv = pinv(Sigma_reg)

                x = row[feats].values.astype(float)
                r = x - predict(models[name], feats, np.array([age_i])).ravel()
                d = mahalanobis(r, S_inv)
                loo[lam][name].append(d)
                Ds[name] = d

            nai = 0.5 * (Ds["SE"] + 0.5 * (Ds["C"] + Ds["G"]))
            loo_nai[lam].append(nai)

        if (i + 1) % 10 == 0 or i == n_td - 1:
            print(f"  processed {i+1}/{n_td}")

    for lam in LAMBDAS:
        for name in BLOCKS:
            loo[lam][name] = np.array(loo[lam][name])
        loo_nai[lam] = np.array(loo_nai[lam])

    print("\nLOO summary (TD):")
    print(f"{'λ':>6s}  {'D_SE':>8s}  {'D_C':>8s}  {'D_G':>8s}  {'NAI':>8s}  {'NAI_max':>8s}")
    for lam in LAMBDAS:
        print(f"{lam:6.2f}  "
              f"{loo[lam]['SE'].mean():8.3f}  "
              f"{loo[lam]['C'].mean():8.3f}  "
              f"{loo[lam]['G'].mean():8.3f}  "
              f"{loo_nai[lam].mean():8.3f}  "
              f"{loo_nai[lam].max():8.3f}")

    print("\n" + "-" * 72)
    print("2. ASD scoring (full TD) + D_G λ-sensitivity")
    print("-" * 72)

    models_full = {}
    residuals_full = {}
    for name, feats in BLOCKS.items():
        models_full[name] = fit_age_models(td, feats)
        X = td[feats].values
        age = td["age"].values
        residuals_full[name] = X - predict(models_full[name], feats, age)

    print(f"{'λ':>6s}  {'mean_DG_TD':>11s}  {'max_DG_TD':>10s}  "
          f"{'ASD_11025':>10s}  {'ASD_11038':>10s}")

    dg_sens = []
    asd_scores = {lam: {} for lam in LAMBDAS}

    for lam in LAMBDAS:
        invs = {}
        for name in BLOCKS:
            S = regularize(np.cov(residuals_full[name], rowvar=False), lam)
            try:
                invs[name] = inv(S)
            except np.linalg.LinAlgError:
                invs[name] = pinv(S)

        dg_td = []
        for _, row in td.iterrows():
            x = row[FEATURES_G].values.astype(float)
            r = x - predict(models_full["G"], FEATURES_G, np.array([row["age"]])).ravel()
            dg_td.append(mahalanobis(r, invs["G"]))
        dg_td = np.array(dg_td)

        asd_dm = {}
        for _, row in asd.iterrows():
            Ds = {}
            for name, feats in BLOCKS.items():
                x = row[feats].values.astype(float)
                r = x - predict(models_full[name], feats, np.array([row["age"]])).ravel()
                Ds[name] = mahalanobis(r, invs[name])
            asd_dm[row["participant_id"]] = Ds

        print(f"{lam:6.2f}  {dg_td.mean():11.3f}  {dg_td.max():10.3f}  "
              f"{asd_dm['sub-11025']['G']:10.3f}  {asd_dm['sub-11038']['G']:10.3f}")

        dg_sens.append({
            "lambda": lam,
            "mean_DG_TD": float(dg_td.mean()),
            "max_DG_TD": float(dg_td.max()),
            "ASD_11025_DG": float(asd_dm["sub-11025"]["G"]),
            "ASD_11038_DG": float(asd_dm["sub-11038"]["G"]),
        })
        asd_scores[lam] = asd_dm

    print("\n" + "-" * 72)
    print("3. EMPIRICAL PERCENTILES (λ = 0.10, LOO TD reference)")
    print("-" * 72)

    lam_ref = 0.10
    for name in ["SE", "C", "G"]:
        dist = loo[lam_ref][name]
        print(f"\n{name}:")
        print(f"  LOO TD  95% = {np.percentile(dist, 95):.3f}")
        print(f"  LOO TD  99% = {np.percentile(dist, 99):.3f}")
        for pid, Ds in asd_scores[lam_ref].items():
            d = Ds[name]
            pct = stats.percentileofscore(dist, d, kind="rank")
            print(f"  {pid}  D={d:.3f}  →  {pct:.1f}-th percentile")

    nai_loo = loo_nai[lam_ref]
    print(f"\nNAI_v05:")
    print(f"  LOO TD  95% = {np.percentile(nai_loo, 95):.3f}")
    print(f"  LOO TD  99% = {np.percentile(nai_loo, 99):.3f}")
    for pid, Ds in asd_scores[lam_ref].items():
        nai = 0.5 * (Ds["SE"] + 0.5 * (Ds["C"] + Ds["G"]))
        pct = stats.percentileofscore(nai_loo, nai, kind="rank")
        print(f"  {pid}  NAI={nai:.3f}  →  {pct:.1f}-th percentile")

    loo_df = td[["participant_id", "age"]].copy()
    for lam in LAMBDAS:
        for name in BLOCKS:
            loo_df[f"D_{name}_LOO_λ{lam}"] = loo[lam][name]
        loo_df[f"NAI_LOO_λ{lam}"] = loo_nai[lam]
    loo_df.to_csv(NORM_DIR / "loo_blocks_v052.csv", index=False)

    summary = {
        "n_td": len(td),
        "n_asd": len(asd),
        "lambdas": LAMBDAS,
        "loo_means": {
            str(lam): {
                "SE": float(loo[lam]["SE"].mean()),
                "C": float(loo[lam]["C"].mean()),
                "G": float(loo[lam]["G"].mean()),
                "NAI": float(loo_nai[lam].mean()),
            } for lam in LAMBDAS
        },
        "dg_sensitivity": dg_sens,
    }
    with open(NORM_DIR / "validation_summary_v052.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, ["SE", "C", "G"]):
        ax.hist(loo[lam_ref][name], bins=12, color="steelblue",
                alpha=0.75, edgecolor="k", label="TD LOO")
        for pid, Ds in asd_scores[lam_ref].items():
            ax.axvline(Ds[name], lw=2,
                       label=f"{pid} ({Ds[name]:.2f})")
        ax.set_title(f"D_{name} (λ=0.10)")
        ax.set_xlabel("Mahalanobis distance")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "loo_blocks_distribution.png", dpi=140)
    plt.close()

    print(f"\nSaved LOO table → {NORM_DIR / 'loo_blocks_v052.csv'}")
    print(f"Saved summary  → {NORM_DIR / 'validation_summary_v052.json'}")
    print(f"Figures        → {FIG_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()