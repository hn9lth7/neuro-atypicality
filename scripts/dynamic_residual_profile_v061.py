from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_DIR = PROJECT_ROOT / "results" / "features"
NORM_DIR = PROJECT_ROOT / "results" / "normative"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "profiles_v061"

FIG_DIR.mkdir(parents=True, exist_ok=True)
NORM_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_D = [
    f"{m}_{b}"
    for m in [
        "mean_delta",
        "cv_delta",
        "mean_degree_cv",
        "temporal_cv_degree_cv",
    ]
    for b in ["theta", "alpha", "beta", "gamma"]
]

LAMBDA = 0.10
TARGET = "sub-11025"
EPS = 1e-12

def fit_age_models(td: pd.DataFrame):
    models = {}
    age = td[["age"]].values
    for feature in FEATURES_D:
        model = LinearRegression()
        model.fit(age, td[feature].values)
        models[feature] = model
    return models

def regularize(Sigma: np.ndarray, lam: float) -> np.ndarray:
    p = Sigma.shape[0]
    target = (np.trace(Sigma) / p) * np.eye(p)
    return (1.0 - lam) * Sigma + lam * target

def main():
    print("=" * 72)
    print("NAI v0.6.2 — Dynamic Residual Profile")
    print("=" * 72)

    spectral = pd.read_csv(FEATURES_DIR / "features_v032.csv")
    dyn = pd.read_csv(FEATURES_DIR / "dynamic_subject_v0.6.csv")

    df = spectral.merge(
        dyn.drop(columns=["age", "sex", "group", "n_runs"], errors="ignore"),
        on="participant_id",
        how="inner",
    )

    if "qc_flag" in df.columns:
        df = df[df["qc_flag"].isin(["ok", "very_low_alpha"])].copy()

    df = df[df["participant_id"] != "sub-10777"].copy()
    df = df.dropna(subset=["age", "group"]).copy()

    td = df[df["group"] == "TD"].copy().reset_index(drop=True)
    target = df[df["participant_id"] == TARGET].copy()

    if len(target) != 1:
        raise ValueError(f"Expected exactly one {TARGET}, found {len(target)}")

    print(f"Canonical TD : {len(td)}")
    print(f"Target       : {TARGET}")

    models = fit_age_models(td)

    age_target = float(target.iloc[0]["age"])
    x = target[FEATURES_D].values.astype(float).ravel()

    predicted = np.array([
        models[f].predict(np.array([[age_target]]))[0]
        for f in FEATURES_D
    ])
    residual = x - predicted

    X_td = td[FEATURES_D].values.astype(float)
    age_td = td["age"].values

    predicted_td = np.column_stack([
        models[f].predict(age_td.reshape(-1, 1))
        for f in FEATURES_D
    ])
    R_td = X_td - predicted_td

    Sigma = np.cov(R_td, rowvar=False)
    Sigma_reg = regularize(Sigma, LAMBDA)
    sigma = np.sqrt(np.diag(Sigma_reg)) + EPS
    z = residual / sigma

    Sigma_inv = np.linalg.pinv(Sigma_reg)
    d2 = float(residual @ Sigma_inv @ residual)
    D_D = float(np.sqrt(max(d2, 0.0)))

    print("\n" + "-" * 72)
    print("TARGET")
    print("-" * 72)
    print(f"Age        : {age_target:.2f}")
    print(f"D_D λ={LAMBDA:.2f} : {D_D:.3f}")

    profile = pd.DataFrame({
        "feature": FEATURES_D,
        "observed": x,
        "age_expected": predicted,
        "residual": residual,
        "residual_sd_reg": sigma,
        "z": z,
        "abs_z": np.abs(z),
    })
    profile = profile.sort_values("abs_z", ascending=False).reset_index(drop=True)

    print("\n" + "-" * 72)
    print("DYNAMIC RESIDUAL PROFILE")
    print("-" * 72)
    print(
        profile[["feature", "observed", "age_expected", "residual", "z"]]
        .round(4)
        .to_string(index=False)
    )

    profile["subblock"] = np.where(
        profile["feature"].str.startswith(("mean_delta", "cv_delta")),
        "D_delta",
        "D_H",
    )

    print("\n" + "-" * 72)
    print("SUB-BLOCK SUMMARY")
    print("-" * 72)

    for block in ["D_delta", "D_H"]:
        sub = profile[profile["subblock"] == block]
        print(f"\n{block}:")
        print(f"  mean |z| = {sub['abs_z'].mean():.3f}")
        print(f"  max  |z| = {sub['abs_z'].max():.3f}")
        print(f"  RMS z   = {np.sqrt(np.mean(sub['z'] ** 2)):.3f}")

    out_csv = NORM_DIR / "dynamic_profile_sub11025_v061.csv"
    profile.to_csv(out_csv, index=False)
    print(f"\nSaved profile → {out_csv}")

    plot_df = profile.set_index("feature").reindex(FEATURES_D)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["crimson" if abs(v) > 2 else "steelblue" for v in plot_df["z"].values]
    ax.bar(np.arange(len(FEATURES_D)), plot_df["z"].values, color=colors)
    ax.axhline(0, color="k", linewidth=1)
    ax.axhline(2, color="gray", linestyle="--", linewidth=1)
    ax.axhline(-2, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(FEATURES_D)))
    ax.set_xticklabels(FEATURES_D, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Age-corrected residual z")
    ax.set_title(
        f"{TARGET} — dynamic residual profile "
        f"(D_D = {D_D:.2f}, λ = {LAMBDA:.2f})"
    )
    plt.tight_layout()

    out_fig = FIG_DIR / "sub-11025_dynamic_profile.png"
    plt.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved figure  → {out_fig}")
    print("=" * 72)

if __name__ == "__main__":
    main()