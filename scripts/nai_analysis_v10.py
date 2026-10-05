from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.blocks import ALL_BLOCKS, SE_FEATURES, C_FEATURES, G_FEATURES, D_FEATURES
from nai.nai.profile import residual_z_profile

FEATURES_DIR = PROJECT_ROOT / "results" / "features"
NORM_DIR = PROJECT_ROOT / "results" / "normative"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "analysis_v10"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NAI_CSV = NORM_DIR / "nai_v10.csv"
LOO_CSV = NORM_DIR / "loo_nai_v10.csv"

LAMBDA_REF = 0.10
LAMBDA_GRID = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

ASD_NAI_BY_LAMBDA = {
    0.01: {"sub-11025": 5.270, "sub-11038": 1.629},
    0.05: {"sub-11025": 4.263, "sub-11038": 1.371},
    0.10: {"sub-11025": 3.864, "sub-11038": 1.242},
    0.20: {"sub-11025": 3.548, "sub-11038": 1.125},
    0.30: {"sub-11025": 3.430, "sub-11038": 1.076},
    0.50: {"sub-11025": 3.419, "sub-11038": 1.058},
}
TD_MEAN_BY_LAMBDA = {
    0.01: 2.522,
    0.05: 2.288,
    0.10: 2.192,
    0.20: 2.121,
    0.30: 2.108,
    0.50: 2.167,
}

def load_data():
    nai = pd.read_csv(NAI_CSV)
    loo = pd.read_csv(LOO_CSV)
    return nai, loo

def plot_loo_distribution(loo: pd.DataFrame, nai: pd.DataFrame) -> None:
    scores = loo["NAI_LOO"].values.astype(float)
    p95 = np.percentile(scores, 95)
    p99 = np.percentile(scores, 99)

    asd = nai[nai["group"] == "ASD"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(scores, bins=12, color="steelblue", edgecolor="k", alpha=0.75, label="TD LOO NAI")
    ax.axvline(p95, color="gray", ls="--", lw=1.5, label=f"P95 = {p95:.3f}")
    ax.axvline(p99, color="black", ls=":", lw=1.5, label=f"P99 = {p99:.3f}")
    for _, row in asd.iterrows():
        ax.axvline(
            row["NAI_v10"],
            lw=2,
            label=f"{row['participant_id']} = {row['NAI_v10']:.3f}",
        )
    ax.set_xlabel("NAI (leave-one-out TD reference)")
    ax.set_ylabel("Count")
    ax.set_title("Empirical LOO TD distribution of NAI v1.0")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "nai_loo_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.sort(scores)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color="steelblue", label="TD LOO eCDF")
    ax.axhline(0.95, color="gray", ls="--", lw=1, label="0.95")
    ax.axvline(p95, color="gray", ls="--", lw=1)
    for _, row in asd.iterrows():
        ax.axvline(row["NAI_v10"], lw=2, label=f"{row['participant_id']}")
    ax.set_xlabel("NAI")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("eCDF of LOO TD NAI with ASD scores marked")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "nai_loo_ecdf.png", dpi=150)
    plt.close(fig)

    print(f"  LOO mean={scores.mean():.3f}  P95={p95:.3f}  P99={p99:.3f}")

def block_contributions(nai: pd.DataFrame) -> pd.DataFrame:
    cols = ["D_SE", "D_C", "D_G", "D_D"]
    total = nai[cols].sum(axis=1)
    out = nai[["participant_id", "group", "age", "NAI_v10"]].copy()
    for c in cols:
        out[f"C_{c[2:]}"] = nai[c] / total  
    out["sum_C"] = out[[f"C_{c[2:]}" for c in cols]].sum(axis=1)
    return out

def plot_block_contribution(row: pd.Series, path: Path, title: str) -> None:
    labels = ["SE", "C", "G", "D"]
    vals = [row[f"C_{b}"] for b in labels]
    distances = [row.get(f"D_{b}", np.nan) for b in labels]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    ax.bar(labels, vals, color=colors, edgecolor="k")
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$C_B = D_B / \sum D$")
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_lambda_sensitivity() -> pd.DataFrame:
    rows = []
    for lam in LAMBDA_GRID:
        rows.append({
            "lambda": lam,
            "TD_mean_NAI": TD_MEAN_BY_LAMBDA[lam],
            "sub-11025": ASD_NAI_BY_LAMBDA[lam]["sub-11025"],
            "sub-11038": ASD_NAI_BY_LAMBDA[lam]["sub-11038"],
        })
    sens = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sens["lambda"], sens["TD_mean_NAI"], "o-", color="steelblue", label="TD mean NAI")
    ax.plot(sens["lambda"], sens["sub-11025"], "s-", color="crimson", label="sub-11025")
    ax.plot(sens["lambda"], sens["sub-11038"], "^-", color="seagreen", label="sub-11038")
    ax.axvline(LAMBDA_REF, color="gray", ls="--", label=f"reference λ={LAMBDA_REF}")
    ax.set_xlabel(r"Shrinkage $\lambda$")
    ax.set_ylabel("NAI")
    ax.set_title("λ-sensitivity of NAI v1.0 (frozen scores)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "nai_lambda_sensitivity.png", dpi=150)
    plt.close(fig)
    return sens

def residual_profiles_for_targets(nai: pd.DataFrame) -> pd.DataFrame:
    spectral = pd.read_csv(FEATURES_DIR / "features_v032.csv")
    dyn = pd.read_csv(FEATURES_DIR / "dynamic_subject_v0.6.csv")

    td_ids = set(nai.loc[nai["group"] == "TD", "participant_id"])
    spec_td = spectral[spectral["participant_id"].isin(td_ids)].copy()
    dyn_td = dyn[dyn["participant_id"].isin(td_ids)].copy()

    frames = []
    for pid in ("sub-11025", "sub-11038"):
        row_n = nai.loc[nai["participant_id"] == pid].iloc[0]
        age = float(row_n["age"])

        row_s = spectral.loc[spectral["participant_id"] == pid].iloc[0]
        X_td = spec_td[SE_FEATURES].values.astype(float)
        age_td = spec_td["age"].values.astype(float)
        x = row_s[SE_FEATURES].values.astype(float)
        prof = residual_z_profile(X_td, age_td, x, age, SE_FEATURES, lam=LAMBDA_REF)
        prof["participant_id"] = pid
        prof["block"] = "SE"
        frames.append(prof)

        row_d = dyn.loc[dyn["participant_id"] == pid].iloc[0]
        X_td = dyn_td[D_FEATURES].values.astype(float)

        if "age" in dyn_td.columns:
            age_td_d = dyn_td["age"].values.astype(float)
        else:
            age_map = spectral.set_index("participant_id")["age"]
            age_td_d = dyn_td["participant_id"].map(age_map).values.astype(float)
        x = row_d[D_FEATURES].values.astype(float)
        prof = residual_z_profile(X_td, age_td_d, x, age, D_FEATURES, lam=LAMBDA_REF)
        prof["participant_id"] = pid
        prof["block"] = "D"
        frames.append(prof)

        for block, feats in [("SE", SE_FEATURES), ("D", D_FEATURES)]:
            p = frames[-1] if block == "D" else frames[-2]
            p = p.set_index("feature").reindex(feats)
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ["crimson" if abs(z) > 2 else "steelblue" for z in p["z"].values]
            ax.bar(np.arange(len(feats)), p["z"].values, color=colors, edgecolor="k")
            ax.axhline(0, color="k", lw=1)
            ax.axhline(2, color="gray", ls="--", lw=1)
            ax.axhline(-2, color="gray", ls="--", lw=1)
            ax.set_xticks(np.arange(len(feats)))
            ax.set_xticklabels(feats, rotation=70, ha="right", fontsize=7)
            ax.set_ylabel("Age-corrected residual z")
            ax.set_title(f"{pid} — residual profile block {block}")
            plt.tight_layout()
            fig.savefig(FIG_DIR / f"residual_profile_{pid.replace('sub-', '')}_{block}.png", dpi=150)
            plt.close(fig)

    return pd.concat(frames, ignore_index=True)

def main():
    print("=" * 72)
    print("NAI v1.0 — Analytical package")
    print("=" * 72)

    nai, loo = load_data()

    print("\n[1] LOO distribution figures")
    plot_loo_distribution(loo, nai)

    print("\n[2] Block contributions")
    contrib = block_contributions(nai)

    contrib = contrib.merge(
        nai[["participant_id", "D_SE", "D_C", "D_G", "D_D"]],
        on="participant_id",
        how="left",
    )
    contrib.to_csv(NORM_DIR / "block_contributions_v10.csv", index=False)

    for pid in ("sub-11025", "sub-11038"):
        row = contrib.loc[contrib["participant_id"] == pid].iloc[0]
        title = (
            f"{pid}  NAI={row['NAI_v10']:.3f}  "
            f"(D_SE={row['D_SE']:.2f}, D_C={row['D_C']:.2f}, "
            f"D_G={row['D_G']:.2f}, D_D={row['D_D']:.2f})"
        )
        plot_block_contribution(
            row,
            FIG_DIR / f"nai_block_contribution_{pid.replace('sub-', '')}.png",
            title,
        )
        print(
            f"  {pid}: "
            f"C_SE={row['C_SE']:.3f} C_C={row['C_C']:.3f} "
            f"C_G={row['C_G']:.3f} C_D={row['C_D']:.3f}"
        )

    top_td = contrib[contrib["group"] == "TD"].nlargest(1, "NAI_v10").iloc[0]
    plot_block_contribution(
        top_td,
        FIG_DIR / "nai_block_contribution_top_td.png",
        f"Top TD {top_td['participant_id']}  NAI={top_td['NAI_v10']:.3f}",
    )

    print("\n[3] λ-sensitivity figure")
    sens = plot_lambda_sensitivity()
    sens.to_csv(NORM_DIR / "lambda_sensitivity_v10.csv", index=False)

    print("\n[4] Residual profiles (SE, D)")
    try:
        profiles = residual_profiles_for_targets(nai)
        profiles.to_csv(NORM_DIR / "residual_profiles_v10.csv", index=False)
        print(f"  Saved residual profiles → residual_profiles_v10.csv")
    except Exception as e:
        print(f"  WARN residual profiles skipped: {e}")

    print(f"\nFigures → {FIG_DIR}")
    print(f"Tables  → {NORM_DIR}")
    print("=" * 72)
    print("Analytical package complete (v1.0 unchanged).")
    print("=" * 72)

if __name__ == "__main__":
    main()