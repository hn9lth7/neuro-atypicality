from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.blocks import SE_FEATURES, C_FEATURES, G_FEATURES, D_FEATURES
from nai.normative.regression import fit_age_models, compute_residuals
from nai.normative.covariance import empirical_covariance, regularize_covariance
from nai.normative.mahalanobis import mahalanobis_distance
from nai.nai.composite import compute_nai

FEAT = PROJECT_ROOT / "results" / "features"
NORM = PROJECT_ROOT / "results" / "normative"
OUT_DIR = PROJECT_ROOT / "results" / "validation"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "validation_b1"

LAMBDA = 0.10
N_BOOT = 1000
SEED = 42
EXCLUDE = {"sub-10777", "sub-10950"}  

def load_merged() -> pd.DataFrame:
    se = pd.read_csv(FEAT / "features_v032.csv")
    cg = pd.read_csv(FEAT / "connectivity_graph_subject_v0.5.csv")
    dy = pd.read_csv(FEAT / "dynamic_subject_v0.6.csv")

    for name, df in [("SE", se), ("CG", cg), ("D", dy)]:
        if len(df) != df["participant_id"].nunique():
            raise ValueError(f"{name}: expected 1 row per subject")

    df = se.merge(cg, on="participant_id", how="inner", suffixes=("", "_cg"))
    df = df.merge(dy, on="participant_id", how="inner", suffixes=("", "_dy"))

    if "age" not in df.columns and "age_cg" in df.columns:
        df["age"] = df["age_cg"]
    if "group" not in df.columns and "group_cg" in df.columns:
        df["group"] = df["group_cg"]

    df = df[~df["participant_id"].astype(str).isin(EXCLUDE)].copy()
    df = df.dropna(subset=["age"])
    df = df[df["group"].isin(["TD", "ASD"])].copy()
    return df.reset_index(drop=True)

def block_D(
    X_all: np.ndarray,
    age_all: np.ndarray,
    X_fit: np.ndarray,
    age_fit: np.ndarray,
    lam: float,
) -> np.ndarray:
    models = fit_age_models(X_fit, age_fit)
    R_fit = compute_residuals(X_fit, age_fit, models)
    Sigma = regularize_covariance(empirical_covariance(R_fit), lam)
    R_all = compute_residuals(X_all, age_all, models)
    return np.array([mahalanobis_distance(R_all[i], Sigma) for i in range(len(X_all))])

def score_once(
    df: pd.DataFrame,
    td_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    age = df["age"].to_numpy(dtype=float)
    out = {}
    blocks = {
        "SE": SE_FEATURES,
        "C": C_FEATURES,
        "G": G_FEATURES,
        "D": D_FEATURES,
    }
    for name, feats in blocks.items():
        X = df[feats].to_numpy(dtype=float)
        out[f"D_{name}"] = block_D(X, age, X[td_idx], age[td_idx], LAMBDA)

    nai = np.array(
        [
            float(
                compute_nai(
                    {
                        "SE": out["D_SE"][i],
                        "C": out["D_C"][i],
                        "G": out["D_G"][i],
                        "D": out["D_D"][i],
                    }
                )
            )
            for i in range(len(df))
        ]
    )
    out["NAI"] = nai
    return out

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("B1 — Bootstrap normative stability")
    print(f"λ={LAMBDA}  B={N_BOOT}  seed={SEED}")
    print("=" * 72)

    df = load_merged()
    td_mask = (df["group"] == "TD").to_numpy()
    td_positions = np.where(td_mask)[0]
    n_td = len(td_positions)
    print(f"Subjects: {len(df)}  TD={n_td}  ASD={(df.group=='ASD').sum()}")

    frozen = pd.read_csv(NORM / "nai_v10.csv")
    nai_col = "NAI_v10" if "NAI_v10" in frozen.columns else "NAI"
    frz = frozen.set_index("participant_id")[nai_col]

    full = score_once(df, td_positions)
    point = pd.DataFrame(
        {
            "participant_id": df["participant_id"],
            "group": df["group"],
            "age": df["age"],
            "NAI_point": full["NAI"],
            "D_SE": full["D_SE"],
            "D_C": full["D_C"],
            "D_G": full["D_G"],
            "D_D": full["D_D"],
        }
    )
    point["NAI_frozen"] = point["participant_id"].map(frz)
    print(
        "Point vs frozen max|Δ NAI| =",
        float((point["NAI_point"] - point["NAI_frozen"]).abs().max()),
    )

    rng = np.random.default_rng(SEED)
    samples = {pid: [] for pid in df["participant_id"]}
    d_store = {
        b: {pid: [] for pid in df["participant_id"]}
        for b in ("SE", "C", "G", "D")
    }

    for b in range(N_BOOT):
        boot_idx = rng.choice(td_positions, size=n_td, replace=True)
        sc = score_once(df, boot_idx)
        for i, pid in enumerate(df["participant_id"]):
            samples[pid].append(float(sc["NAI"][i]))
            for blk in ("SE", "C", "G", "D"):
                d_store[blk][pid].append(float(sc[f"D_{blk}"][i]))
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT}")

    rows = []
    for pid, vals in samples.items():
        for b_i, v in enumerate(vals):
            rows.append({"participant_id": pid, "boot": b_i, "NAI": v})
    samp_df = pd.DataFrame(rows)
    samp_path = OUT_DIR / "bootstrap_nai_samples_b1.csv"
    samp_df.to_csv(samp_path, index=False)

    sum_rows = []
    for i, pid in enumerate(df["participant_id"]):
        arr = np.asarray(samples[pid], dtype=float)
        sum_rows.append(
            {
                "participant_id": pid,
                "group": df.loc[i, "group"],
                "age": float(df.loc[i, "age"]),
                "NAI_point": float(point.loc[i, "NAI_point"]),
                "NAI_frozen": float(point.loc[i, "NAI_frozen"])
                if pd.notna(point.loc[i, "NAI_frozen"])
                else np.nan,
                "NAI_boot_mean": float(arr.mean()),
                "NAI_boot_std": float(arr.std(ddof=1)),
                "NAI_q025": float(np.quantile(arr, 0.025)),
                "NAI_q50": float(np.quantile(arr, 0.50)),
                "NAI_q975": float(np.quantile(arr, 0.975)),
                "NAI_CI_width": float(np.quantile(arr, 0.975) - np.quantile(arr, 0.025)),
            }
        )
    summary = pd.DataFrame(sum_rows)

    frz_rank = point.set_index("participant_id")["NAI_frozen"].rank()
    rhos = []
    for b_i in range(N_BOOT):
        boot_scores = samp_df[samp_df["boot"] == b_i].set_index("participant_id")["NAI"]
        common = frz_rank.index.intersection(boot_scores.index)
        rho = pd.Series(frz_rank.loc[common]).corr(
            boot_scores.loc[common], method="spearman"
        )
        rhos.append(float(rho))
    rhos = np.asarray(rhos, dtype=float)

    meta = {
        "lambda": LAMBDA,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "n_subjects": int(len(df)),
        "n_td": int(n_td),
        "spearman_rho_vs_frozen_mean": float(np.nanmean(rhos)),
        "spearman_rho_vs_frozen_median": float(np.nanmedian(rhos)),
        "spearman_rho_vs_frozen_q025": float(np.nanquantile(rhos, 0.025)),
        "spearman_rho_vs_frozen_q975": float(np.nanquantile(rhos, 0.975)),
        "point_vs_frozen_max_abs_delta": float(
            (point["NAI_point"] - point["NAI_frozen"]).abs().max()
        ),
    }

    sum_path = OUT_DIR / "bootstrap_summary_b1.csv"
    summary.to_csv(sum_path, index=False)
    with open(OUT_DIR / "bootstrap_summary_b1.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("-" * 72)
    print("Rank stability (Spearman NAI_boot vs NAI_frozen):")
    print(
        f"  mean={meta['spearman_rho_vs_frozen_mean']:.4f}  "
        f"median={meta['spearman_rho_vs_frozen_median']:.4f}  "
        f"95% CI [{meta['spearman_rho_vs_frozen_q025']:.4f}, "
        f"{meta['spearman_rho_vs_frozen_q975']:.4f}]"
    )
    print(f"Saved samples → {samp_path}")
    print(f"Saved summary → {sum_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rhos, bins=30, edgecolor="black", alpha=0.8)
    ax.axvline(meta["spearman_rho_vs_frozen_median"], color="C1", label="median ρ")
    ax.set_xlabel("Spearman ρ (bootstrap NAI vs frozen NAI)")
    ax.set_ylabel("Count")
    ax.set_title("B1 rank stability under TD bootstrap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bootstrap_rank_stability.png", dpi=140)
    plt.close(fig)

    s = summary.sort_values("NAI_point")
    fig, ax = plt.subplots(figsize=(8, max(6, 0.22 * len(s))))
    y = np.arange(len(s))
    ax.hlines(y, s["NAI_q025"], s["NAI_q975"], color="0.6", lw=1)
    colors = ["C3" if g == "ASD" else "C0" for g in s["group"]]
    ax.scatter(s["NAI_point"], y, c=colors, s=18, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(s["participant_id"], fontsize=7)
    ax.set_xlabel("NAI")
    ax.set_title("B1 bootstrap 95% intervals (TD resample → rescore)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nai_ci_forest.png", dpi=140)
    plt.close(fig)

    print(f"Figures → {FIG_DIR}")
    print("=" * 72)
    print("B1 bootstrap finished (frozen core unchanged).")

if __name__ == "__main__":
    main()