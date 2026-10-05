from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
NORM_DIR = PROJECT_ROOT / "results" / "normative"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "stability_v063"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUN_LEVEL = FEATURES_DIR / "dynamic_rest_v0.6.csv"
SUBJECT = FEATURES_DIR / "dynamic_subject_v0.6.csv"
NORM = NORM_DIR / "normative_blocks_v06.csv"

CORE = [
    "mean_delta",
    "cv_delta",
    "mean_degree_cv",
    "temporal_cv_degree_cv",
]
BANDS = ["theta", "alpha", "beta", "gamma"]
KEY_FEATURES = [
    "cv_delta_gamma",
    "mean_delta_theta",
    "cv_delta_beta",
    "cv_delta_alpha",
    "mean_delta_alpha",
]

def main():
    print("=" * 72)
    print("NAI v0.6.3 — Dynamic Stability Audit")
    print("=" * 72)

    run = pd.read_csv(RUN_LEVEL)
    subj = pd.read_csv(SUBJECT)
    norm = pd.read_csv(NORM) if NORM.exists() else None

    print("\n" + "-" * 72)
    print("1. sub-11025 — RUN-LEVEL DYNAMIC FEATURES")
    print("-" * 72)

    s11025 = run[run["participant_id"] == "sub-11025"].copy()
    print(f"Runs found: {s11025['run'].nunique()}")
    print(f"Rows: {len(s11025)}")

    print("\nPer-run, per-band:")
    for run_id, g in s11025.groupby("run"):
        print(f"\n  run-{run_id}  (n_windows={g['n_windows'].iloc[0]}, "
              f"duration={g['duration_s'].iloc[0]:.1f}s)")
        for _, row in g.iterrows():
            print(f"    {row['band']:6s}  "
                  f"meanΔ={row['mean_delta']:.4f}  "
                  f"cvΔ={row['cv_delta']:.4f}  "
                  f"mean_deg_cv={row['mean_degree_cv']:.4f}  "
                  f"tcv_deg_cv={row['temporal_cv_degree_cv']:.4f}")

    s_mean = subj[subj["participant_id"] == "sub-11025"]
    if len(s_mean) == 1:
        print("\n  Subject-level means (aggregated):")
        row = s_mean.iloc[0]
        for f in KEY_FEATURES:
            if f in row:
                print(f"    {f:30s}  {row[f]:.4f}")

    print("\n" + "-" * 72)
    print("2. KEY DRIVERS — run consistency (sub-11025)")
    print("-" * 72)

    for metric, band in [
        ("cv_delta", "gamma"),
        ("mean_delta", "theta"),
        ("cv_delta", "beta"),
        ("cv_delta", "alpha"),
        ("mean_delta", "alpha"),
    ]:
        vals = s11025[s11025["band"] == band][["run", metric]].sort_values("run")
        if len(vals) == 0:
            continue
        print(f"\n  {metric}_{band}:")
        for _, r in vals.iterrows():
            print(f"    run-{r['run']}: {r[metric]:.4f}")
        if len(vals) > 1:
            spread = vals[metric].max() - vals[metric].min()
            print(f"    range across runs: {spread:.4f}")

    print("\n" + "-" * 72)
    print("3. HIGH-D_D TD SUBJECTS (comparison)")
    print("-" * 72)

    if norm is not None and "D_D" in norm.columns:
        td_high = (
            norm[norm["group"] == "TD"]
            .nlargest(5, "D_D")[["participant_id", "D_D", "age"]]
        )
        print(td_high.round(3).to_string(index=False))

        high_ids = td_high["participant_id"].tolist()
    else:
        high_ids = ["sub-10212", "sub-10769", "sub-10124"]
        print("(normative file missing D_D — using fallback IDs)")

    print("\nRun-level key features for high-D_D TD:")
    for pid in high_ids:
        g = run[run["participant_id"] == pid]
        if g.empty:
            continue
        print(f"\n  {pid}  (runs={g['run'].nunique()})")
        for metric, band in [("cv_delta", "gamma"), ("mean_delta", "theta")]:
            vals = g[g["band"] == band][["run", metric]].sort_values("run")
            if vals.empty:
                continue
            vstr = ", ".join(f"r{r['run']}={r[metric]:.3f}" for _, r in vals.iterrows())
            print(f"    {metric}_{band}: {vstr}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    metrics_plot = [
        ("cv_delta", "gamma", "cv_delta_gamma"),
        ("mean_delta", "theta", "mean_delta_theta"),
        ("cv_delta", "beta", "cv_delta_beta"),
        ("mean_delta", "alpha", "mean_delta_alpha"),
    ]

    for ax, (metric, band, title) in zip(axes, metrics_plot):
        vals = s11025[s11025["band"] == band][["run", metric]].sort_values("run")
        if vals.empty:
            ax.set_visible(False)
            continue
        ax.bar(vals["run"].astype(str), vals[metric], color="steelblue", edgecolor="k")
        ax.set_title(title)
        ax.set_xlabel("run")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("sub-11025 — run-level dynamic features", fontsize=12)
    plt.tight_layout()
    out_fig = FIG_DIR / "sub-11025_run_stability.png"
    plt.savefig(out_fig, dpi=140)
    plt.close()
    print(f"\nSaved figure → {out_fig}")

    print("\n" + "-" * 72)
    print("AUDIT NOTES")
    print("-" * 72)
    print("""
  Check:
  1. Are cv_delta_gamma and mean_delta_theta elevated in BOTH runs?
  2. Is one run extreme and the other typical?
  3. Do high-D_D TD subjects show a similar pattern
     (gamma transition CV + theta mean Δ)?

  If both runs of sub-11025 show the same direction → subject-level
  elevation is more credible.
  If only one run drives the mean → treat with extra caution.
""")
    print("=" * 72)

if __name__ == "__main__":
    main()