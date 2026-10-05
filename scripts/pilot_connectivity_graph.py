from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import load_raw_bids
from nai.preprocessing.pipeline import preprocess_minimal
from nai.connectivity.phase import bandpass_filter, compute_plv
from nai.graph.metrics import (
    mean_degree, degree_cv, weighted_clustering,
    global_efficiency, mean_path_length
)
from nai.graph.spectral import algebraic_connectivity, laplacian_entropy

BIDS_ROOT = PROJECT_ROOT / "data" / "raw" / "ds006780"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "pilot_v05"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT = "10025"
RUN = "01"

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

def main():
    print("=" * 70)
    print(f"NAI v0.5 Pilot — sub-{SUBJECT} run-{RUN}")
    print("=" * 70)

    raw = load_raw_bids(BIDS_ROOT, subject=SUBJECT, run=RUN)
    raw_clean = preprocess_minimal(raw)
    data = raw_clean.get_data()         
    sfreq = raw_clean.info["sfreq"]
    ch_names = raw_clean.ch_names

    print(f"Channels : {len(ch_names)}")
    print(f"Duration : {raw_clean.times[-1]:.1f} s")
    print(f"sfreq    : {sfreq} Hz")

    results = {}

    for band, (fmin, fmax) in BANDS.items():
        print(f"\n--- {band.upper()} ({fmin}-{fmax} Hz) ---")

        data_bp = bandpass_filter(data, sfreq, fmin, fmax)

        W = compute_plv(data_bp)
        print(f"  PLV matrix  mean={W.mean():.3f}  median={np.median(W):.3f}")

        md  = mean_degree(W)
        dcv = degree_cv(W)
        cw  = weighted_clustering(W)
        eg  = global_efficiency(W)
        lp  = mean_path_length(W)
        lam2 = algebraic_connectivity(W)
        hl  = laplacian_entropy(W)

        print(f"  mean_degree        : {md:.4f}")
        print(f"  degree_cv          : {dcv:.4f}")
        print(f"  clustering         : {cw:.4f}")
        print(f"  global_efficiency  : {eg:.4f}")
        print(f"  mean_path_length   : {lp:.4f}")
        print(f"  algebraic_conn λ₂  : {lam2:.4f}")
        print(f"  laplacian_entropy  : {hl:.4f}")

        results[band] = {
            "W": W,
            "mean_degree": md,
            "degree_cv": dcv,
            "clustering": cw,
            "global_efficiency": eg,
            "mean_path": lp,
            "lambda2": lam2,
            "laplacian_entropy": hl,
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(W, ax=axes[0], cmap="viridis", square=True,
                    cbar_kws={"shrink": 0.7}, xticklabels=False, yticklabels=False)
        axes[0].set_title(f"{band} PLV matrix")

        deg = W.sum(axis=1) - np.diag(W)
        axes[1].hist(deg, bins=15, color="steelblue", edgecolor="k", alpha=0.8)
        axes[1].axvline(deg.mean(), color="crimson", ls="--", label=f"mean={deg.mean():.2f}")
        axes[1].set_title(f"{band} degree distribution")
        axes[1].set_xlabel("Weighted degree")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(FIG_DIR / f"sub-{SUBJECT}_run-{RUN}_{band}_connectivity.png", dpi=140)
        plt.close()
        print(f"  Saved figure → sub-{SUBJECT}_run-{RUN}_{band}_connectivity.png")

    print("\n" + "=" * 70)
    print("SUMMARY — Graph metrics")
    print("=" * 70)
    print(f"{'band':8s} {'mean_d':>8s} {'cv_d':>7s} {'clust':>7s} "
          f"{'eff':>7s} {'path':>7s} {'λ₂':>7s} {'H_L':>7s}")
    for band, r in results.items():
        print(f"{band:8s} {r['mean_degree']:8.3f} {r['degree_cv']:7.3f} "
              f"{r['clustering']:7.3f} {r['global_efficiency']:7.3f} "
              f"{r['mean_path']:7.3f} {r['lambda2']:7.3f} {r['laplacian_entropy']:7.3f}")

    print(f"\nFigures saved in: {FIG_DIR}")
    print("Pilot finished.")

if __name__ == "__main__":
    main()