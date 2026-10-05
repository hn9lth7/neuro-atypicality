from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import load_raw_bids, find_resting_state_files
from nai.preprocessing.pipeline import preprocess_minimal
from nai.connectivity.phase import bandpass_filter, compute_plv
from nai.graph.metrics import (
    mean_degree, degree_cv, weighted_clustering,
    global_efficiency, mean_path_length
)
from nai.graph.spectral import algebraic_connectivity, laplacian_entropy

BIDS_ROOT = PROJECT_ROOT / "data" / "raw" / "ds006780"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "graph_sensitivity_v051"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT = PROJECT_ROOT / "results" / "features" / "graph_sensitivity_audit_v051.csv"

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

DENSITIES = {
    "dense": None,         
    "top10": 0.10,
    "top20": 0.20,
    "top30": 0.30,
}

N_RUNS_TO_AUDIT = 12

def threshold_matrix(W: np.ndarray, density: float | None) -> np.ndarray:
    if density is None:
        return W.copy()

    W = W.copy()
    np.fill_diagonal(W, 0.0)
    triu = W[np.triu_indices_from(W, k=1)]
    n_keep = max(1, int(len(triu) * density))
    thresh = np.partition(triu, -n_keep)[-n_keep]
    W_thr = np.where(W >= thresh, W, 0.0)
    np.fill_diagonal(W_thr, 0.0)
    return W_thr

def compute_metrics(W: np.ndarray) -> dict:
    return {
        "mean_degree": mean_degree(W),
        "degree_cv": degree_cv(W),
        "clustering": weighted_clustering(W),
        "global_efficiency": global_efficiency(W),
        "mean_path_length": mean_path_length(W),
        "lambda2": algebraic_connectivity(W),
        "laplacian_entropy": laplacian_entropy(W),
        "n_edges": int((W > 0).sum() // 2),
        "density": float((W > 0).sum() / (W.shape[0] * (W.shape[0] - 1))),
    }

def main():
    print("=" * 72)
    print("NAI v0.5.1 — Graph Sensitivity Audit")
    print("=" * 72)

    files = find_resting_state_files(BIDS_ROOT)[:N_RUNS_TO_AUDIT]
    print(f"Auditing {len(files)} runs")

    rows = []

    for path in files:
        subject = run = None
        for part in path.name.split("_"):
            if part.startswith("sub-"):
                subject = part.replace("sub-", "")
            elif part.startswith("run-"):
                run = part.replace("run-", "")

        print(f"\n{path.name}")

        try:
            raw = load_raw_bids(BIDS_ROOT, subject=subject, run=run)
            raw_clean = preprocess_minimal(raw)
            data = raw_clean.get_data()
            sfreq = raw_clean.info["sfreq"]

            for band, (fmin, fmax) in BANDS.items():
                data_bp = bandpass_filter(data, sfreq, fmin, fmax)
                W_dense = compute_plv(data_bp)

                for dens_name, dens_val in DENSITIES.items():
                    W = threshold_matrix(W_dense, dens_val)
                    metrics = compute_metrics(W)

                    rec = {
                        "participant_id": f"sub-{subject}",
                        "run": run,
                        "band": band,
                        "density_regime": dens_name,
                        **metrics,
                    }
                    rows.append(rec)

                    print(f"  {band:5s} {dens_name:6s}  "
                          f"edges={metrics['n_edges']:4d}  "
                          f"clust={metrics['clustering']:.3f}  "
                          f"eff={metrics['global_efficiency']:.3f}  "
                          f"λ₂={metrics['lambda2']:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}")

    metrics_to_plot = [
        "clustering", "global_efficiency", "mean_path_length",
        "lambda2", "laplacian_entropy", "mean_degree"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics_to_plot):
        sns.boxplot(
            data=df, x="density_regime", y=metric, hue="band",
            ax=ax, palette="Set2"
        )
        ax.set_title(metric)
        ax.legend_.remove() if ax != axes[0] else None

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="band")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sensitivity_boxplots.png", dpi=140)
    plt.close()

    print("\nSpearman correlation of metrics: dense ↔ top20")
    for metric in metrics_to_plot:
        dense = df[df["density_regime"] == "dense"][metric].values
        top20 = df[df["density_regime"] == "top20"][metric].values
        if len(dense) == len(top20):
            rho = pd.Series(dense).corr(pd.Series(top20), method="spearman")
            print(f"  {metric:22s}  ρ = {rho:.3f}")

    print(f"\nFigures → {FIG_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()