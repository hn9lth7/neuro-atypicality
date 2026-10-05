from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import load_raw_bids
from nai.preprocessing.pipeline import preprocess_minimal
from nai.connectivity.dynamic import compute_windowed_plv
from nai.dynamics.transitions import transition_series, dynamic_summary
from nai.graph.metrics import (
    mean_degree, degree_cv, weighted_clustering,
    global_efficiency, mean_path_length
)
from nai.graph.spectral import laplacian_entropy

BIDS_ROOT = PROJECT_ROOT / "data" / "raw" / "ds006780"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "pilot_v061"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT = "10025"
RUN = "01"
WINDOW_SEC = 10.0
STEP_SEC = 5.0

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

GRAPH_FUNCS = {
    "mean_degree": mean_degree,
    "degree_cv": degree_cv,
    "clustering": weighted_clustering,
    "global_efficiency": global_efficiency,
    "mean_path_length": mean_path_length,
    "laplacian_entropy": laplacian_entropy,
}

def main():
    print("=" * 72)
    print("NAI v0.6.1 Pilot — Dynamic PLV + Graph trajectories")
    print(f"sub-{SUBJECT} run-{RUN}  |  window={WINDOW_SEC}s  step={STEP_SEC}s")
    print("=" * 72)

    raw = load_raw_bids(BIDS_ROOT, subject=SUBJECT, run=RUN)
    raw_clean = preprocess_minimal(raw)
    data = raw_clean.get_data()
    sfreq = raw_clean.info["sfreq"]
    duration = raw_clean.times[-1]
    print(f"Duration : {duration:.1f} s")

    all_summaries = []

    for band, (fmin, fmax) in BANDS.items():
        print(f"\n--- {band.upper()} ({fmin}-{fmax} Hz) ---")

        matrices = compute_windowed_plv(
            data, sfreq, fmin, fmax,
            window_sec=WINDOW_SEC, step_sec=STEP_SEC
        )
        T = len(matrices)
        print(f"  Windows : {T}")

        deltas = transition_series(matrices)
        dsum = dynamic_summary(deltas)
        print(f"  PLV Δ   mean={dsum['mean_delta']:.4f}  "
              f"CV={dsum['cv_delta']:.3f}  H={dsum['temporal_entropy']:.3f}")

        trajectories = {name: [] for name in GRAPH_FUNCS}
        for W in matrices:
            for name, func in GRAPH_FUNCS.items():
                trajectories[name].append(func(W))

        print(f"  {'metric':22s}  {'mean':>8s}  {'std':>8s}  {'CV':>7s}")
        for name, series in trajectories.items():
            arr = np.array(series)
            mu = arr.mean()
            sd = arr.std()
            cv = sd / (abs(mu) + 1e-12)
            print(f"  {name:22s}  {mu:8.4f}  {sd:8.4f}  {cv:7.3f}")
            all_summaries.append({
                "band": band,
                "metric": name,
                "mean": mu,
                "std": sd,
                "cv": cv,
            })

        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.ravel()

        axes[0].plot(deltas, "o-", color="steelblue", lw=1.5)
        axes[0].set_title(f"{band} — Δ(t)")
        axes[0].set_xlabel("transition")
        axes[0].grid(True, alpha=0.3)

        for i, (name, series) in enumerate(trajectories.items(), start=1):
            if i >= len(axes):
                break
            axes[i].plot(series, "s-", color="darkorange", lw=1.5, markersize=4)
            axes[i].set_title(name)
            axes[i].set_xlabel("window")
            axes[i].grid(True, alpha=0.3)

        plt.suptitle(f"sub-{SUBJECT} run-{RUN} — {band}", fontsize=12)
        plt.tight_layout()
        out = FIG_DIR / f"sub-{SUBJECT}_run-{RUN}_{band}_graph_dyn.png"
        plt.savefig(out, dpi=130)
        plt.close()
        print(f"  Saved → {out.name}")

    print("\n" + "=" * 72)
    print("GRAPH METRIC VARIABILITY (CV across windows)")
    print("=" * 72)
    print(f"{'band':8s} {'metric':22s} {'mean':>8s} {'std':>8s} {'CV':>7s}")
    for r in all_summaries:
        print(f"{r['band']:8s} {r['metric']:22s} "
              f"{r['mean']:8.4f} {r['std']:8.4f} {r['cv']:7.3f}")

    print(f"\nFigures → {FIG_DIR}")
    print("Pilot finished.")

if __name__ == "__main__":
    main()