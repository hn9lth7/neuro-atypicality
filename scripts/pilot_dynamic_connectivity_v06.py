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

BIDS_ROOT = PROJECT_ROOT / "data" / "raw" / "ds006780"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "pilot_v06"
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

def main():
    print("=" * 70)
    print(f"NAI v0.6 Pilot — Dynamic Connectivity")
    print(f"sub-{SUBJECT} run-{RUN}  |  window={WINDOW_SEC}s  step={STEP_SEC}s")
    print("=" * 70)

    raw = load_raw_bids(BIDS_ROOT, subject=SUBJECT, run=RUN)
    raw_clean = preprocess_minimal(raw)
    data = raw_clean.get_data()
    sfreq = raw_clean.info["sfreq"]
    duration = raw_clean.times[-1]

    print(f"Duration : {duration:.1f} s")
    print(f"sfreq    : {sfreq} Hz")
    print(f"Expected windows ≈ {int((duration - WINDOW_SEC) / STEP_SEC) + 1}")

    results = {}

    for band, (fmin, fmax) in BANDS.items():
        print(f"\n--- {band.upper()} ({fmin}-{fmax} Hz) ---")

        matrices = compute_windowed_plv(
            data, sfreq, fmin, fmax,
            window_sec=WINDOW_SEC, step_sec=STEP_SEC
        )
        T = len(matrices)
        print(f"  Windows : {T}")

        deltas = transition_series(matrices)
        summary = dynamic_summary(deltas)

        print(f"  mean Δ  : {summary['mean_delta']:.4f}")
        print(f"  std Δ   : {summary['std_delta']:.4f}")
        print(f"  CV Δ    : {summary['cv_delta']:.4f}")
        print(f"  max Δ   : {summary['max_delta']:.4f}")
        print(f"  H(Δ)    : {summary['temporal_entropy']:.4f}")

        results[band] = {
            "matrices": matrices,
            "deltas": deltas,
            "summary": summary,
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(deltas, "o-", color="steelblue", lw=1.5, markersize=5)
        axes[0].axhline(summary["mean_delta"], color="crimson", ls="--",
                        label=f"mean={summary['mean_delta']:.3f}")
        axes[0].set_xlabel("Transition index")
        axes[0].set_ylabel("Δ (relative Frobenius)")
        axes[0].set_title(f"{band} — transition magnitude")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        mean_plv = [W[np.triu_indices_from(W, k=1)].mean() for W in matrices]
        axes[1].plot(mean_plv, "s-", color="darkorange", lw=1.5, markersize=5)
        axes[1].set_xlabel("Window index")
        axes[1].set_ylabel("Mean PLV")
        axes[1].set_title(f"{band} — mean connectivity over time")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out = FIG_DIR / f"sub-{SUBJECT}_run-{RUN}_{band}_dynamic.png"
        plt.savefig(out, dpi=140)
        plt.close()
        print(f"  Saved → {out.name}")

    print("\n" + "=" * 70)
    print("DYNAMIC SUMMARY")
    print("=" * 70)
    print(f"{'band':8s} {'n_win':>6s} {'meanΔ':>8s} {'stdΔ':>8s} "
          f"{'CVΔ':>7s} {'maxΔ':>8s} {'H(Δ)':>7s}")
    for band, r in results.items():
        s = r["summary"]
        print(f"{band:8s} {len(r['matrices']):6d} "
              f"{s['mean_delta']:8.4f} {s['std_delta']:8.4f} "
              f"{s['cv_delta']:7.3f} {s['max_delta']:8.4f} "
              f"{s['temporal_entropy']:7.3f}")

    print(f"\nFigures → {FIG_DIR}")
    print("Pilot finished.")

if __name__ == "__main__":
    main()