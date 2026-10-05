from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import load_raw_bids
from nai.preprocessing.pipeline import preprocess_minimal

BIDS_ROOT = PROJECT_ROOT / "data" / "raw" / "ds006780"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "audit_v031"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTLIERS = {
    "10777": "TD — extreme theta_abs",
    "11025": "ASD — very low alpha_rel",
}

def inspect_subject(subject: str, label: str):
    print("\n" + "=" * 70)
    print(f"AUDITING sub-{subject}  ({label})")
    print("=" * 70)

    eeg_dir = BIDS_ROOT / f"sub-{subject}" / "eeg"
    if not eeg_dir.exists():
        print(f"  Directory not found: {eeg_dir}")
        return

    bdf_files = sorted(eeg_dir.glob("*Restingstate*_eeg.bdf"))
    print(f"  Found {len(bdf_files)} resting-state runs")

    for bdf in bdf_files:
        run = "??"
        for part in bdf.stem.split("_"):
            if part.startswith("run-"):
                run = part.replace("run-", "")
                break

        print(f"\n  --- run-{run} ---")
        try:
            raw = load_raw_bids(BIDS_ROOT, subject=subject, run=run)
            raw.load_data()

            print(f"  Duration     : {raw.times[-1]:.1f} s")
            print(f"  Channels     : {len(raw.ch_names)}")
            print(f"  sfreq        : {raw.info['sfreq']} Hz")

            raw_clean = preprocess_minimal(raw)

            raw_uv = raw_clean.copy()
            raw_uv.apply_function(lambda x: x * 1e6, channel_wise=False)

            data = raw_uv.get_data() 

            ch_std = data.std(axis=1)
            ch_mean_abs = np.abs(data).mean(axis=1)

            print(f"  EEG channels after pick: {raw_uv.info['nchan']}")
            print(f"  Channel std  (µV) — median: {np.median(ch_std):.1f}, "
                  f"max: {ch_std.max():.1f} ({raw_uv.ch_names[int(ch_std.argmax())]})")
            print(f"  Channel |amp| (µV) — median: {np.median(ch_mean_abs):.1f}, "
                  f"max: {ch_mean_abs.max():.1f} ({raw_uv.ch_names[int(ch_mean_abs.argmax())]})")

            psd = raw_uv.compute_psd(method="welch", fmin=1.0, fmax=45.0, verbose=False)
            psds, freqs = psd.get_data(return_freqs=True)  
            mean_psd = psds.mean(axis=0)

            bands = {
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 13.0),
                "beta":  (13.0, 30.0),
                "gamma": (30.0, 45.0),
            }
            trapz = getattr(np, "trapezoid", None) or np.trapz

            print("  Mean band power (µV²):")
            band_powers = {}
            for name, (fmin, fmax) in bands.items():
                idx = (freqs >= fmin) & (freqs < fmax)
                p = float(trapz(mean_psd[idx], freqs[idx]))
                band_powers[name] = p
                print(f"    {name:6s}: {p:12.2f}")

            total = sum(band_powers.values()) + 1e-12
            print("  Relative band power:")
            for name in bands:
                rel = band_powers[name] / total
                print(f"    {name:6s}: {rel:8.4f}")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].semilogy(freqs, mean_psd, color="steelblue", lw=1.8)
            axes[0].set_title(f"sub-{subject} run-{run} — mean PSD")
            axes[0].set_xlabel("Frequency (Hz)")
            axes[0].set_ylabel("Power (µV²/Hz)")
            axes[0].axvspan(4, 8, alpha=0.15, color="orange", label="theta")
            axes[0].axvspan(8, 13, alpha=0.15, color="green", label="alpha")
            axes[0].legend()
            axes[0].set_xlim(1, 45)
            axes[0].grid(True, alpha=0.3)

            for i in range(psds.shape[0]):
                axes[1].semilogy(freqs, psds[i], color="gray", alpha=0.25, lw=0.6)
            axes[1].semilogy(freqs, mean_psd, color="crimson", lw=2.0, label="mean")
            axes[1].set_title(f"sub-{subject} run-{run} — all channels")
            axes[1].set_xlabel("Frequency (Hz)")
            axes[1].set_ylabel("Power (µV²/Hz)")
            axes[1].legend()
            axes[1].set_xlim(1, 45)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            out_psd = FIG_DIR / f"sub-{subject}_run-{run}_psd.png"
            plt.savefig(out_psd, dpi=140)
            plt.close()
            print(f"  Saved PSD → {out_psd.name}")

            try:
                theta_idx = (freqs >= 4) & (freqs < 8)
                theta_power = trapz(psds[:, theta_idx], freqs[theta_idx], axis=1)
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                mne.viz.plot_topomap(theta_power, raw_uv.info, axes=ax, show=False,
                                     cmap="Reds")
                ax.set_title(f"sub-{subject} run-{run}\ntheta power topomap")
                plt.tight_layout()
                out_topo = FIG_DIR / f"sub-{subject}_run-{run}_theta_topo.png"
                plt.savefig(out_topo, dpi=140)
                plt.close()
                print(f"  Saved topo → {out_topo.name}")
            except Exception as e:
                print(f"  Topomap skipped: {e}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

def main():
    print("=" * 70)
    print("NAI v0.3.1 — Outlier Audit (corrected µV scaling)")
    print("=" * 70)

    for sub, label in OUTLIERS.items():
        inspect_subject(sub, label)

    print("\n" + "=" * 70)
    print("Audit finished.")
    print(f"Figures saved in: {FIG_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()