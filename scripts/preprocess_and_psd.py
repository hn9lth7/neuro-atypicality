from pathlib import Path
import sys
import numpy as np
import pandas as pd
import mne

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = PROJECT_ROOT / "data" / "raw" / "ds006780"
RESULTS_DIR = PROJECT_ROOT / "results" / "features"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import load_raw_bids
from nai.io.metadata import load_participants

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

SUBJECT = "10025"
RUN = "01"

def preprocess_minimal(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw = raw.copy()

    raw.load_data()

    raw.pick(picks="eeg")

    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing="ignore")

    raw.filter(l_freq=1.0, h_freq=45.0, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=60.0, verbose=False)

    raw.set_eeg_reference("average", projection=False, verbose=False)

    return raw

def compute_band_powers(raw: mne.io.BaseRaw) -> dict:
    raw_uv = raw.copy()
    raw_uv.apply_function(lambda x: x * 1e6, channel_wise=False)

    psd = raw_uv.compute_psd(method="welch", fmin=1.0, fmax=45.0, verbose=False)
    freqs = psd.freqs
    psd_data = psd.get_data()          

    mean_psd = psd_data.mean(axis=0)   

    powers = {}
    for name, (fmin, fmax) in BANDS.items():
        idx = np.logical_and(freqs >= fmin, freqs < fmax)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        powers[f"{name}_abs"] = float(trapz(mean_psd[idx], freqs[idx]))

    total = sum(powers[f"{b}_abs"] for b in BANDS)
    for name in BANDS:
        powers[f"{name}_rel"] = powers[f"{name}_abs"] / total if total > 0 else 0.0

    powers["theta_alpha"] = powers["theta_abs"] / powers["alpha_abs"] if powers["alpha_abs"] > 0 else np.nan
    powers["theta_beta"]  = powers["theta_abs"] / powers["beta_abs"]  if powers["beta_abs"]  > 0 else np.nan
    powers["alpha_beta"]  = powers["alpha_abs"] / powers["beta_abs"]  if powers["beta_abs"]  > 0 else np.nan

    return powers

def main():
    print(f"Loading sub-{SUBJECT}, run-{RUN} ...")
    raw = load_raw_bids(BIDS_ROOT, subject=SUBJECT, run=RUN)

    print("Preprocessing (minimal) ...")
    raw_clean = preprocess_minimal(raw)

    print(f"Channels after pick: {len(raw_clean.ch_names)}")
    print(f"Duration           : {raw_clean.times[-1]:.1f} s")

    print("Computing band powers ...")
    powers = compute_band_powers(raw_clean)

    participants = load_participants(BIDS_ROOT)
    row = participants[participants["participant_id"] == f"sub-{SUBJECT}"].iloc[0]

    record = {
        "participant_id": f"sub-{SUBJECT}",
        "age": row["age"],
        "sex": row["sex"],
        "group": row["group"],
        "run": RUN,
        "duration_s": round(raw_clean.times[-1], 1),
        "n_channels": len(raw_clean.ch_names),
        **powers,
    }

    df = pd.DataFrame([record])
    out_path = RESULTS_DIR / "features_rest_v0.1_one_subject.csv"
    df.to_csv(out_path, index=False)

    print("\n=== FEATURES ===")
    print(df.T.to_string())
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()