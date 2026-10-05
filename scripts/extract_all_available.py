from pathlib import Path
import sys
import numpy as np
import pandas as pd
import mne
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = PROJECT_ROOT / "data" / "raw" / "ds006780"
RESULTS_DIR = PROJECT_ROOT / "results" / "features"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import find_resting_state_files, load_raw_bids
from nai.io.metadata import load_participants


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

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
    mean_psd = psd.get_data().mean(axis=0)

    trapz = getattr(np, "trapezoid", None) or np.trapz
    powers = {}
    for name, (fmin, fmax) in BANDS.items():
        idx = np.logical_and(freqs >= fmin, freqs < fmax)
        powers[f"{name}_abs"] = float(trapz(mean_psd[idx], freqs[idx]))

    total = sum(powers[f"{b}_abs"] for b in BANDS)
    for name in BANDS:
        powers[f"{name}_rel"] = powers[f"{name}_abs"] / total if total > 0 else 0.0

    powers["theta_alpha"] = powers["theta_abs"] / powers["alpha_abs"] if powers["alpha_abs"] > 0 else np.nan
    powers["theta_beta"]  = powers["theta_abs"] / powers["beta_abs"]  if powers["beta_abs"]  > 0 else np.nan
    powers["alpha_beta"]  = powers["alpha_abs"] / powers["beta_abs"]  if powers["beta_abs"]  > 0 else np.nan
    return powers

def parse_bids_filename(path: Path):
    name = path.stem  
    parts = name.split("_")
    subject = parts[0].replace("sub-", "")
    run = None
    for p in parts:
        if p.startswith("run-"):
            run = p.replace("run-", "")
            break
    return subject, run

def main():
    participants = load_participants(BIDS_ROOT)
    files = find_resting_state_files(BIDS_ROOT)
    print(f"Found {len(files)} resting-state files")

    records = []
    errors = []

    for path in tqdm(files, desc="Processing"):
        try:
            subject, run = parse_bids_filename(path)
            raw = load_raw_bids(BIDS_ROOT, subject=subject, run=run)
            raw_clean = preprocess_minimal(raw)
            powers = compute_band_powers(raw_clean)

            row = participants[participants["participant_id"] == f"sub-{subject}"]
            if row.empty:
                age = sex = group = np.nan
            else:
                row = row.iloc[0]
                age, sex, group = row["age"], row["sex"], row["group"]

            record = {
                "participant_id": f"sub-{subject}",
                "age": age,
                "sex": sex,
                "group": group,
                "run": run,
                "duration_s": round(raw_clean.times[-1], 1),
                "n_channels": len(raw_clean.ch_names),
                **powers,
            }
            records.append(record)

        except Exception as e:
            errors.append((str(path), str(e)))
            print(f"\nError on {path.name}: {e}")

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "participants_features_rest_v0.1.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"Successfully processed: {len(df)} runs")
    print(f"Errors: {len(errors)}")
    print(f"Saved → {out_path}")
    print("\nGroup counts:")
    print(df["group"].value_counts(dropna=False))
    print("\nFirst rows:")
    print(df.head().to_string())

if __name__ == "__main__":
    main()