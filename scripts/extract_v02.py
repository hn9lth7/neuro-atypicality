from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import find_resting_state_files, load_raw_bids
from nai.io.metadata import load_participants
from nai.preprocessing.pipeline import preprocess_minimal
from nai.qc.signal_quality import compute_qc
from nai.spectral.power import compute_band_powers
from nai.spectral.entropy import compute_spectral_entropy

BIDS_ROOT = PROJECT_ROOT / "data" / "raw" / "ds006780"
RESULTS_DIR = PROJECT_ROOT / "results" / "features"
OUT_CSV = RESULTS_DIR / "participants_features_rest_v0.2.csv"

def parse_bids_filename(path: Path) -> dict:
    name = path.stem 
    parts = name.split("_")
    subject = parts[0].replace("sub-", "")
    run = "01"
    for p in parts:
        if p.startswith("run-"):
            run = p.replace("run-", "")
            break
    return {"subject": subject, "run": run}

def process_one_file(path: Path, participants: pd.DataFrame) -> dict | None:
    info = parse_bids_filename(path)
    subject = info["subject"]
    run = info["run"]

    try:
        raw = load_raw_bids(BIDS_ROOT, subject=subject, run=run)

        qc = compute_qc(raw)

        raw_clean = preprocess_minimal(raw)

        powers = compute_band_powers(raw_clean)

        entropy = compute_spectral_entropy(raw_clean)

        meta = participants[participants["participant_id"] == f"sub-{subject}"]
        if len(meta) == 0:
            age, sex, group = None, None, None
        else:
            row = meta.iloc[0]
            age = row.get("age")
            sex = row.get("sex")
            group = row.get("group")

        features = {
            "participant_id": f"sub-{subject}",
            "age": age,
            "sex": sex,
            "group": group,
            "run": run,
            **qc,
            **powers,
            **entropy,
        }
        return features

    except Exception as e:
        print(f"\nError on {path.name}: {e}")
        return None

def main():
    print("=" * 70)
    print("NAI v0.2 — Run-level extraction (QC + Power + Entropy)")
    print("=" * 70)

    participants = load_participants(BIDS_ROOT)
    print(f"Participants in metadata: {len(participants)}")

    files = find_resting_state_files(BIDS_ROOT)
    print(f"Found resting-state files: {len(files)}")

    if not files:
        print("No files found. Exiting.")
        return

    records = []
    errors = 0

    for path in tqdm(files, desc="Processing"):
        feat = process_one_file(path, participants)
        if feat is not None:
            records.append(feat)
        else:
            errors += 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 70)
    print(f"Successfully processed : {len(df)}")
    print(f"Errors                 : {errors}")
    print(f"Saved → {OUT_CSV}")

    if len(df) > 0:
        print("\nGroup counts (runs):")
        print(df["group"].value_counts(dropna=False))
        print("\nColumns:")
        print(list(df.columns))
        print("\nFirst rows:")
        print(df.head(3).to_string())

if __name__ == "__main__":
    main()