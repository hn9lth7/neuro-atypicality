from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = PROJECT_ROOT / "data" / "raw" / "ds006780"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import load_raw_bids
from nai.io.metadata import load_participants

def main():
    subject = "10025"  
    run = "01"

    print(f"Loading sub-{subject}, run-{run} ...")
    raw = load_raw_bids(BIDS_ROOT, subject=subject, run=run)

    print("\n=== RAW INFO ===")
    print(raw)
    print(f"\nSampling rate : {raw.info['sfreq']} Hz")
    print(f"Channels     : {len(raw.ch_names)}")
    print(f"Duration     : {raw.times[-1]:.1f} s")
    print(f"Channel types: {set(raw.get_channel_types())}")

    participants = load_participants(BIDS_ROOT)
    row = participants[participants["participant_id"] == f"sub-{subject}"]
    if not row.empty:
        print("\n=== PARTICIPANT ===")
        print(row[["participant_id", "age", "sex", "group"]].to_string(index=False))

    print("\nComputing PSD (this may take a moment)...")
    raw.compute_psd(fmax=50).plot(show=False)   

    print("\nDone.")

if __name__ == "__main__":
    main()