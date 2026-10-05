from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = PROJECT_ROOT / "data" / "raw" / "ds006780"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import find_resting_state_files
from nai.io.metadata import load_participants

def main():
    print("=" * 70)
    print("NAI v0.1 — DS006780 INSPECTION")
    print("=" * 70)

    if not BIDS_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found:\n{BIDS_ROOT}")

    print(f"\nDataset root: {BIDS_ROOT}")

    participants = load_participants(BIDS_ROOT)

    print("\n" + "-" * 70)
    print("PARTICIPANTS")
    print("-" * 70)
    print(f"Number of participants: {len(participants)}")
    print("\nColumns:")
    print(list(participants.columns))
    print("\nFirst 8 rows:")
    print(participants.head(8).to_string())

    resting_files = find_resting_state_files(BIDS_ROOT)

    print("\n" + "-" * 70)
    print("RESTING-STATE EEG FILES")
    print("-" * 70)
    print(f"Found files: {len(resting_files)}")

    for path in resting_files[:30]:         
        print(" ", path.relative_to(BIDS_ROOT))
    if len(resting_files) > 30:
        print(f"  ... and {len(resting_files) - 30} more")

    subjects = sorted({p.parent.parent.name for p in resting_files})

    print("\n" + "-" * 70)
    print("SUBJECTS WITH RESTING-STATE")
    print("-" * 70)
    print(f"Count: {len(subjects)}")
    for s in subjects:
        print(" ", s)

    print("\nInspection finished successfully.")

if __name__ == "__main__":
    main()