from pathlib import Path
from typing import List, Optional

from mne_bids import BIDSPath, read_raw_bids

def find_resting_state_files(bids_root: str | Path) -> List[Path]:
    bids_root = Path(bids_root)
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    files = sorted(
        p for p in bids_root.rglob("*Restingstate*_eeg.bdf")
        if p.is_file()
    )
    return files

def get_subjects_with_resting(bids_root: str | Path) -> List[str]:
    files = find_resting_state_files(bids_root)
    subjects = sorted({
        p.parts[p.parts.index(next(part for part in p.parts if part.startswith("sub-")))]
        for p in files
    })

    subjects = sorted({p.parent.parent.name for p in files if p.parent.name == "eeg"})
    return subjects

def load_raw_bids(
    bids_root: str | Path,
    subject: str,
    run: str | int = "01",
    task: str = "Restingstate",
):
    bids_root = Path(bids_root)
    subject = subject.replace("sub-", "")
    run = f"{int(run):02d}" if isinstance(run, int) or run.isdigit() else run

    bids_path = BIDSPath(
        root=bids_root,
        subject=subject,
        task=task,
        run=run,
        datatype="eeg",
        suffix="eeg",
        extension=".bdf",
    )

    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    return raw