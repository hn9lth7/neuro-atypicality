from pathlib import Path
import pandas as pd

def load_participants(bids_root: str | Path) -> pd.DataFrame:
    bids_root = Path(bids_root)
    path = bids_root / "participants.tsv"

    if not path.exists():
        raise FileNotFoundError(f"participants.tsv not found: {path}")

    df = pd.read_csv(path, sep="\t")
    return df