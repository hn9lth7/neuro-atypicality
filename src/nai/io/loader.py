from __future__ import annotations

from pathlib import Path

from nai.io.bids import load_raw_bids

def load_resting_raw(
    bids_root: str | Path,
    subject: str,
    run: str | int = "01",
) -> object:
    sub = subject.replace("sub-", "")
    return load_raw_bids(bids_root=bids_root, subject=sub, run=run)