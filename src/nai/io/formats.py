from __future__ import annotations

from pathlib import Path

def stem_of(path: str | Path) -> str:
    return Path(path).stem