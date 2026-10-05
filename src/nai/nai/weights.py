from __future__ import annotations

EQUAL_WEIGHTS = {"SE": 0.25, "C": 0.25, "G": 0.25, "D": 0.25}

def equal_weights() -> dict[str, float]:
    return dict(EQUAL_WEIGHTS)