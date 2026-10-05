from __future__ import annotations

import numpy as np

def run_level_cv(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.nanstd(v) / (np.nanmean(v) + 1e-12))