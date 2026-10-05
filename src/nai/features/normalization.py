from __future__ import annotations

import numpy as np

def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sd = np.nanstd(x, axis=axis, keepdims=True) + 1e-12
    return (x - mu) / sd