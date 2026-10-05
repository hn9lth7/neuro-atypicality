from __future__ import annotations

import numpy as np

def fraction_extreme_samples(
    data_uv: np.ndarray,
    threshold_uv: float = 150.0,
) -> float:
    x = np.asarray(data_uv, dtype=float)
    return float(np.mean(np.abs(x) > threshold_uv))