from __future__ import annotations

from typing import Mapping

import numpy as np

def compute_nai(
    distances: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    keys = list(distances.keys())
    if not keys:
        raise ValueError("distances must not be empty")

    if weights is None:
        w = {k: 1.0 / len(keys) for k in keys}
    else:
        w = dict(weights)
        missing = set(keys) - set(w)
        if missing:
            raise ValueError(f"weights missing keys: {missing}")
        s = sum(w[k] for k in keys)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {s}")

    return float(sum(w[k] * float(distances[k]) for k in keys))

def compute_nai_batch(
    distance_matrix: dict[str, np.ndarray],
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    keys = list(distance_matrix.keys())
    n = len(next(iter(distance_matrix.values())))

    if weights is None:
        w = {k: 1.0 / len(keys) for k in keys}
    else:
        w = dict(weights)

    out = np.zeros(n, dtype=float)
    for k in keys:
        out += w[k] * np.asarray(distance_matrix[k], dtype=float)
    return out