from __future__ import annotations

from typing import Sequence

import numpy as np

def vector_from_dict(features: dict[str, float], names: Sequence[str]) -> np.ndarray:
    return np.array([float(features[n]) for n in names], dtype=float)

def dict_from_vector(x: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    x = np.asarray(x, dtype=float).ravel()
    if len(x) != len(names):
        raise ValueError("length mismatch")
    return {n: float(x[i]) for i, n in enumerate(names)}