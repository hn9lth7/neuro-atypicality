from __future__ import annotations

import numpy as np

def symmetrize(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    return 0.5 * (W + W.T)

def zero_diagonal(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    return W

def clean_connectivity_matrix(W: np.ndarray) -> np.ndarray:
    W = symmetrize(W)
    W = np.clip(W, 0.0, 1.0)
    return zero_diagonal(W)

def upper_triangle_values(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    return W[np.triu_indices_from(W, k=1)]

def n_edges(W: np.ndarray) -> int:
    return int(np.count_nonzero(np.triu(W, k=1)))

def threshold_by_density(W: np.ndarray, density: float | None) -> np.ndarray:
    W = clean_connectivity_matrix(W)
    if density is None:
        return W

    density = float(density)
    if not (0.0 < density <= 1.0):
        raise ValueError("density must be in (0, 1]")

    triu = upper_triangle_values(W)
    n_keep = max(1, int(len(triu) * density))
    thresh = np.partition(triu, -n_keep)[-n_keep]

    W_thr = np.where(W >= thresh, W, 0.0)
    return zero_diagonal(W_thr)