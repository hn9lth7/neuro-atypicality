from __future__ import annotations

import numpy as np
from numpy.linalg import inv, pinv, LinAlgError

def _invert(Sigma: np.ndarray) -> np.ndarray:
    try:
        return inv(Sigma)
    except LinAlgError:
        return pinv(Sigma)

def mahalanobis_distance(r: np.ndarray, Sigma: np.ndarray) -> float:
    r = np.asarray(r, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)

    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be square")
    if r.shape[0] != Sigma.shape[0]:
        raise ValueError("r and Sigma dimension mismatch")

    S_inv = _invert(Sigma)
    d2 = float(r @ S_inv @ r)
    return float(np.sqrt(max(d2, 0.0)))

def mahalanobis_batch(R: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    S_inv = _invert(Sigma)

    mid = R @ S_inv
    d2 = np.sum(mid * R, axis=1)
    return np.sqrt(np.maximum(d2, 0.0))