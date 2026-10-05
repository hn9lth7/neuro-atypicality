from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

def ledoit_wolf_covariance(R: np.ndarray, *, assume_centered: bool = True) -> tuple[np.ndarray, float]:
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"R must be 2D, got shape {R.shape}")
    n, p = R.shape
    if n < 2:
        raise ValueError("Need at least 2 samples for covariance")
    if p < 1:
        raise ValueError("Need at least 1 feature")

    lw = LedoitWolf(assume_centered=assume_centered, store_precision=False)
    lw.fit(R)
    Sigma = np.asarray(lw.covariance_, dtype=float)
    Sigma = 0.5 * (Sigma + Sigma.T)
    shrinkage = float(lw.shrinkage_)
    return Sigma, shrinkage

def is_symmetric(A: np.ndarray, tol: float = 1e-10) -> bool:
    A = np.asarray(A, dtype=float)
    return bool(np.allclose(A, A.T, atol=tol))

def is_positive_definite(A: np.ndarray, tol: float = 1e-10) -> bool:
    A = np.asarray(A, dtype=float)
    try:
        np.linalg.cholesky(A + tol * np.eye(A.shape[0]))
        return True
    except np.linalg.LinAlgError:
        return False