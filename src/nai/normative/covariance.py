from __future__ import annotations

import numpy as np
from numpy.linalg import matrix_rank, cond

def empirical_covariance(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError("R must be 2-D")
    if R.shape[0] < 2:
        raise ValueError("Need at least 2 samples for covariance")
    return np.cov(R, rowvar=False)

def regularize_covariance(Sigma: np.ndarray, lam: float) -> np.ndarray:
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be square")
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0, 1]")

    p = Sigma.shape[0]
    target = (np.trace(Sigma) / p) * np.eye(p)
    return (1.0 - lam) * Sigma + lam * target

def covariance_diagnostics(
    Sigma: np.ndarray,
    lam: float = 0.10,
    tol: float = 1e-8,
) -> dict:
    Sigma = np.asarray(Sigma, dtype=float)
    S_reg = regularize_covariance(Sigma, lam)
    return {
        "n_features": int(Sigma.shape[0]),
        "rank_reg": int(matrix_rank(S_reg, tol=tol)),
        "cond_raw": float(cond(Sigma)),
        "cond_reg": float(cond(S_reg)),
        "lambda": float(lam),
    }