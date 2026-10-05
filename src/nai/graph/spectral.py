from __future__ import annotations
import numpy as np

def laplacian(W: np.ndarray) -> np.ndarray:
    d = W.sum(axis=1)
    D = np.diag(d)
    return D - W

def algebraic_connectivity(W: np.ndarray) -> float:
    L = laplacian(W)
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])

def laplacian_entropy(W: np.ndarray) -> float:
    L = laplacian(W)
    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.maximum(eigvals, 0.0)        
    total = eigvals.sum() + 1e-12
    p = eigvals / total
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))