from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import shortest_path

from nai.connectivity.matrices import clean_connectivity_matrix

def _prepare(W: np.ndarray) -> np.ndarray:
    return clean_connectivity_matrix(W)

def mean_degree(W: np.ndarray) -> float:
    W = _prepare(W)
    d = W.sum(axis=1)
    return float(d.mean())

def degree_cv(W: np.ndarray) -> float:
    W = _prepare(W)
    d = W.sum(axis=1)
    mu = d.mean()
    return float(d.std() / (mu + 1e-12))

def weighted_clustering(W: np.ndarray) -> float:
    W = _prepare(W)
    W_cbrt = np.cbrt(W)
    triangles = np.diag(W_cbrt @ W_cbrt @ W_cbrt)
    deg = (W > 0).sum(axis=1).astype(float)
    possible = deg * (deg - 1)
    possible[possible <= 0] = np.nan
    c = triangles / possible
    return float(np.nanmean(c))

def global_efficiency(W: np.ndarray) -> float:
    W = _prepare(W)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(W > 0, 1.0 / W, 0.0)
    dist = shortest_path(D, directed=False, unweighted=False)
    n = W.shape[0]
    inv = np.zeros_like(dist)
    mask = (dist > 0) & np.isfinite(dist)
    inv[mask] = 1.0 / dist[mask]
    return float(inv.sum() / (n * (n - 1)))

def mean_path_length(W: np.ndarray) -> float:
    W = _prepare(W)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(W > 0, 1.0 / W, 0.0)
    dist = shortest_path(D, directed=False, unweighted=False)
    mask = (dist > 0) & np.isfinite(dist) & (dist < 1e6)
    if mask.sum() == 0:
        return float("nan")
    return float(dist[mask].mean())

def graph_metrics_dict(W: np.ndarray) -> dict[str, float]:
    return {
        "mean_degree": mean_degree(W),
        "degree_cv": degree_cv(W),
        "clustering": weighted_clustering(W),
        "global_efficiency": global_efficiency(W),
        "mean_path_length": mean_path_length(W),
    }