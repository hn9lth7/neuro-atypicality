from __future__ import annotations

import numpy as np
from scipy.signal import coherence

from nai.connectivity.matrices import clean_connectivity_matrix

def compute_coherence_matrix(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    nperseg: int | None = None,
) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    n_ch = x.shape[0]
    if nperseg is None:
        nperseg = min(256, x.shape[1])
    W = np.zeros((n_ch, n_ch), dtype=float)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            f, Cxy = coherence(x[i], x[j], fs=sfreq, nperseg=nperseg)
            mask = (f >= fmin) & (f <= fmax)
            val = float(np.mean(Cxy[mask])) if mask.any() else 0.0
            W[i, j] = W[j, i] = val
    return clean_connectivity_matrix(W)