from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from nai.connectivity.matrices import clean_connectivity_matrix

def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    order: int = 4,
) -> np.ndarray:
    nyq = sfreq / 2.0
    low = fmin / nyq
    high = min(fmax / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=1)

def compute_plv(data: np.ndarray, clean: bool = True) -> np.ndarray:
    analytic = hilbert(data, axis=1)
    phase = np.angle(analytic)
    n_ch = phase.shape[0]

    W = np.zeros((n_ch, n_ch), dtype=float)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            dphi = phase[i] - phase[j]
            plv = float(np.abs(np.mean(np.exp(1j * dphi))))
            W[i, j] = W[j, i] = plv

    if clean:
        W = clean_connectivity_matrix(W)
    else:
        np.fill_diagonal(W, 0.0)

    return W

def compute_plv_band(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    order: int = 4,
) -> np.ndarray:
    x = bandpass_filter(data, sfreq, fmin, fmax, order=order)
    return compute_plv(x, clean=True)