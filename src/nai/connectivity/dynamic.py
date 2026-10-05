from __future__ import annotations
import numpy as np
from .phase import bandpass_filter, compute_plv

def sliding_windows(data: np.ndarray, sfreq: float,
                    window_sec: float = 10.0,
                    step_sec: float = 5.0):
    n_times = data.shape[1]
    win = int(window_sec * sfreq)
    step = int(step_sec * sfreq)

    starts = range(0, n_times - win + 1, step)
    for start in starts:
        end = start + win
        yield start, end, data[:, start:end]

def compute_windowed_plv(data: np.ndarray, sfreq: float,
                         fmin: float, fmax: float,
                         window_sec: float = 10.0,
                         step_sec: float = 5.0) -> list[np.ndarray]:
    data_bp = bandpass_filter(data, sfreq, fmin, fmax)
    matrices = []
    for _, _, wdata in sliding_windows(data_bp, sfreq, window_sec, step_sec):
        W = compute_plv(wdata)
        np.fill_diagonal(W, 0.0)
        matrices.append(W)
    return matrices