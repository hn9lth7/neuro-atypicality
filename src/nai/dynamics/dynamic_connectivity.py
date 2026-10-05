from __future__ import annotations

import numpy as np

from nai.connectivity.phase import compute_plv_band
from nai.dynamics.transitions import transition_magnitudes, summarize_transitions
from nai.dynamics.windows import sliding_windows

def windowed_plv_series(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    window_s: float = 10.0,
    step_s: float = 5.0,
) -> list[np.ndarray]:
    wins = sliding_windows(data.shape[1], sfreq, window_s, step_s)
    out = []
    for a, b in wins:
        out.append(compute_plv_band(data[:, a:b], sfreq, fmin, fmax))
    return out

def plv_transition_summary(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    window_s: float = 10.0,
    step_s: float = 5.0,
) -> dict[str, float]:
    series = windowed_plv_series(data, sfreq, fmin, fmax, window_s, step_s)
    if len(series) < 2:
        return {
            "mean_delta": float("nan"),
            "cv_delta": float("nan"),
            "n_windows": float(len(series)),
        }
    deltas = transition_magnitudes(series)
    stats = summarize_transitions(deltas)
    stats["n_windows"] = float(len(series))
    return stats