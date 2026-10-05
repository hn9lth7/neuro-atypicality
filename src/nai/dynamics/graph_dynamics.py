from __future__ import annotations

import numpy as np

from nai.dynamics.dynamic_connectivity import windowed_plv_series
from nai.graph.metrics import graph_metrics_dict

def graph_metric_trajectories(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    window_s: float = 10.0,
    step_s: float = 5.0,
) -> dict[str, np.ndarray]:
    series = windowed_plv_series(data, sfreq, fmin, fmax, window_s, step_s)
    keys = ["mean_degree", "degree_cv", "clustering", "global_efficiency", "mean_path_length"]
    traj = {k: [] for k in keys}
    for W in series:
        m = graph_metrics_dict(W)
        for k in keys:
            traj[k].append(m[k])
    return {k: np.asarray(v, dtype=float) for k, v in traj.items()}

def summarize_graph_trajectories(traj: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, arr in traj.items():
        out[f"{k}_mean"] = float(np.nanmean(arr))
        out[f"{k}_std"] = float(np.nanstd(arr))
        mu = float(np.nanmean(arr))
        out[f"{k}_cv"] = float(np.nanstd(arr) / (mu + 1e-12))
    return out