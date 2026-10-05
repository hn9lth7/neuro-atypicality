from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

ABS_BANDS = ("delta", "theta", "alpha", "beta", "gamma")
EPS = 1e-12

def aggregate_se_subject(run_df: pd.DataFrame) -> dict[str, float]:
    required = [f"{b}_abs" for b in ABS_BANDS] + ["spectral_entropy_mean"]
    missing = [c for c in required if c not in run_df.columns]
    if missing:
        raise KeyError(f"aggregate_se_subject missing columns: {missing}")

    mean_abs = {b: float(run_df[f"{b}_abs"].mean()) for b in ABS_BANDS}
    total = sum(mean_abs.values()) + EPS

    out: dict[str, float] = {f"{b}_abs": mean_abs[b] for b in ABS_BANDS}
    for b in ABS_BANDS:
        out[f"{b}_rel"] = mean_abs[b] / total

    out["log_theta_alpha"] = float(
        np.log((mean_abs["theta"] + EPS) / (mean_abs["alpha"] + EPS))
    )
    out["log_theta_beta"] = float(
        np.log((mean_abs["theta"] + EPS) / (mean_abs["beta"] + EPS))
    )
    out["spectral_entropy_mean"] = float(run_df["spectral_entropy_mean"].mean())
    return out

def mean_columns(run_df: pd.DataFrame, columns: Iterable[str]) -> dict[str, float]:
    out = {}
    for c in columns:
        if c not in run_df.columns:
            raise KeyError(c)
        out[c] = float(run_df[c].mean())
    return out