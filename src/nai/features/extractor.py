from __future__ import annotations

import numpy as np
import pandas as pd

from nai.features.blocks import (
    ALL_BLOCKS,
    C_FEATURES,
    D_FEATURES,
    G_FEATURES,
    SE_FEATURES,
)
from nai.nai.composite import compute_nai
from nai.normative.covariance import regularize_covariance
from nai.normative.mahalanobis import mahalanobis_distance
from nai.normative.regression import compute_residuals, fit_age_models

FEATURE_MAP = {
    "SE": SE_FEATURES,
    "C": C_FEATURES,
    "G": G_FEATURES,
    "D": D_FEATURES,
}

def score_nai_from_dataframe(
    df: pd.DataFrame,
    *,
    group_col: str = "group",
    age_col: str = "age",
    id_col: str = "participant_id",
    td_label: str = "TD",
    lam: float = 0.10,
) -> pd.DataFrame:
    if age_col not in df.columns or group_col not in df.columns:
        raise KeyError(f"df must contain '{age_col}' and '{group_col}'")

    mask_td = (df[group_col].values == td_label)
    if mask_td.sum() < 2:
        raise ValueError("Need at least 2 TD subjects to fit normative model")

    age = df[age_col].values.astype(float)
    n = len(df)

    out_cols: dict[str, object] = {}
    if id_col in df.columns:
        out_cols[id_col] = df[id_col].values
    out_cols[age_col] = age
    out_cols[group_col] = df[group_col].values

    distances: dict[str, np.ndarray] = {}
    for name, cols in FEATURE_MAP.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Block {name}: missing columns {missing[:8]}")

        X = df[cols].values.astype(float)
        models = fit_age_models(X[mask_td], age[mask_td])
        R_td = compute_residuals(X[mask_td], age[mask_td], models)
        R_all = compute_residuals(X, age, models)
        Sigma = regularize_covariance(np.cov(R_td, rowvar=False), lam=lam)
        d = np.array(
            [mahalanobis_distance(R_all[i], Sigma) for i in range(n)],
            dtype=float,
        )
        distances[name] = d
        out_cols[f"D_{name}"] = d

    nai = np.array(
        [
            compute_nai({b: float(distances[b][i]) for b in ALL_BLOCKS})
            for i in range(n)
        ],
        dtype=float,
    )
    out_cols["NAI"] = nai
    return pd.DataFrame(out_cols)