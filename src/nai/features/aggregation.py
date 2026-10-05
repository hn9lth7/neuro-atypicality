from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

META_CANDIDATES = (
    "participant_id",
    "age",
    "sex",
    "group",
    "qc_flag",
    "handedness",
)

def subject_level_mean(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    id_col: str = "participant_id",
    meta_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    if id_col not in df.columns:
        raise KeyError(f"Missing id column: {id_col}")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:10]}")

    if meta_cols is None:
        meta_cols = [c for c in META_CANDIDATES if c in df.columns and c != id_col]

    g = df.groupby(id_col, sort=True)

    feat = g[list(feature_cols)].mean()
    n_runs = g.size().rename("n_runs")

    parts = [feat, n_runs]
    for c in meta_cols:
        if c in df.columns:
            parts.append(g[c].first().rename(c))

    out = pd.concat(parts, axis=1).reset_index()
    return out

def assert_one_row_per_subject(df: pd.DataFrame, id_col: str = "participant_id") -> None:
    n = len(df)
    n_u = df[id_col].nunique()
    if n != n_u:
        raise ValueError(f"Expected 1 row per {id_col}, got {n} rows / {n_u} ids")