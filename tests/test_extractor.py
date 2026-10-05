from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nai.features.blocks import (
    ALL_BLOCKS,
    C_FEATURES,
    D_FEATURES,
    G_FEATURES,
    SE_FEATURES,
)
from nai.features.extractor import score_nai_from_dataframe

def _synthetic_df(n_td: int = 12, n_asd: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_td + n_asd
    data: dict = {
        "participant_id": [f"s{i:02d}" for i in range(n)],
        "age": rng.uniform(8.0, 13.0, n),
        "group": ["TD"] * n_td + ["ASD"] * n_asd,
    }
    for cols in (SE_FEATURES, C_FEATURES, G_FEATURES, D_FEATURES):
        for c in cols:
            data[c] = rng.standard_normal(n)
    return pd.DataFrame(data)

def test_output_columns_and_nrows():
    df = _synthetic_df()
    out = score_nai_from_dataframe(df)
    assert len(out) == len(df)
    for col in ["D_SE", "D_C", "D_G", "D_D", "NAI", "age", "group"]:
        assert col in out.columns

def test_finite_scores():
    out = score_nai_from_dataframe(_synthetic_df())
    for col in ["D_SE", "D_C", "D_G", "D_D", "NAI"]:
        assert np.isfinite(out[col].values).all()

def test_nai_equals_mean_of_blocks():
    out = score_nai_from_dataframe(_synthetic_df())
    expected = out[["D_SE", "D_C", "D_G", "D_D"]].mean(axis=1).values
    assert np.allclose(out["NAI"].values, expected, rtol=0, atol=1e-12)

def test_td_only_fitting_invariant():
    df = _synthetic_df(n_td=15, n_asd=3, seed=1)
    out1 = score_nai_from_dataframe(df)

    df2 = df.copy()
    asd = df2["group"] == "ASD"
    for cols in (SE_FEATURES, C_FEATURES, G_FEATURES, D_FEATURES):
        for c in cols:
            df2.loc[asd, c] = 1e6  # extreme

    out2 = score_nai_from_dataframe(df2)
    td = out1["group"] == "TD"
    for col in ["D_SE", "D_C", "D_G", "D_D", "NAI"]:
        assert np.allclose(
            out1.loc[td, col].values,
            out2.loc[td, col].values,
            rtol=0,
            atol=1e-10,
        )

def test_missing_block_raises():
    df = _synthetic_df()
    df = df.drop(columns=[SE_FEATURES[0]])
    with pytest.raises(KeyError):
        score_nai_from_dataframe(df)