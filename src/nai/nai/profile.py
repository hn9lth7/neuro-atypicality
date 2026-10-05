from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from nai.normative.regression import fit_age_models, compute_residuals, predict_age
from nai.normative.covariance import empirical_covariance, regularize_covariance

def residual_z_profile(
    X_td: np.ndarray,
    age_td: np.ndarray,
    x: np.ndarray,
    age: float,
    feature_names: Sequence[str],
    lam: float = 0.10,
) -> pd.DataFrame:
    X_td = np.asarray(X_td, dtype=float)
    age_td = np.asarray(age_td, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    feature_names = list(feature_names)

    if X_td.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match n_features")
    if x.shape[0] != len(feature_names):
        raise ValueError("x length must match n_features")

    models = fit_age_models(X_td, age_td)
    R_td = compute_residuals(X_td, age_td, models)
    Sigma = regularize_covariance(empirical_covariance(R_td), lam)
    sigma = np.sqrt(np.diag(Sigma)) + 1e-12

    expected = predict_age(models, age).ravel()
    residual = x - expected
    z = residual / sigma

    profile = pd.DataFrame({
        "feature": feature_names,
        "observed": x,
        "age_expected": expected,
        "residual": residual,
        "residual_sd_reg": sigma,
        "z": z,
        "abs_z": np.abs(z),
    })
    return profile.sort_values("abs_z", ascending=False).reset_index(drop=True)