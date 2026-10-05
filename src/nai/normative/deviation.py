from __future__ import annotations

import numpy as np

from nai.normative.covariance import empirical_covariance, regularize_covariance
from nai.normative.mahalanobis import mahalanobis_distance
from nai.normative.regression import compute_residuals, fit_age_models

def residual_mahalanobis(
    X_td: np.ndarray,
    age_td: np.ndarray,
    x: np.ndarray,
    age: float,
    lam: float = 0.1,
) -> float:
    models = fit_age_models(X_td, age_td)
    R = compute_residuals(X_td, age_td, models)
    Sigma = regularize_covariance(empirical_covariance(R), lam)
    r = compute_residuals(np.asarray(x, float).reshape(1, -1), np.array([age]), models)[0]
    return float(mahalanobis_distance(r, Sigma))