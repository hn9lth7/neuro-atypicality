from __future__ import annotations

from typing import Any

import numpy as np

from nai.normative.covariance import empirical_covariance, regularize_covariance
from nai.normative.regression import compute_residuals, fit_age_models

def fit_td_reference(
    X_td: np.ndarray,
    age_td: np.ndarray,
    lam: float = 0.1,
) -> dict[str, Any]:
    models = fit_age_models(X_td, age_td)
    R = compute_residuals(X_td, age_td, models)
    Sigma_raw = empirical_covariance(R)
    Sigma = regularize_covariance(Sigma_raw, lam)
    return {
        "models": models,
        "Sigma_raw": Sigma_raw,
        "Sigma": Sigma,
        "lam": float(lam),
        "n_td": int(X_td.shape[0]),
        "n_features": int(X_td.shape[1]),
    }