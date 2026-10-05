from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

from .regression import fit_age_models, compute_residuals, predict_age
from .covariance import empirical_covariance, regularize_covariance
from .mahalanobis import mahalanobis_distance

def loo_mahalanobis(
    X: np.ndarray,
    age: np.ndarray,
    lam: float = 0.10,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    age = np.asarray(age, dtype=float)
    n = X.shape[0]
    D = np.empty(n, dtype=float)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, age_tr = X[mask], age[mask]
        models = fit_age_models(X_tr, age_tr)
        R_tr = compute_residuals(X_tr, age_tr, models)
        Sigma = regularize_covariance(empirical_covariance(R_tr), lam)

        r_i = X[i] - predict_age(models, age[i]).ravel()
        D[i] = mahalanobis_distance(r_i, Sigma)

    return D

def lambda_sensitivity(
    X_td: np.ndarray,
    age_td: np.ndarray,
    X_score: np.ndarray,
    age_score: np.ndarray,
    lambdas: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50),
) -> dict[float, np.ndarray]:
    X_td = np.asarray(X_td, dtype=float)
    age_td = np.asarray(age_td, dtype=float)
    X_score = np.asarray(X_score, dtype=float)
    age_score = np.asarray(age_score, dtype=float)

    models = fit_age_models(X_td, age_td)
    R_td = compute_residuals(X_td, age_td, models)
    Sigma_raw = empirical_covariance(R_td)
    R_score = compute_residuals(X_score, age_score, models)

    out: dict[float, np.ndarray] = {}
    for lam in lambdas:
        Sigma = regularize_covariance(Sigma_raw, float(lam))
        from .mahalanobis import mahalanobis_batch
        out[float(lam)] = mahalanobis_batch(R_score, Sigma)
    return out

def empirical_percentile(reference: np.ndarray, value: float) -> float:
    reference = np.asarray(reference, dtype=float)
    return float(stats.percentileofscore(reference, value, kind="rank"))