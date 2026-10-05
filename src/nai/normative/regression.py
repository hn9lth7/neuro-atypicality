from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def fit_age_models(X: np.ndarray, age: np.ndarray) -> list[Any]:
    X = np.asarray(X, dtype=float)
    age = np.asarray(age, dtype=float).reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if age.shape[0] != X.shape[0]:
        raise ValueError("age length must match X rows")

    models: list[Any] = []
    for j in range(X.shape[1]):
        m = LinearRegression()
        m.fit(age, X[:, j])
        models.append(m)
    return models

def fit_age_models_quadratic(X: np.ndarray, age: np.ndarray) -> list[Any]:
    X = np.asarray(X, dtype=float)
    age = np.asarray(age, dtype=float).reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if age.shape[0] != X.shape[0]:
        raise ValueError("age length must match X rows")

    models: list[Any] = []
    for j in range(X.shape[1]):
        model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression(),
        )
        model.fit(age, X[:, j])
        models.append(model)
    return models

def predict_age(models: list[Any], age: np.ndarray) -> np.ndarray:
    age = np.asarray(age, dtype=float).reshape(-1, 1)
    cols = [m.predict(age) for m in models]
    return np.column_stack(cols)

def compute_residuals(
    X: np.ndarray,
    age: np.ndarray,
    models: list[Any],
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X_hat = predict_age(models, age)
    if X.shape != X_hat.shape:
        raise ValueError(f"Shape mismatch: X {X.shape} vs X_hat {X_hat.shape}")
    return X - X_hat