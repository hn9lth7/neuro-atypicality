from __future__ import annotations

import numpy as np
import pandas as pd

from nai.product.model_bundle import ModelBundle, residuals_from_bundle

from nai.normative.mahalanobis import mahalanobis_distance
from nai.nai.composite import compute_nai

def score_row(row, bundle: ModelBundle) -> dict:
    age = float(row["age"])
    D = {}
    for block, feats in bundle.feature_schema.items():
        x = np.array([float(row[f]) for f in feats], dtype=float)
        r = residuals_from_bundle(x, age, block, bundle).ravel()
        D[block] = float(mahalanobis_distance(r, bundle.covariance[block]))
    return {
        "D_SE": D["SE"],
        "D_C": D["C"],
        "D_G": D["G"],
        "D_D": D["D"],
        "NAI": float(compute_nai(D)),
    }