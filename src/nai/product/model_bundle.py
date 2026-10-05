from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class ModelBundle:
    root: Path
    manifest: dict
    feature_schema: dict[str, list[str]]
    age_models: dict  
    covariance: dict[str, np.ndarray]  
    lambda_: float

    def feature_names(self, block: str) -> list[str]:
        return list(self.feature_schema[block])

def load_bundle(root: str | Path) -> ModelBundle:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    age = json.loads((root / "age_models.json").read_text(encoding="utf-8"))
    cov = {
        b: np.load(root / f"covariance_{b}.npy")
        for b in ("SE", "C", "G", "D")
    }
    for b, feats in schema.items():
        assert cov[b].shape == (len(feats), len(feats)), (b, cov[b].shape)
        assert len(age["blocks"][b]) == len(feats), b
    return ModelBundle(
        root=root,
        manifest=manifest,
        feature_schema=schema,
        age_models=age,
        covariance=cov,
        lambda_=float(manifest.get("lambda", 0.1)),
    )

def residuals_from_bundle(
    X: np.ndarray,
    age: np.ndarray,
    block: str,
    bundle: ModelBundle,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    age = np.asarray(age, dtype=float).reshape(-1)
    names = bundle.feature_names(block)
    coeffs = bundle.age_models["blocks"][block]
    intercept = np.array([coeffs[n]["intercept"] for n in names])
    slope = np.array([coeffs[n]["slope"] for n in names])
    if X.ndim == 1:
        X = X.reshape(1, -1)
        age = age.reshape(1)
    X_hat = intercept[None, :] + slope[None, :] * age[:, None]
    return X - X_hat

