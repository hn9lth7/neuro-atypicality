from __future__ import annotations

import numpy as np

def frobenius_delta(W1: np.ndarray, W2: np.ndarray, eps: float = 1e-12) -> float:
    W1 = np.asarray(W1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    num = np.linalg.norm(W2 - W1, ord="fro")
    den = np.linalg.norm(W1, ord="fro") + eps
    return float(num / den)

def transition_series(matrices: list[np.ndarray]) -> np.ndarray:
    if len(matrices) < 2:
        return np.array([], dtype=float)
    return np.array(
        [frobenius_delta(matrices[t], matrices[t + 1]) for t in range(len(matrices) - 1)],
        dtype=float,
    )

def dynamic_summary(deltas: np.ndarray) -> dict[str, float]:
    deltas = np.asarray(deltas, dtype=float).ravel()
    if deltas.size == 0:
        return {
            "n_transitions": 0.0,
            "mean_delta": float("nan"),
            "std_delta": float("nan"),
            "cv_delta": float("nan"),
            "max_delta": float("nan"),
            "temporal_entropy": float("nan"),
        }

    mean_d = float(deltas.mean())
    std_d = float(deltas.std())
    cv = std_d / (mean_d + 1e-12)

    n_bins = min(8, max(3, deltas.size // 2))
    hist, _ = np.histogram(deltas, bins=n_bins, density=True)
    hist = hist[hist > 0]
    p = hist / hist.sum()
    entropy = float(-np.sum(p * np.log(p + 1e-12)))

    return {
        "n_transitions": float(deltas.size),
        "mean_delta": mean_d,
        "std_delta": std_d,
        "cv_delta": cv,
        "max_delta": float(deltas.max()),
        "temporal_entropy": entropy,
    }