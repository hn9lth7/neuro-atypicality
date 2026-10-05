from __future__ import annotations

import numpy as np

def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if r is None:
        r = 0.2 * float(np.std(x))
    if n <= m + 1:
        return float("nan")

    def _count(mm: int) -> float:
        templates = np.array([x[i : i + mm] for i in range(n - mm)])
        c = 0
        for i in range(len(templates)):
            dif = np.max(np.abs(templates - templates[i]), axis=1)
            c += int(np.sum(dif <= r)) - 1 
        return float(c)

    B = _count(m)
    A = _count(m + 1)
    if B == 0 or A == 0:
        return float("nan")
    return float(-np.log(A / B))