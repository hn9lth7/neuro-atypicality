from __future__ import annotations

import numpy as np
from math import factorial

def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < order * delay:
        return float("nan")
    perms: dict[tuple, int] = {}
    for i in range(n - (order - 1) * delay):
        window = x[i : i + order * delay : delay]
        pattern = tuple(np.argsort(window))
        perms[pattern] = perms.get(pattern, 0) + 1
    counts = np.array(list(perms.values()), dtype=float)
    p = counts / counts.sum()
    H = float(-np.sum(p * np.log2(p)))
    return H / np.log2(factorial(order))