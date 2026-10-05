from __future__ import annotations

import numpy as np

from nai.entropy.permutation_entropy import permutation_entropy
from nai.entropy.sample_entropy import sample_entropy

def channel_complexity(x: np.ndarray) -> dict[str, float]:
    return {
        "permutation_entropy": permutation_entropy(x),
        "sample_entropy": sample_entropy(x),
    }

def mean_complexity(data: np.ndarray) -> dict[str, float]:
    pe, se = [], []
    for i in range(data.shape[0]):
        d = channel_complexity(data[i])
        pe.append(d["permutation_entropy"])
        se.append(d["sample_entropy"])
    return {
        "permutation_entropy_mean": float(np.nanmean(pe)),
        "sample_entropy_mean": float(np.nanmean(se)),
    }