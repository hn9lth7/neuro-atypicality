from __future__ import annotations

import numpy as np

from nai.connectivity.matrices import clean_connectivity_matrix, threshold_matrix

def connectivity_to_graph_matrix(
    W: np.ndarray,
    density: float | None = None,
) -> np.ndarray:
    W = clean_connectivity_matrix(W)
    if density is None:
        return W
    return threshold_matrix(W, density)