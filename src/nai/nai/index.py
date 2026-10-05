from __future__ import annotations

from nai.nai.composite import compute_nai
from nai.nai.components import BlockDistances
from nai.nai.weights import equal_weights

def nai_from_blocks(blocks: BlockDistances, weights: dict[str, float] | None = None) -> float:
    w = weights or equal_weights()
    d = blocks.as_dict()
    payload = {
        "SE": d["D_SE"],
        "C": d["D_C"],
        "G": d["D_G"],
        "D": d["D_D"],
    }
    try:
        return float(compute_nai(payload))
    except Exception:
        return float(sum(w[k] * payload[k] for k in payload) / sum(w.values()))