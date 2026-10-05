from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import load_raw_bids
from nai.preprocessing import preprocess_minimal
from nai.qc import compute_qc
from nai.spectral.power import compute_band_powers
from nai.spectral.entropy import compute_spectral_entropy
from nai.features.blocks import SE_FEATURES
from nai.connectivity.phase import compute_plv_band
from nai.graph.metrics import graph_metrics_dict
from nai.graph.spectral import laplacian_entropy
from nai.dynamics.windows import sliding_windows
from nai.dynamics.transitions import transition_series, dynamic_summary
from nai.connectivity.phase import compute_plv_band as plv_band

def resolve_bids_root() -> Path:
    root = PROJECT_ROOT / "data" / "raw" / "ds006780"
    if (root / "ds006780").is_dir():
        root = root / "ds006780"
    return root

def assert_finite_dict(d: dict, name: str) -> None:
    for k, v in d.items():
        if isinstance(v, (float, int, np.floating, np.integer)):
            if not np.isfinite(float(v)):
                raise ValueError(f"{name}.{k} is not finite: {v}")

def main() -> None:
    print("=" * 70)
    print("v1.1 integration pilot — sub-10025 run-01")
    print("=" * 70)

    raw = load_raw_bids(resolve_bids_root(), subject="10025", run="01")
    print("[1] load OK", raw)

    raw_c = preprocess_minimal(raw)
    qc = compute_qc(raw_c)
    print("[2] preprocess + QC", qc)
    assert qc["n_channels"] == 64, qc

    powers = compute_band_powers(raw_c)
    ent = compute_spectral_entropy(raw_c)
    se = {**powers, **ent}
    missing = [k for k in SE_FEATURES if k not in se]
    if missing:
        raise KeyError(f"SE missing keys: {missing}; have={list(se)}")
    se6 = {k: float(se[k]) for k in SE_FEATURES}
    assert_finite_dict(se6, "SE")
    print("[3] SE 6 features OK")
    for k, v in se6.items():
        print(f"     {k:28s} {v:.6g}")

    data = raw_c.get_data()  
    sfreq = float(raw_c.info["sfreq"])
    W = compute_plv_band(data, sfreq, 8.0, 13.0)
    assert W.shape == (64, 64)
    assert np.allclose(W, W.T)
    assert np.allclose(np.diag(W), 0)
    g = graph_metrics_dict(W)
    g["laplacian_entropy"] = laplacian_entropy(W)
    assert_finite_dict(g, "graph")
    print("[4] PLV+graph alpha OK", {k: round(v, 4) for k, v in g.items()})

    wins = sliding_windows(data.shape[1], sfreq, window_s=10.0, step_s=5.0)
    print(f"[5] windows n={len(wins)}")
    mats = []
    for a, b in wins:
        mats.append(plv_band(data[:, a:b], sfreq, 4.0, 8.0))
    deltas = transition_series(mats)
    dyn = dynamic_summary(deltas)
    assert_finite_dict(
        {k: v for k, v in dyn.items() if k != "n_transitions" or v > 0},
        "dyn",
    )
    print("[5] dynamics OK", dyn)

    print("=" * 70)
    print("PILOT PASS — library path works on one subject")
    print("Next: assemble 54 features + score vs frozen tables")
    print("=" * 70)

if __name__ == "__main__":
    main()