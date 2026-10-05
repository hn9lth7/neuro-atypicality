from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import load_raw_bids
from nai.preprocessing import preprocess_minimal
from nai.spectral.power import compute_band_powers
from nai.spectral.entropy import compute_spectral_entropy
from nai.connectivity.phase import compute_plv_band, bandpass_filter, compute_plv
from nai.graph.metrics import graph_metrics_dict
from nai.graph.spectral import laplacian_entropy
from nai.dynamics.windows import sliding_windows
from nai.dynamics.transitions import transition_series, dynamic_summary
from nai.features.blocks import (
    SE_FEATURES,
    C_FEATURES,
    G_FEATURES,
    D_FEATURES,
    ALL_BLOCKS,
    BLOCK_DIMS,
)

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

WINDOW_S = 10.0
STEP_S = 5.0

def resolve_bids_root() -> Path:
    root = PROJECT_ROOT / "data" / "raw" / "ds006780"
    if (root / "ds006780").is_dir():
        root = root / "ds006780"
    return root

def plv_mean_median(W: np.ndarray) -> tuple[float, float]:
    iu = np.triu_indices_from(W, k=1)
    vals = W[iu]
    return float(vals.mean()), float(np.median(vals))

def assemble_se(raw) -> dict[str, float]:
    powers = compute_band_powers(raw)
    ent = compute_spectral_entropy(raw)
    feat = {**powers, **ent}
    return {k: float(feat[k]) for k in SE_FEATURES}

def assemble_c_g(data: np.ndarray, sfreq: float) -> tuple[dict[str, float], dict[str, float]]:
    c: dict[str, float] = {}
    g: dict[str, float] = {}
    for band, (fmin, fmax) in BANDS.items():
        W = compute_plv_band(data, sfreq, fmin, fmax)
        mu, med = plv_mean_median(W)
        c[f"plv_mean_{band}"] = mu
        c[f"plv_median_{band}"] = med

        m = graph_metrics_dict(W)
        m["laplacian_entropy"] = float(laplacian_entropy(W))
        for metric in (
            "mean_degree",
            "degree_cv",
            "clustering",
            "global_efficiency",
            "mean_path_length",
            "laplacian_entropy",
        ):
            g[f"{metric}_{band}"] = float(m[metric])
    return c, g

def assemble_d(data, sfreq):
    d = {}
    wins = sliding_windows(data.shape[1], sfreq, WINDOW_S, STEP_S)
    for band, (fmin, fmax) in BANDS.items():
        x = bandpass_filter(data, sfreq, fmin, fmax) 
        mats, deg_cvs = [], []
        for a, b in wins:
            W = compute_plv(x[:, a:b], clean=True)  
            mats.append(W)
            deg_cvs.append(graph_metrics_dict(W)["degree_cv"])
        summary = dynamic_summary(transition_series(mats))
        d[f"mean_delta_{band}"] = float(summary["mean_delta"])
        d[f"cv_delta_{band}"] = float(summary["cv_delta"])
        deg_cvs = np.asarray(deg_cvs, float)
        mu = float(deg_cvs.mean())
        d[f"mean_degree_cv_{band}"] = mu
        d[f"temporal_cv_degree_cv_{band}"] = float(deg_cvs.std() / (mu + 1e-12))
    return d

def main() -> None:
    print("=" * 72)
    print("v1.1 — 54-D feature assembly  |  sub-10025 run-01")
    print("=" * 72)
    print("BLOCK_DIMS:", BLOCK_DIMS)

    raw = load_raw_bids(resolve_bids_root(), subject="10025", run="01")
    raw = preprocess_minimal(raw)
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])

    se = assemble_se(raw)
    c, g = assemble_c_g(data, sfreq)
    dfeat = assemble_d(data, sfreq)

    features: dict[str, float] = {}
    features.update(se)
    features.update(c)
    features.update(g)
    features.update(dfeat)

    ordered_names = SE_FEATURES + C_FEATURES + G_FEATURES + D_FEATURES
    assert len(ordered_names) == 54, len(ordered_names)

    missing = [k for k in ordered_names if k not in features]
    extra = [k for k in features if k not in ordered_names]
    if missing:
        raise KeyError(f"missing: {missing}")
    if extra:
        raise KeyError(f"extra (not in canonical): {extra}")

    vec = np.array([features[k] for k in ordered_names], dtype=float)
    if not np.all(np.isfinite(vec)):
        bad = [ordered_names[i] for i, v in enumerate(vec) if not np.isfinite(v)]
        raise ValueError(f"non-finite: {bad}")

    print(f"TOTAL features: {len(ordered_names)}  (assert 54)")
    print(f"SE={len(se)} C={len(c)} G={len(g)} D={len(dfeat)}")
    print("-" * 72)
    for k in ordered_names:
        print(f"  {k:32s}  {features[k]:.8g}")
    print("=" * 72)
    print("ASSEMBLY PASS — 54 canonical features, finite, ordered")
    print("=" * 72)

    out = PROJECT_ROOT / "results" / "features" / "v11_assembly_sub-10025_run-01.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("participant_id,run," + ",".join(ordered_names) + "\n")
        f.write(
            "sub-10025,01,"
            + ",".join(f"{features[k]:.10g}" for k in ordered_names)
            + "\n"
        )
    print(f"Saved → {out}")

if __name__ == "__main__":
    main()