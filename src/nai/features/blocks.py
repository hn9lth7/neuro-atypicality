from __future__ import annotations

SE_FEATURES: list[str] = [
    "alpha_rel",
    "beta_rel",
    "theta_rel",
    "spectral_entropy_mean",
    "log_theta_alpha",
    "log_theta_beta",
]

_BANDS = ("theta", "alpha", "beta", "gamma")

C_FEATURES: list[str] = (
    [f"plv_mean_{b}" for b in _BANDS]
    + [f"plv_median_{b}" for b in _BANDS]
)

_GRAPH_METRICS = (
    "mean_degree",
    "degree_cv",
    "clustering",
    "global_efficiency",
    "mean_path_length",
    "laplacian_entropy",
)

G_FEATURES: list[str] = [
    f"{m}_{b}" for m in _GRAPH_METRICS for b in _BANDS
]

G_AUX_FEATURES: list[str] = [f"lambda2_{b}" for b in _BANDS]

_DYNAMIC_METRICS = (
    "mean_delta",
    "cv_delta",
    "mean_degree_cv",
    "temporal_cv_degree_cv",
)

D_FEATURES: list[str] = [
    f"{m}_{b}" for m in _DYNAMIC_METRICS for b in _BANDS
]

ALL_BLOCKS: dict[str, list[str]] = {
    "SE": SE_FEATURES,
    "C": C_FEATURES,
    "G": G_FEATURES,
    "D": D_FEATURES,
}

BLOCK_DIMS: dict[str, int] = {
    name: len(feats) for name, feats in ALL_BLOCKS.items()
}

assert BLOCK_DIMS["SE"] == 6
assert BLOCK_DIMS["C"] == 8
assert BLOCK_DIMS["G"] == 24
assert BLOCK_DIMS["D"] == 16