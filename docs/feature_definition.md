# Feature Definition NAI v1.0

Single source of truth in code: `src/nai/features/blocks.py`.

## SE Spectral / Entropy ($p=6$)

| Feature | Definition (Summary) |
|:---|:---|
| `alpha_rel`, `beta_rel`, `theta_rel` | Relative band power |
| `spectral_entropy_mean` | Mean spectral entropy across channels |
| `log_theta_alpha`, `log_theta_beta` | $\log((P_	heta+ arepsilon)/(P_ lpha+ arepsilon))$, similarly for beta |

Relative powers are compositional; log-ratios reduce pure ratio instability.

## C Connectivity ($p=8$)

Phase-locking value (PLV) on band-passed signals, bands $	heta,  lpha,  eta, \gamma$:

- `plv_mean_*`
- `plv_median_*`

## G Static Graph ($p=24$)

From weighted PLV graphs (dense, diagonal zeroed), per band:

- `mean_degree`, `degree_cv`, `clustering`
- `global_efficiency`, `mean_path_length`, `laplacian_entropy`

**Not in core NAI:** `lambda2_*` (algebraic connectivity) auxiliary only.

## D Dynamic ($p=16$)

Sliding windows: **10 s**, step **5 s**.

Per band:

- `mean_delta`, `cv_delta` PLV matrix transition magnitude statistics
- `mean_degree_cv`, `temporal_cv_degree_cv` degree-heterogeneity dynamics

## Aggregation

Run-level features **subject-level mean** before normative modeling.