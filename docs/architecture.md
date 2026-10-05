# Architecture v1.0 — Neuro-Atypicality Index

## 1. Overview

**NAI** (Neuro-Atypicality Index) is an exploratory computational framework for quantifying multidimensional neurophysiological atypicality relative to an age-dependent typically developing (TD) reference population.

The framework is **not** a clinical diagnostic classifier.

Core principle:

```text
raw EEG ──> features ──> age-corrected residuals ──> block Mahalanobis distances ──> unified NAI ──> validation / residual profiles
```

Baseline composite (v1.0):

$$\text{NAI}_{\text{v1.0}} = \frac{1}{4} \left( D_{\text{SE}} + D_{\text{C}} + D_{\text{G}} + D_{\text{D}} \right)$$

Feature blocks:

| Symbol | Domain |
|--------|--------|
| $D_{\text{SE}}$ | Spectral / entropy |
| $D_{\text{C}}$ | Functional connectivity (PLV) |
| $D_{\text{G}}$ | Static graph structure |
| $D_{\text{D}}$ | Dynamic connectivity / graph features |

Block weights are equal by design and are **not** optimized on the available ASD sample.

---

## 2. Scientific Scope

NAI quantifies how an individual neurophysiological profile **deviates** from an age-dependent normative reference distribution.

Pipeline foundations:

1. EEG preprocessing and quality control  
2. Extraction of spectral, connectivity, graph, and dynamic features  
3. Age correction on the TD reference cohort  
4. Covariance estimation on age-corrected TD residuals  
5. Regularized Mahalanobis distance per feature block  
6. Integration of block distances into a composite index  
7. Leave-one-out validation and sensitivity analysis  
8. Residual profiles for multivariate interpretation  

**Status:** exploratory research framework.  
**Not:** a validated clinical diagnostic test.

---

## 3. Repository Layout

```text
neuro-atypicality/
├── data/raw/ds006780/
├── src/nai/
│   ├── io/                 # BIDS load, metadata
│   ├── preprocessing/      # Minimal EEG pipeline
│   ├── qc/                 # Signal quality
│   ├── spectral/           # Power, entropy
│   ├── connectivity/       # PLV, matrices, dynamic phase
│   ├── graph/              # Graph metrics, Laplacian spectrum
│   ├── dynamics/           # Transitions, temporal features
│   ├── features/           # Block contracts, aggregation
│   ├── normative/          # Regression, covariance, Mahalanobis, validation
│   ├── nai/                # Composite index, residual profiles
│   ├── statistics/         # Sensitivity, stability
│   └── visualization/
├── scripts/                # Orchestration only (02 … 30)
├── results/
│   ├── features/
│   ├── normative/
│   ├── validation/
│   └── figures/
├── docs/
│   ├── architecture.md
│   ├── mathematical_model.md
│   └── validation_protocol.md
└── requirements.txt
```

Detailed script inventory is maintained under `scripts/` (development history v0.1–v1.0).

---

## 4. Architectural Principles

### 4.1 Computation vs orchestration

| Layer | Location | Role |
|-------|----------|------|
| Library | `src/nai/` | Reusable math and feature logic |
| Scripts | `scripts/` | Load data, call library, save outputs |

Scripts must **not** reimplement normative mathematics.

### 4.2 Features vs normative model

Feature extraction yields subject-level matrices.  
Normative modelling is **block-agnostic**: the same residualisation, shrinkage, and Mahalanobis routines apply to SE, C, G, and D.

### 4.3 Single canonical cohort

All v1.0 blocks share one cohort definition:

| Rule | Value |
|------|--------|
| Base | QC-clean subjects |
| Exclude | `sub-10777` (extreme artifact) |
| Exclude | missing metadata |
| Retain | `sub-11025` with flag `very_low_alpha` |
| TD (norm fit) | $n_{\text{TD}} = 39$ |
| ASD (score only) | $n_{\text{ASD}} = 2$ |

The cohort must not change silently across blocks.

---

## 5. Feature Blocks

| Block | Dim $p$ | Content |
|-------|-----------|---------|
| **SE** | 6 | Relative band power, spectral entropy, log ratios |
| **C** | 8 | PLV mean and median × 4 bands |
| **G** | 24 | Six graph metrics × 4 bands |
| **D** | 16 | Four dynamic metrics × 4 bands |
| **Total** | **54** | |

### 5.1 SE — Spectral / entropy

$$p_{\text{SE}} = 6$$

Features: `alpha_rel`, `beta_rel`, `theta_rel`, `spectral_entropy_mean`, `log_theta_alpha`, `log_theta_beta`.

### 5.2 C — Connectivity

$$p_{\text{C}} = 8$$

Features: `plv_mean_{\theta,\alpha,\beta,\gamma}`, `plv_median_{\theta,\alpha,\beta,\gamma}` (PLV).

### 5.3 G — Static graph

$$p_{\text{G}} = 6 \times 4 = 24$$

Metrics: `mean_degree`, `degree_cv`, `clustering`, `global_efficiency`, `mean_path_length`, `laplacian_entropy`.  
Bands: $\theta, \alpha, \beta, \gamma$.  

Algebraic connectivity $\lambda_2$ is available but **not** in the core v1.0 G block.

### 5.4 D — Dynamics

$$p_{\text{D}} = 4 \times 4 = 16$$

Metrics: `mean_delta`, `cv_delta`, `mean_degree_cv`, `temporal_cv_degree_cv`.  

Interpretive substructure (not separate NAI weights):

- $D_\Delta$: transition variability (`mean_delta`, `cv_delta`)
- $D_H$: graph-state variability (`mean_degree_cv`, `temporal_cv_degree_cv`)

---

## 6. Data Flow

```text
BIDS EEG ──> IO ──> Preprocessing + QC
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Spectral    Connectivity   Dynamics
             │             │             │
             │        Static PLV ──> Graph
             │             │             │
             ▼             ▼             ▼
            SE            C, G           D
             └─────────────┬─────────────┘
                           ▼
                 Subject-level features
                           ▼
                 Canonical cohort filter
                           ▼
                 Age models (TD only)
                           ▼
                        Residuals r_B
                           ▼
                       Σ_{B,λ} (shrinkage)
                           ▼
                 D_SE, D_C, D_G, D_D
                           ▼
                       NAI v1.0
                    ╱            ╲
             Validation     Residual profile
```

---

## 7. Normative Modelling

For each block $B \in \{\text{SE}, \text{C}, \text{G}, \text{D}\}$ with matrix $X_B \in \mathbb{R}^{n \times p_B}$, the normative model is estimated **on TD subjects only**.

### 7.1 Age correction

For feature $j$:

$$x_j = \beta_{0j} + \beta_{1j}\,\text{age} + \varepsilon_j$$

fitted on TD. For age $a$:

$$\hat{x}_j(a) = \hat{\beta}_{0j} + \hat{\beta}_{1j}\,a, \qquad r_j = x_j - \hat{x}_j(a).$$

Residual vector:

$$\mathbf{r}_B = (r_1,\ldots,r_{p_B})^{\text{T}}.$$

Age correction **precedes** covariance estimation.

---

## 8. Residual Covariance

With \(R_{\mathrm{TD},B}\) the matrix of TD residuals:

$$
\Sigma_B = \mathrm{Cov}\bigl(R_{\mathrm{TD},B}\bigr).
$$

Regularization is required (correlated features, limited \(n_{\mathrm{TD}}\)).

---

## 9. Shrinkage

$$
\Sigma_{B,\lambda}
=
(1-\lambda)\,\Sigma_B
+
\lambda\,\frac{\mathrm{tr}(\Sigma_B)}{p_B}\,I.
$$

---

## 10. Block Mahalanobis Distance

$$D_B = \sqrt{\mathbf{r}_B^{\text{T}} \Sigma_{B,\lambda}^{-1} \mathbf{r}_B}$$

yielding $D_{\text{SE}}$, $D_{\text{C}}$, $D_{\text{G}}$, $D_{\text{D}}$.

These are **multivariate** deviations under the residual covariance, not sums of univariate $z$-scores.

---

## 11. Unified Index

$$\text{NAI}_{\text{v1.0}} = \frac{1}{4} \left( D_{\text{SE}} + D_{\text{C}} + D_{\text{G}} + D_{\text{D}} \right) = \sum_B w_B D_B, \qquad w_B = \tfrac{1}{4}$$

Equal weights avoid fitting the composite to $n_{\text{ASD}} = 2$.

---

## 12. Interpretation Levels

| Level | Object | Role |
|-------|--------|------|
| 1 | $\text{NAI}_{\text{v1.0}}$ | Overall multidimensional deviation |
| 2 | $D_{\text{SE}}, D_{\text{C}}, D_{\text{G}}, D_{\text{D}}$ | Domain contributions |
| 3 | Feature residual $z$ | Descriptive drivers (require stability evidence) |

A large marginal $z$ is not automatically a stable physiological effect.

---

## 13. Dynamic Stability Qualification

v0.6.3 distinguished **multivariate** $D_{\text{D}}$ from **feature-level** attribution.

Example (`sub-11025`): `mean_delta_theta` was stable across runs; `cv_delta_gamma` was not.

Feature-level claims must separate multivariate contribution from run-stable evidence.

---

## 14. Validation

### 14.1 Leave-one-out (TD)

For each TD subject $i$: fit age models and $\Sigma_{\lambda}$ on $\text{TD}\setminus\{i\}$, then score $i$.  
Produces an empirical LOO reference without self-fitting.

### 14.2 Lambda sensitivity

Evaluate the full pipeline over $\lambda \in \{0.01,\ldots,0.50\}$.  
Primary metric: rank stability of subject scores.  
$\lambda = 0.10$ is a working baseline, not a data-mined optimum.

### 14.3 Empirical percentiles

$$P(s) = \text{rank of } s \text{ within LOO TD scores}$$

Descriptive only ($n_{\text{TD}} = 39$); not population probabilities.

### 14.4 Residual profiles

Report for selected subjects/blocks: feature, observed, age-expected, residual, residual SD, $z$, $|z|$.  
Profiles **complement** Mahalanobis distances; they do not replace them.

---

## 15. Canonical Artifacts

```text
results/features/
├── participants_features_subject_v0.3_clean.csv   # SE + QC
├── connectivity_graph_subject_v0.5.csv            # C + G
└── dynamic_subject_v0.6.csv                       # D

results/normative/
├── covariance_{SE,C,G,D}_*.npy
├── loo_*.csv
└── residual profiles

results/validation/
results/figures/
```

v1.0 unified outputs may use `*_v10` suffixes when produced by `scripts/30_nai_v10_unified.py`.

---

## 16. Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `normative/regression.py` | Age models, predictions, residuals (label-agnostic) |
| `normative/covariance.py` | Empirical covariance, shrinkage, condition numbers |
| `normative/mahalanobis.py` | Block-wise Mahalanobis distances |
| `normative/validation.py` | LOO validation, λ-grid sensitivity, empirical percentiles |
| `features/blocks.py` | Single source of truth for SE/C/G/D feature lists |
| `features/aggregation.py` | Run → subject aggregation |
| `nai/composite.py` | Equal-weight NAI composite |
| `nai/profile.py` | Residual profiles and summary tables |
| `statistics/sensitivity.py` | Threshold / window robustness |
| `statistics/stability.py` | Run-level stability analysis |

---

## 17. Script Contract

`scripts/30_nai_v10_unified.py` orchestrates:

```text
load → merge blocks → canonical cohort → TD reference
  → fit SE/C/G/D models → D_B → NAI
  → LOO → λ sensitivity → percentiles → save
```

All mathematics via `src/nai/`.

---

## 18. Reproducibility

Every reported analysis must record: dataset ID, feature-block version, cohort definition, $n_{\text{TD}}$, $n_{\text{ASD}}$, $\lambda$, block dimensions, covariance condition numbers, validation method, and output paths.

---

## 19. Interpretation Rules

1. NAI measures multidimensional **normative deviation**, not diagnosis.  
2. TD defines the reference.  
3. Age correction before covariance.  
4. Regularized covariance is mandatory.  
5. Equal weights are the baseline.  
6. Weights are not optimized on $n_{\text{ASD}} = 2$.  
7. Large $D_B$ alone does not establish an ASD-specific effect.  
8. Feature-level claims need stability evidence.  
9. LOO percentiles are descriptive.  
10. Results remain exploratory pending independent validation.

---

## 20. Implementation Status

Completed through prior stages:

- dataset IO
- preprocessing
- QC
- SE/C/G/D extraction
- subject aggregation
- block-wise normative models
- shrinkage sensitivity
- LOO validation
- residual profiles
- dynamic stability audit

v1.0 unifies these components under one frozen normative and composite scoring contract.

---

## 21. Non-Goals (v1.0)

- Weight optimization on ASD labels  
- Clinical classifier training  
- Diagnostic validity claims  
- New feature families in the frozen core  
- Complexity increases solely to improve separation  
- Feature selection using ASD labels  
- Group inference from $n_{\text{ASD}} = 2$  
- Replacing normative scoring with supervised classification  

---

## 22. Final Model

$$\text{EEG} \longrightarrow \{\text{SE},\,\text{C},\,\text{G},\,\text{D}\} \longrightarrow \text{age correction} \longrightarrow \text{TD residual covariance} \longrightarrow \{D_{\text{SE}},\,D_{\text{C}},\,D_{\text{G}},\,D_{\text{D}}\} \longrightarrow \text{NAI}_{\text{v1.0}}$$

$$\text{NAI}_{\text{v1.0}} = \frac{1}{4} \left( D_{\text{SE}} + D_{\text{C}} + D_{\text{G}} + D_{\text{D}} \right)$$

Validation stack:

$$\text{LOO} + \lambda\text{-sensitivity} + \text{empirical calibration} + \text{residual profiles} + \text{stability analysis}$$

Feature extraction, normative modelling, multivariate scoring, validation, and interpretation remain **separate computational layers**.