# NAI v1.0 — Validation Protocol

**Status:** Validation documentation for Architecture v1.0 freeze  
**Date:** 2026-09-20  
**Related:** `docs/architecture.md`

---

## 1. Purpose

This document records how NAI v1.0 scores are validated. Validation supports the claim that the four-block composite is a **computationally stable normative atypicality measure** relative to a TD reference. It does **not** establish clinical diagnostic validity.

---

## 2. Model Under Validation

$$NAI_{\mathrm{v1.0}} = \frac{1}{4}\bigl(D_{SE}+D_C+D_G+D_D\bigr)$$

Each $D_B$ is an age-adjusted residual Mahalanobis distance with shrinkage $\lambda = 0.10$, estimated on TD only.

| Block | Dimension | Domain |
| :--- | :---: | :--- |
| SE | 6 | Spectral / entropy |
| C | 8 | PLV connectivity |
| G | 24 | Static graph metrics |
| D | 16 | Dynamic transition / degree-heterogeneity |

---

## 3. Canonical Cohort

| Item | Value |
| :--- | :--- |
| Total clean subjects | 41 |
| TD (normative fit) | 39 |
| ASD (scoring only) | 2 |
| Excluded | `sub-10777` (extreme spectral artifact) |
| Retained with QC flag | `sub-11025` (`very_low_alpha`) |
| Unit of analysis | Subject (runs aggregated) |

No feature selection or weight learning uses ASD labels.

---

## 4. Validation Components

### 4.1 Leave-One-Out (TD)

For each TD subject $i$:

1. Fit age models and residual covariance on $\mathrm{TD} \setminus \{i\}$.
2. Apply shrinkage $\lambda = 0.10$.
3. Score $D_{SE}, D_C, D_G, D_D$ and $NAI$ for subject $i$.

Observed LOO summary ($\lambda = 0.10$):

| Quantity | Mean | Median | Max | P95 | P99 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $D_{SE}$ | 1.55 | 1.35 | 3.34 | — | — |
| $D_C$ | 2.13 | 2.02 | 3.87 | — | — |
| $D_G$ | 2.34 | 2.28 | 4.91 | — | — |
| $D_D$ | 4.32 | 4.13 | 10.48 | — | — |
| **NAI** | **2.58** | **2.46** | **4.70** | **4.01** | **4.52** |

### 4.2 Full-Model Scoring of ASD

ASD subjects are scored with the full TD normative model (they never enter the fit).

| Subject | $D_{SE}$ | $D_C$ | $D_G$ | $D_D$ | NAI | Empirical Rank vs. LOO TD NAI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sub-11025` | 3.35 | 2.87 | 3.45 | 5.79 | **3.86** | **92.3rd** |
| `sub-11038` | 0.54 | 0.85 | 1.03 | 2.55 | **1.24** | **0.0th** |

**Interpretation:** `sub-11025` is elevated relative to the observed LOO TD distribution but **does not exceed** the empirical TD 95th percentile (4.01).

### 4.3 $\lambda$-Sensitivity

Grid: $\lambda \in \{0.01, 0.05, 0.10, 0.20, 0.30, 0.50\}$.

Primary checks:
- Rank stability of subject scores across $\lambda$.
- Qualitative ordering of the two ASD profiles (`sub-11025` > bulk TD > `sub-11038`) preserved across the grid.

Default working value: $\lambda = 0.10$ (not optimized on ASD).

### 4.4 Covariance Diagnostics

At $\lambda = 0.10$ (TD residual covariance):

| Block | Rank | Cond (raw) | Cond (reg) |
| :--- | :---: | :---: | :---: |
| SE | 6/6 | $\approx 3.5 \times 10^3$ | $\approx 41$ |
| C | 8/8 | $\approx 5.0 \times 10^3$ | $\approx 45$ |
| G | 24/24 | $\approx 3.0 \times 10^8$ | $\approx 131$ |
| D | 16/16 | $\approx 2.2 \times 10^2$ | $\approx 41$ |

All blocks full rank after shrinkage; G remains the least well-conditioned but usable under regularization.

### 4.5 Residual Profiles

Age-corrected residual z-profiles are used for **interpretation only**.

For `sub-11025` (dynamic block), the largest marginal drivers were `cv_delta_gamma` and `mean_delta_theta`, with stronger aggregate deviation in the PLV-transition sub-block than in degree-heterogeneity.

### 4.6 Run-Level Stability (Dynamic)

| Feature | Run Consistency (`sub-11025`) | Claim Level |
| :--- | :--- | :--- |
| `mean_delta_theta` | Stable across both runs | Supported subject-level contributor |
| `cv_delta_gamma` | Driven mainly by one run | Multivariate contributor only; not a stable trait claim |

High-$D_D$ TD subjects also showed heterogeneous run-level gamma transition CV. Feature-level claims require stability evidence.

---

## 5. Interpretation Rules

1. NAI measures multidimensional deviation from a TD age-dependent reference — **not** a diagnostic label.
2. Empirical percentiles are **descriptive** ranks within $n_{\mathrm{TD}}=39$ LOO scores, not population probabilities.
3. $n_{\mathrm{ASD}}=2$ precludes ASD-vs-TD group inference.
4. QC flags (e.g., `very_low_alpha`) must accompany any individual high-NAI narrative.
5. Weights remain equal; they are not tuned on ASD scores.
6. Feature attribution is allowed only where run-level stability supports it.

---

## 6. Non-Claims (v1.0)

v1.0 does **not** claim:
- Clinical diagnostic performance;
- ASD-specific biomarkers;
- Optimal weights or thresholds;
- Generalization beyond the present resting-state cohort and age range;
- That a single feature (e.g., gamma transition CV) is a stable individual marker without run-level support.

---

## 7. Artifact Index & Freeze Statement

```text
results/normative/
  nai_v10.csv
  loo_nai_v10.csv
  model_summary_v10.json
  covariance_{SE,C,G,D}_v10.npy

results/figures/
  validation_*
  profiles_*
  stability_*
```

**Freeze Statement:** NAI v1.0 is frozen as an exploratory four-block normative atypicality framework with documented LOO calibration, regularization sensitivity, residual interpretation, and run-level stability qualification. Further clinical claims require independent samples and a separate validation study.