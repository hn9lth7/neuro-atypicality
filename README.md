# NAI

### Neuro-Atypicality Index

**A four-block normative framework for quantifying multidimensional atypicality in resting-state EEG relative to an age-dependent typically developing (TD) reference.**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-v1.0%20frozen-blue.svg)]()
[![Docker](https://img.shields.io/badge/docker-nichyk2026%2Fnai--api%3A0.1-2496ED.svg)](https://hub.docker.com/r/nichyk2026/nai-api)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Research use only. Not a medical device. Not a clinical diagnostic tool for ASD or any other condition.**

| | |
|---|---|
| **54-D** feature space (SE · C · G · D) | **Age-corrected** residual Mahalanobis blocks |
| Frozen TD norm (OpenNeuro ds006780) | LOO · λ-sensitivity · release audit |
| Scoring API + Docker image | Fully documented methods & limits |

---

## Contents

- [Abstract](#abstract)
- [At a Glance](#at-a-glance)
- [Motivation](#motivation)
- [Contributions](#contributions)
- [Method at a Glance](#method-at-a-glance)
- [Mathematical Formulation](#mathematical-formulation)
- [Feature Blocks](#feature-blocks)
- [Normative Model](#normative-model)
- [Composite Index](#composite-index)
- [Validation](#validation)
- [Selected Results](#selected-results)
- [Documentation Map](#documentation-map)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Scoring](#api-scoring)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Related Work](#related-work)
- [Citation](#citation)
- [License](#license)

---

## Abstract

Clinical EEG pipelines often reduce subjects to a binary label (e.g. ASD vs control). NAI instead maps resting-state EEG to a **quantitative atypicality profile**: how far an individual’s feature vector lies from an **age-dependent TD normative distribution**.

Raw multi-channel EEG is transformed into a **54-dimensional** subject-level vector spanning spectral/entropy, phase connectivity, static graph topology, and dynamic connectivity. On a TD reference cohort, each block is residualised against age, a shrinkage-regularised residual covariance is estimated, and a **Mahalanobis distance** is computed per block. The baseline index is the equal-weight mean of the four block distances.

NAI is an **exploratory normative framework**. It is not trained as an ASD classifier and is not intended for clinical diagnosis. External pilot scoring (Sheffield) shows that optimistic in-sample separation does not automatically transfer under leave-one-out or cross-cohort reference fitting.

---

## At a Glance

| | |
|---|---|
| **Problem** | Binary EEG classifiers obscure *structure* of deviation from typical development. |
| **Idea** | Age-adjusted residual Mahalanobis distances in complementary EEG feature blocks. |
| **Outcome** | Interpretable NAI score + block distances + residual profiles against a frozen TD norm. |

| Classical approach | NAI |
|--------------------|-----|
| ASD vs control labels | Deviation from TD norm |
| Single feature family | SE + connectivity + graph + dynamics |
| Black-box prediction | Block-wise distances + residual profile |
| Often overfits small *n* | Explicit LOO / λ / transfer documentation |

---

## Motivation

Resting-state EEG studies of neurodevelopmental conditions frequently optimise for group separation. With small labelled samples, reported AUCs are fragile and hard to transport across sites.

NAI reframes the target:

> Given a pediatric resting-state recording, how atypical is the subject’s **neurophysiological feature profile** relative to an age-matched TD reference?

This supports research use cases (cohort characterisation, residual profiling, API scoring) without claiming diagnostic validity.

---

## Contributions

- **Four-block feature contract** (54-D): spectral/entropy, PLV connectivity, static graph metrics, sliding-window dynamics  
- **Age-dependent normative model** fitted on TD only (linear residualisation + shrinkage covariance)  
- **Equal-weight composite** $\mathrm{NAI} = \frac14\sum_B D_B$ with frozen $\lambda = 0.10$  
- **Validation layer**: leave-one-out on TD, λ-sensitivity, empirical percentiles, residual profiles, release audit  
- **Product layer**: feature-row validation, model bundle, JSON reports, FastAPI + Docker image  
- **Honest external pilot notes** (Sheffield): in-sample optimism vs LOO / transfer / supervised nested CV  

---

## Method at a Glance

```text
Resting-state EEG
        │
   Preprocessing + QC
        │
   ┌────┴────┬──────────┬──────────┐
   SE        C          G          D
 spectral   PLV       graph    dynamics
        │
   Subject-level aggregation
        │
   TD-only age models  →  residuals r
        │
   Σ_λ = (1−λ)Σ + λI
        │
   D_B = √(rᵀ Σ_λ⁻¹ r)   for B ∈ {SE, C, G, D}
        │
   NAI = (D_SE + D_C + D_G + D_D) / 4
```

---

## Mathematical Formulation

### Feature vector

Each subject is represented by a stacked vector

$$
\mathbf{x}
=
\bigl[
\mathbf{x}_{\mathrm{SE}},\;
\mathbf{x}_{\mathrm{C}},\;
\mathbf{x}_{\mathrm{G}},\;
\mathbf{x}_{\mathrm{D}}
\bigr]
\in \mathbb{R}^{54}.
$$

### Age residualisation (per feature $j$ in block $B$)

On the TD training set:

$$
x_j = \beta_{0j} + \beta_{1j}\,\mathrm{age} + \varepsilon_j.
$$

Residuals for any subject with age $a$:

$$
r_j = x_j - \bigl(\hat\beta_{0j} + \hat\beta_{1j}\,a\bigr).
$$

### Regularised residual covariance

$$
\Sigma_{\lambda}
=
(1-\lambda)\,\widehat{\mathrm{Cov}}(\mathbf{r}_{\mathrm{TD}})
+
\lambda\, I,
\qquad
\lambda = 0.10\ \text{(default)}.
$$

### Block Mahalanobis distance

$$
D_B
=
\sqrt{
\mathbf{r}_B^{\mathsf T}
\Sigma_{B,\lambda}^{-1}
\mathbf{r}_B
}.
$$

### Composite index (v1.0 baseline)

$$
\boxed{
\mathrm{NAI}_{v1.0}
=
\frac{1}{4}
\bigl(
D_{\mathrm{SE}} + D_{\mathrm{C}} + D_{\mathrm{G}} + D_{\mathrm{D}}
\bigr)
}
$$

Weights are equal by design and are **not** optimised on ASD labels.

Formal equations, shrinkage diagnostics, and non-claims:  
**[`docs/mathematical_model.md`](docs/mathematical_model.md)**

---

## Feature Blocks

| Block | Dimension | Contents |
|-------|-----------|----------|
| **SE** | 6 | Relative band powers (θ, α, β), spectral entropy, log power ratios |
| **C** | 8 | PLV mean and median in θ, α, β, γ |
| **G** | 24 | mean degree, degree CV, clustering, global efficiency, mean path length, Laplacian entropy × 4 bands |
| **D** | 16 | mean/CV of PLV transition magnitude; mean/temporal CV of degree heterogeneity × 4 bands |

Feature names and contracts: **[`config/features.yaml`](config/features.yaml)**, **[`src/nai/features/blocks.py`](src/nai/features/blocks.py)**, **[`models/nai_v1/feature_schema.json`](models/nai_v1/feature_schema.json)**

Preprocessing defaults (band-pass 1–45 Hz, notch, average reference):  
**[`config/preprocessing.yaml`](config/preprocessing.yaml)**, **[`src/nai/preprocessing/pipeline.py`](src/nai/preprocessing/pipeline.py)**

---

## Normative Model

| Item | Value |
|------|--------|
| Reference data | OpenNeuro [ds006780](https://openneuro.org/datasets/ds006780) |
| Fit population | TD only (**n = 39**) |
| ASD on discovery | **n = 2** (scored only; not used in fit) |
| Age model | Linear in age, feature-wise |
| Shrinkage | $\lambda = 0.10$ |
| Age scoring window | 8.0–13.0 years (out-of-range → warning) |

Frozen artifacts: **[`models/nai_v1/`](models/nai_v1/)** (`covariance_*.npy`, `age_models.json`, `manifest.json`)

Architecture and contracts: **[`docs/architecture.md`](docs/architecture.md)**

---

## Composite Index

$$
\mathrm{NAI}
=
\frac{D_{\mathrm{SE}}+D_{\mathrm{C}}+D_{\mathrm{G}}+D_{\mathrm{D}}}{4}
$$

Optional relative block contributions (descriptive, not a new model):

$$
C_B = \frac{D_B}{D_{\mathrm{SE}}+D_{\mathrm{C}}+D_{\mathrm{G}}+D_{\mathrm{D}}}.
$$

Implementation: **[`src/nai/nai/`](src/nai/nai/)**, **[`src/nai/product/score.py`](src/nai/product/score.py)**

Product non-claims: **[`docs/product/p0_product_scope.md`](docs/product/p0_product_scope.md)**

---

## Validation

| Layer | Role |
|-------|------|
| Leave-one-out (TD) | Out-of-sample block distances and NAI for the reference population |
| λ-grid | Sensitivity of ranks/scores to shrinkage |
| Empirical percentiles | Rank of held-out or external scores vs LOO TD |
| Residual profiles | Which features drive $D_B$ |
| Release audit | Schema, PD covariances, formula consistency, cohort integrity |

Protocol: **[`docs/validation_protocol.md`](docs/validation_protocol.md)**  
Statistical plan: **[`docs/statistical_plan.md`](docs/statistical_plan.md)**

---

## Selected Results

**Discovery (canonical clean cohort)**

| Quantity | Approx. value |
|----------|----------------|
| LOO TD NAI mean | 2.58 |
| LOO TD NAI P95 / P99 | 4.01 / 4.52 |
| Release audit | 70 PASS / 0 FAIL |

**External pilot (Sheffield resting cohort, n = 46)**

| Experiment | AUC (approx.) | Reading |
|------------|---------------|---------|
| NAI in-sample (CTRL in fit) | 0.85 | Optimistic |
| NAI exact LOO | 0.49 | Chance |
| Discovery TD → Sheffield | 0.38 | Domain shift |
| Supervised nested CV (54-D) | ~0.60 | Weak / unstable |

Full narrative: **[`docs/results_v10.md`](docs/results_v10.md)**, **[`docs/ml/`](docs/ml/)**

---

## Documentation Map

| Path | Description |
|------|-------------|
| [`docs/architecture.md`](docs/architecture.md) | System design, data flow, block dimensions, non-goals |
| [`docs/mathematical_model.md`](docs/mathematical_model.md) | Formal math for residuals, $\Sigma_\lambda$, $D_B$, NAI |
| [`docs/methods_v10.md`](docs/methods_v10.md) | Methods narrative for v1.0 |
| [`docs/results_v10.md`](docs/results_v10.md) | Frozen results, LOO, contributions, limitations |
| [`docs/validation_protocol.md`](docs/validation_protocol.md) | LOO, λ, calibration rules |
| [`docs/statistical_plan.md`](docs/statistical_plan.md) | Analysis plan and reporting rules |
| [`docs/dataset.md`](docs/dataset.md) | Discovery dataset notes |
| [`docs/feature_definition.md`](docs/feature_definition.md) | Feature definitions |
| [`docs/preprocessing_protocol.md`](docs/preprocessing_protocol.md) | Preprocessing protocol |
| [`docs/research_question.md`](docs/research_question.md) | Research question |
| [`docs/product/p0_product_scope.md`](docs/product/p0_product_scope.md) | Product claims and non-claims |
| [`docs/ml/`](docs/ml/) | Sheffield / ML pilots (C4–C6, nested CV) |

---

## Repository Structure

```text
neuro-atypicality/
├── api/                     # FastAPI: CSV features → NAI JSON
├── config/                  # preprocessing.yaml, features.yaml, default.yaml
├── models/nai_v1/            # Frozen normative bundle
├── src/nai/
│   ├── preprocessing/       # Minimal EEG pipeline
│   ├── spectral/            # Power, entropy
│   ├── connectivity/        # PLV / phase
│   ├── graph/               # Graph metrics
│   ├── dynamics/            # Windowed transitions
│   ├── features/            # Block contracts, aggregation
│   ├── normative/           # Age models, covariance, Mahalanobis, LOO helpers
│   ├── nai/                 # Composite index, profiles
│   └── product/             # validate_input, score_row, reports, model_bundle
├── scripts/                 # Extraction, normative runs, audits, ML pilots
├── docs/                    # Architecture, math, methods, product, ml/
├── results/                 # Features, scores, figures (partial)
└── tests/
```

---

## Installation

```bash
git clone https://github.com/<YOU>/neuro-atypicality.git
cd neuro-atypicality
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
source .venv/bin/activate
pip install -r requirements.txt
```

For the HTTP API:

```bash
pip install fastapi "uvicorn[standard]" python-multipart
```

---

## Quick Start

**Library scoring** (subject already reduced to 54-D + `age`):

```python
from pathlib import Path
from nai.product.model_bundle import load_bundle
from nai.product.score import score_row
from nai.product.validate_input import validate_feature_row

bundle = load_bundle(Path("models/nai_v1"))
vr = validate_feature_row(row)   # mapping / Series
if vr.ok:
    scores = score_row(row, bundle)
    # scores: D_SE, D_C, D_G, D_D, NAI
```

**Local API:**

```bash
uvicorn api.app:app --reload --port 8000
```

---

## API Scoring

Docker:

```bash
docker pull nichyk2026/nai-api:0.1
docker run --rm -p 8000:8000 \
  -e NAI_API_KEYS=dev-key-change-me \
  nichyk2026/nai-api:0.1
```

```bash
curl -s http://localhost:8000/v1/health

curl -s -X POST http://localhost:8000/v1/score \
  -H "X-API-Key: dev-key-change-me" \
  -F "file=@path/to/features_54d.csv"
```

CSV must include `age` and the 54 feature columns defined in the schema.  
Responses include RUO disclaimer, block distances, and NAI.

---

## Reproducibility

```bash
python scripts/30_nai_v10_unified.py
python scripts/31_release_audit_v10.py
python scripts/32_nai_analysis_v10.py
```

Parity target: frozen tables under `results/normative/` and bundle under `models/nai_v1/`.

---

## Limitations

- Single discovery dataset, protocol, and age band  
- Discovery $n_{\mathrm{ASD}} = 2$ → no group-level ASD inference  
- Equal weights are a baseline, not label-optimised  
- Cross-site resting EEG may exhibit strong domain shift  
- LOO percentiles are empirical ranks within $n_{\mathrm{TD}} = 39$, not population probabilities  

---

## Related Work

Separate repository on spectral graph optimisation:  
[SSPES — Spectral Subspace Predictor for Edge Switching](https://github.com/hn9lth7/spectral-subspace-predictor)

---

## Citation

NAI v1.0 is an exploratory normative atypicality framework for resting-state EEG. It is **not** a validated clinical diagnostic biomarker.

Please cite this repository and the OpenNeuro dataset **ds006780** when using the code or model bundle.

---

## License

MIT (unless otherwise noted).  
Third-party datasets remain under their original terms.