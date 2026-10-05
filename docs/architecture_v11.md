# NAI v1.1 Architecture

## Status

**Status:** CLOSED — critical-path architecture and scoring parity confirmed.

`v1.1` is a modular implementation of the frozen `v1.0` NAI mathematical core.
It does not introduce new NAI mathematics, weights, features, or clinical ML.

---

## Critical Path

```text
EEG / BIDS
    ↓
IO + metadata
    ↓
preprocessing + QC
    ↓
spectral / connectivity / graph / dynamics
    ↓
feature blocks
    ↓
subject aggregation
    ↓
age-dependent normative model
    ↓
block Mahalanobis distances
    ↓
NAI composite + residual profile
```

### Core Modules

| Functional Area | Core Module Path |
| :--- | :--- |
| **I/O & Metadata** | `nai.io.bids`<br>`nai.io.metadata` |
| **Preprocessing & QC** | `nai.preprocessing.pipeline`<br>`nai.qc.signal_quality` |
| **Spectral Extraction** | `nai.spectral.power`<br>`nai.spectral.entropy` |
| **Connectivity Processing** | `nai.connectivity.phase`<br>`nai.connectivity.dynamic` |
| **Graph Analysis** | `nai.graph.metrics`<br>`nai.graph.spectral` |
| **Temporal Dynamics** | `nai.dynamics.transitions`<br>`nai.dynamics.windows` |
| **Feature Layer** | `nai.features.blocks`<br>`nai.features.subject_aggregation`<br>`nai.features.extractor` |
| **Normative Modeling** | `nai.normative.regression`<br>`nai.normative.covariance`<br>`nai.normative.mahalanobis`<br>`nai.normative.robustness` |
| **Composite & Profiling** | `nai.nai.composite`<br>`nai.nai.profile` |

---

## Parity Criterion

The canonical 41-subject cohort was rescored through the `v1.1` library API and compared with the frozen `v1.0` output.

### Observed Parity Metrics

| Block / Index | Max Absolute Difference |
| :--- | :--- |
| `D_{SE}` | $2.220 \times 10^{-16}$ |
| `D_{C}` | $2.220 \times 10^{-16}$ |
| `D_{G}` | $4.441 \times 10^{-16}$ |
| `D_{D}` | $4.441 \times 10^{-16}$ |
| **`NAI`** | **$4.441 \times 10^{-16}$** |

```text
SCORING PARITY PASS
```

---

## Out of Scope

The following components are strictly excluded from the `v1.1` critical path:

* Clinical ML / classifiers
* SVM / logistic baselines
* Group-comparison statistics
* Visualization / reporting layer
* New external datasets
* NAI weight optimization
* Covariance / shrinkage retuning
* New NAI mathematics

---

## Legacy Scripts

Older scripts containing local implementations of normative functions remain as historical/legacy research scripts.
They are not part of the `v1.1` scoring API and are not rewritten as part of this architecture freeze.

---

## Scientific Status

`v1.1` does not constitute external clinical validation.

B1/B2 provide internal stability and uncertainty analyses.
An independent pediatric ASD+TD external validation cohort matching the frozen 54-D representation was not established in the current validation cycle.