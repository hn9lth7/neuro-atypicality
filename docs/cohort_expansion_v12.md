# NAI v1.2 — Cohort Expansion

**Status:** Exploratory extension of frozen NAI v1.0  
**Does not modify:** `nai_v10.csv`, LOO v1.0 artifacts, or the mathematical definition of NAI.

---

## 1. Purpose

NAI v1.0 was limited by incomplete resting-state data availability for ASD participants on disk ($n_{	ext{ASD}} = 2$). After downloading missing OpenNeuro `ds006780` Restingstate BDF files, feature extraction and QC were repeated on the expanded set.

v1.2 answers:

> Under the **same frozen normative definition** as v1.0, how do multidimensional atypicality scores distribute when the ASD scoring sample is expanded to all available QC-clean resting-state subjects?

No new feature families, weight optimization, or diagnostic classifier are introduced.

---

## 2. Cohort

| Role | Group | $n$ | Notes |
|:---|:---|:---:|:---|
| Normative fit | TD | 39 | QC-clean; same role as v1.0 |
| Scoring | ASD | 63 | QC-clean; not used in covariance fit |
| Excluded | SIB | 0 | Intentionally excluded |
| Excluded | NaN group / extreme artifact / missing metadata | per QC | See subject-level QC flags |

**Total scored subjects after merge:** 102 (39 TD + 63 ASD).

**Source:** OpenNeuro `ds006780`, eyes-open resting-state EEG (BioSemi, 64 channels, 512 Hz).

### Age Distribution

| Group | $n$ | Mean | SD | Min | Max |
|:---|:---:|:---:|:---:|:---:|:---:|
| TD | 39 | 10.55 | 1.79 | 8.0 | 12.9 |
| ASD | 63 | 10.74 | 1.46 | 8.0 | **14.5** |

**Limitation:** One ASD participant is older than the TD maximum (14.5 vs 12.9). Linear age correction may extrapolate outside the TD support for that individual.

Approximate age bins (years):

| Bin | ASD | TD |
|:---|:---:|:---:|
| 8–<9 | 9 | 11 |
| 9–<10 | 9 | 5 |
| 10–<11 | 17 | 3 |
| 11–<12 | 14 | 8 |
| 12–<13 | 12 | 12 |
| 13–<14 | 1 | 0 |

TD coverage is thinner in the 10–11 year range.

---

## 3. Frozen Methodology (Unchanged from v1.0)

```text
Raw resting-state EEG
  │
  ▼
Preprocessing + QC
  │
  ▼
Feature blocks SE (6), C (8), G (24), D (16)
  │
  ▼
Subject-level aggregation
  │
  ▼
TD-only linear age correction
  │
  ▼
Residual covariance with shrinkage λ = 0.10
  │
  ▼
Block Mahalanobis distances D_SE, D_C, D_G, D_D
  │
  ▼
NAI = mean(D_SE, D_C, D_G, D_D)
```

- **No ASD observations** enter age regression or residual covariance.
- **Equal weights**; not optimized on ASD labels.
- Feature definitions and block membership are those of NAI v1.0.

### Artifacts

| File | Role |
|:---|:---|
| `results/normative/nai_v10.csv` | **Frozen** v1.0 baseline (untouched) |
| `results/normative/nai_v12_expanded.csv` | Expanded in-sample scores |
| `results/normative/loo_nai_v12_td.csv` | LOO TD NAI scores |
| `results/normative/asd_scores_vs_loo_v12.csv` | ASD vs LOO thresholds |
| `results/normative/loo_summary_v12.json` | Summary metrics |

---

## 4. Descriptive Group Statistics (Full TD Fit)

| Group | $n$ | Mean NAI | Median | SD | Min | Max |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| TD | 39 | 2.199 | 2.164 | 0.559 | 1.209 | 3.340 |
| ASD | 63 | 3.157 | 2.647 | 2.158 | 1.176 | 16.428 |

ASD shows a **higher median** and a **heavier upper tail**, with **substantial overlap** with TD.

### Extreme Individual Profiles (QC Review, Not Automatic Exclusion)

| Subject | NAI | Dominant Pattern (Descriptive) |
|:---|:---:|:---|
| `sub-11936` | 16.43 | All blocks elevated; C/G extreme; very low alpha_rel |
| `sub-2713` | 9.91 | Especially D / G |
| `sub-11244` | 5.91 | Isolated high $D_D$ |

Amplitude / line-noise metrics for these subjects did **not** show catastrophic SE-QC failure comparable to the previously excluded extreme artifact case. They remain labeled **extreme NAI / QC review required**, not confirmed preprocessing failure, pending optional run-level connectivity audit.

---

## 5. Leave-One-Out TD Calibration (Primary Threshold Analysis)

For each of the 39 TD subjects:

1. Fit age models and regularized residual covariances on the remaining 38 TD.
2. Score the held-out TD LOO NAI.
3. Build the empirical LOO TD distribution.

ASD subjects are scored once with the **full 39-TD** normative model and compared to LOO thresholds (not to in-sample TD scores).

### LOO TD NAI Summary

| Metric | Value |
|:---|---:|
| Mean | 2.591 |
| Median | 2.463 |
| Max | 4.811 |
| **P95** | **4.040** |
| **P99** | **4.596** |

### ASD vs LOO TD

| Metric | Value |
|:---|---:|
| Median NAI | 2.647 |
| Median empirical rank vs LOO TD | 61.5th |
| **Above LOO P95** | **7 / 63 (11.1%)** |
| **Above LOO P99** | **6 / 63 (9.5%)** |

### Sensitivity (Excluding `sub-11936` and `sub-2713`)

| Metric | Value |
|:---|---:|
| Above LOO P95 | **5 / 61 (8.2%)** |

The elevated upper tail is **not** solely driven by the two most extreme scores, but the fraction above LOO P95 remains a **minority**.

### Contrast with In-Sample TD Percentiles

Comparing ASD to **in-sample** TD NAI percentiles (full fit) gave ~22/63 ($ pprox 35\%$) above TD P95. That figure is **optimistic** relative to LOO-based thresholds and should **not** be used as the primary reporting metric for v1.2.

---

## 6. Interpretation

1. Expanding the ASD scoring sample from 2 to 63 subjects enables **distributional** description of NAI under the frozen model.
2. Under LOO TD calibration, most ASD scores fall within the bulk of the TD LOO distribution (median rank $ pprox$ 62nd percentile).
3. A **minority** of ASD individuals (~11%) exceed the LOO P95 threshold; ~9.5% exceed LOO P99.
4. Overlap with TD remains large; NAI is **not** a binary separator of groups in this dataset.
5. Individual residual / block profiles remain the preferred scientific product for extreme cases.

---

## 7. Limitations

- Single dataset / site / resting-state protocol.
- Small TD normative sample ($n = 39$) for high-dimensional blocks (especially $G$, $p = 24$).
- Age support mismatch (ASD up to 14.5 years).
- In-sample vs LOO distinction must be respected in reporting.
- Extreme NAI cases need optional run-level connectivity/dynamic audit before physiological claims.
- **No external validation** in v1.2.
- **Not a clinical diagnostic biomarker**; no sensitivity/specificity is claimed as clinical performance.

---

## 8. Scientific Status Statement

NAI v1.2 is an exploratory cohort expansion of the frozen four-block normative atypicality framework (v1.0). Using leave-one-out TD calibration, a minority of ASD participants exceeded empirical LOO high percentiles, with substantial overlap between groups. These results describe multidimensional deviation scores under a fixed normative definition and do **not** establish ASD-specific diagnostic validity.

---

## 9. Recommended Next Steps (Outside This Freeze)

1. Optional run-level connectivity/graph audit for `sub-11936` (and related extremes).
2. Robustness: covariance estimator / age model sensitivity on the expanded cohort.
3. External validation on an independent EEG cohort.
4. Do **not** silently alter v1.0 or v1.2 formulas after documentation freeze.