# NAI v1.2 — Robustness Audit

**Status:** Completed exploratory robustness package for cohort-expanded NAI v1.2  
**Does not modify** NAI formula, feature definitions, `nai_v10.csv`, or primary `nai_v12_expanded.csv` scores.

---

## 1. Purpose

Assess whether the main descriptive conclusions of NAI v1.2 depend critically on:

- Residual covariance regularization;
- Age-correction specification;
- A small number of extreme individual profiles;
- Subjects with very few resting-state runs.

This is **sensitivity analysis**, not clinical validation and not external validation.

---

## 2. Frozen Baseline

| Item | Definition |
| :--- | :--- |
| Formula | $NAI = \frac{1}{4}(D_{SE}+D_C+D_G+D_D)$ |
| Blocks | $SE=6$, $C=8$, $G=24$, $D=16$ |
| Shrinkage | $\lambda = 0.10$ (baseline) |
| Age model | Linear (TD-only) |
| Cohort | 39 TD (fit) + 63 ASD (score) |
| LOO TD P95 / P99 | **4.040** / **4.596** |
| Primary ASD exceedance | **7/63 (11.1%)** > LOO P95; **6/63 (9.5%)** > LOO P99 |

**Artifacts:** `results/normative/robustness_v12/`

---

## 3. Covariance Sensitivity

NAI was recomputed for $\lambda \in \{0.01, 0.05, 0.10, 0.20, 0.30, 0.50\}$ and compared to Ledoit–Wolf residual covariance (same linear age models).

| Estimator | Spearman vs. $\lambda=0.10$ (All) | TD | ASD |
| :--- | :---: | :---: | :---: |
| $\lambda=0.01$ | 0.982 | 0.966 | 0.984 |
| $\lambda=0.05$ | 0.997 | 0.995 | 0.997 |
| $\lambda=0.10$ | 1.000 | 1.000 | 1.000 |
| $\lambda=0.20$ | 0.995 | 0.992 | 0.997 |
| $\lambda=0.30$ | 0.988 | 0.985 | 0.993 |
| $\lambda=0.50$ | 0.969 | 0.963 | 0.977 |
| Ledoit–Wolf | **0.994** | — | — |

**Interpretation:** Subject ranking is highly stable across shrinkage strength and Ledoit–Wolf. Absolute mean NAI shifts with $\lambda$ (less shrinkage $\rightarrow$ larger distances), but rank-based conclusions are not driven by the specific choice $\lambda=0.10$.

---

## 4. Age-Model Sensitivity

Linear vs. quadratic TD-only age regression; $\lambda=0.10$ fixed.

| Cohort | Spearman NAI (Linear $\leftrightarrow$ Quadratic) |
| :--- | :---: |
| All | **0.986** |
| TD | 0.983 |
| ASD | 0.983 |

### 4.1 Age Extrapolation

TD age support: **8.0–12.9** years. ASD includes subjects beyond this range:

| Subject | Age | NAI Linear | NAI Quadratic |
| :--- | :---: | :---: | :---: |
| `sub-11325` | **14.5** | 2.92 | **4.39** |
| `sub-11547` | 13.0 | 2.65 | 2.70 |

`sub-11325` shows a large rank shift under quadratic age correction and is annotated as `age_extrapolation`. It is **not** removed from the main scoring table; sensitivity analyses may exclude ages > TD max.

**Interpretation:** For most of the cohort, linear age correction is adequate. Scores for subjects outside TD age support are model-sensitive and should be interpreted with caution.

---

## 5. Extreme-Profile Sensitivity

LOO thresholds held fixed. No refit of the normative model.

| Analysis | $n$ ASD | Median NAI | > LOO P95 | > LOO P99 |
| :--- | :---: | :---: | :---: | :---: |
| All ASD | 63 | 2.647 | **7 (11.1%)** | 6 (9.5%) |
| − `sub-11936` | 62 | 2.634 | 6 (9.7%) | 5 (8.1%) |
| − `sub-2713` | 62 | 2.634 | 6 (9.7%) | 5 (8.1%) |
| − Both | 61 | 2.622 | **5 (8.2%)** | 4 (6.6%) |
| − Both − Age-extrapolated | 60 | 2.612 | **5 (8.3%)** | 4 (6.7%) |

**Interpretation:** The LOO high-tail exceedance is **not** solely attributable to the two most extreme NAI profiles. After their removal, ~8% of remaining ASD subjects still exceed LOO P95.

Extreme cases remain labeled **QC review required** (not confirmed preprocessing failure) based on subject-level amplitude / line-noise metrics; optional run-level connectivity audit is out of scope for this robustness freeze.

---

## 6. Run-Count Sensitivity

Stratification of **frozen** NAI scores by number of resting-state runs used in subject-level aggregation. Normative models were **not** refit within run-count strata.

| Cohort | $n$ | Median NAI | > LOO P95 |
| :--- | :---: | :---: | :---: |
| ASD, 1 run | **2** | 4.258 | 1/2 |
| ASD, $\ge 2$ runs | 61 | 2.622 | 6 (9.8%) |
| ASD, $\ge 3$ runs | 54 | 2.592 | 5 (9.3%) |
| ASD, $\ge 5$ runs | 48 | 2.612 | 4 (8.3%) |
| TD, $\ge 2$ runs | 39 | 2.164 | 0 |

Single-run ASD subjects:

| Subject | Age | NAI | Notes |
| :--- | :---: | :---: | :--- |
| `sub-11562` | 12.0 | 5.426 | Above LOO P99 |
| `sub-11759` | 11.6 | 3.090 | Below LOO P95 |

**Interpretation:** Single-run ASD subjects show a higher descriptive NAI distribution, but the subgroup is **very small ($n = 2$)**, so **no inference** about run-count effects is warranted. Subjects with `n_runs = 1` remain in the main cohort with a `few_runs` / `single_run` annotation for sensitivity reporting. Restricting to $\ge 2$ or $\ge 3$ runs yields LOO P95 exceedance rates of ~9–10%, close to the full-sample 11.1%.

---

## 7. Integrated Interpretation

Across covariance shrinkage, Ledoit–Wolf estimation, linear vs. quadratic age correction, extreme-profile exclusion, and run-count stratification:

1. **Rank stability** of NAI is high for covariance and age-model variants in the bulk of the sample.
2. **LOO-based high-tail exceedance** remains in a minority of ASD subjects (~8–11% above LOO P95 under primary and sensitivity definitions).
3. **Substantial overlap** with the TD LOO distribution is preserved.
4. Documented caveats: age extrapolation (`sub-11325`), extreme multimodal profiles (`sub-11936`, `sub-2713`), and very small single-run ASD subgroup.

Recommended summary language:

> The descriptive NAI distribution and high-tail exceedance estimates were broadly stable across covariance shrinkage, age-model specification, extreme-profile exclusion, and run-count sensitivity analyses.

Avoid: “validated”, “clinically robust”, “diagnostically robust”.

---

## 8. Limitations

- Single site / dataset (ds006780); no external cohort in this audit.
- TD normative sample still small ($n = 39$), especially relative to the 24-dimensional graph block.
- LOO calibrates TD thresholds only; ASD scores use the full TD fit (intentional, not LOO-for-ASD).
- Extreme connectivity/graph drivers for `sub-11936` not fully resolved at run level in this package.
- Single-run analyses underpowered ($n = 2$).

---

## 9. Scientific Status

NAI v1.2 robustness checks support treating the expanded-cohort descriptive results and LOO threshold analysis as **internally consistent under reasonable analytic alternatives**, within one dataset. They do **not** establish generalizability, clinical utility, or ASD-specific biomarker validity.

---

## 10. Freeze and Next Step

```text
v1.0 frozen baseline
  → v1.2 cohort expansion (39 TD + 63 ASD)
  → LOO TD calibration
  → robustness audit (covariance, age, extremes, runs)  ← COMPLETE
  → FREEZE v1.2 documentation
  → external validation on an independent EEG cohort
```