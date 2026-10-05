# Robustness: Age Model (v1.1)

**Status:** Completed experiment (does not modify NAI v1.0)  
**Script:** `scripts/35_robustness_age_v11.py`  
**Outputs:** `results/normative/robustness_age_v11/`

---

## 1. Question

Is the Neural Atypicality Index (NAI) **rank-stable** under a change of the age residualization model, when features, covariance, and the composite formula are held fixed?

---

## 2. Design

| Component | Fixed as in v1.0 |
| :--- | :--- |
| Cohort | 41 subjects (39 TD + 2 ASD) |
| Feature blocks | SE=6, C=8, G=24, D=16 |
| Residual covariance | Ridge shrinkage, $\lambda=0.10$ |
| Distance | Block residual Mahalanobis $D_B$ |
| Composite | $NAI = \frac{1}{4}(D_{SE}+D_C+D_G+D_D)$ |
| Feature tables | Frozen (no re-extraction) |

**Only change:** age model used for residualization.

| Model | Form (per feature $j$) |
| :--- | :--- |
| **Linear (v1.0 default)** | $x_j = \beta_0 + \beta_1 \,\mathrm{age} + \varepsilon$ |
| **Quadratic** | $x_j = \beta_0 + \beta_1 \,\mathrm{age} + \beta_2 \,\mathrm{age}^2 + \varepsilon$ |

Age models and covariance are fit on **TD only**. ASD subjects are scored only.

This experiment tests **sensitivity**, not whether quadratic is a superior predictive model.

---

## 3. Results

### 3.1 Rank stability (all subjects)

| Metric | Spearman rho | Mean abs(delta) | Max abs(delta) |
|---|---:|---:|---:|
| D_SE | 0.974 | 0.099 | 0.292 |
| D_C | 0.989 | 0.064 | 0.213 |
| D_G | 0.986 | 0.061 | 0.233 |
| D_D | 0.969 | 0.118 | 0.424 |
| **NAI** | **0.983** | **0.063** | **0.185** |

### 3.2 Leave-one-out TD NAI

| Age model | LOO mean | LOO P95 | Spearman $\rho$ (LOO) |
| :--- | :---: | :---: | :---: |
| Linear | 2.583 | 4.006 | — |
| Quadratic | 2.662 | 4.034 | **0.989** |

### 3.3 ASD individuals (descriptive)

| Subject | NAI (linear) | NAI (quadratic) | Emp. rank vs LOO TD (lin) | Emp. rank vs LOO TD (quad) |
| :--- | :---: | :---: | :---: | :---: |
| `sub-11025` | 3.864 | 3.939 | 92.3rd | **89.7th** |
| `sub-11038` | 1.242 | 1.427 | 0th | 0th |

For `sub-11025`, absolute NAI increased under quadratic residualization, while the empirical LOO percentile **decreased** slightly (92.3rd $\rightarrow$ 89.7th). Both values remain below the corresponding LOO TD 95th percentile ($\approx 4.0$). This illustrates that raw distance and rank relative to the LOO reference are not interchangeable.

---

## 4. Comparison with covariance robustness

| Experiment | $\rho_{\mathrm{NAI}}$ | LOO $\rho$ |
| :--- | :---: | :---: |
| Covariance: $\lambda=0.10$ vs Ledoit–Wolf | 0.995 | 0.991 |
| Age: linear vs quadratic | 0.983 | 0.989 |

Age-model perturbation produced **somewhat larger** rank movement than the covariance-estimator perturbation, but rank structure remained high in both cases. The dynamic block was again among the more sensitive components ($\rho_{D_D}=0.969$).

---

## 5. Interpretation

Quadratic age residualization preserves high subject-level rank stability of NAI on this cohort and age band (approximately 8–13 years). Absolute block distances and LOO reference quantiles shift modestly. Individual empirical percentiles can move by a small number of ranks even when absolute NAI increases.

**Not claimed:** quadratic is better than linear; age nonlinearity is required; results generalize beyond $n_{\mathrm{TD}}=39$.

**Supported claim:** linear age correction, as used in frozen NAI v1.0, yields ranks that are robust to a simple nonlinear alternative under the present design.

---

## 6. Limitations

1. Narrow age range; quadratic terms may matter more over wider developmental spans.
2. $n_{\mathrm{TD}}=39$: quadratic fits are underpowered for strong claims about curvature.
3. $n_{\mathrm{ASD}}=2$: ASD rows are individual case descriptions only.
4. Covariance fixed at $\lambda=0.10$; joint age $\times$ covariance sensitivity not tested here.
5. Sensitivity experiment only—not model selection by ASD separation.

---

## 7. Conclusion

Replacing the linear age model with a quadratic age model preserved high subject-level rank stability of NAI ($\rho=0.983$; LOO $\rho=0.989$), while producing modest changes in absolute distances and empirical reference percentiles. The dynamic block showed the greatest rank sensitivity ($\rho=0.969$), followed by the spectral block ($\rho=0.974$). For the two ASD individuals, the empirical LOO percentile of `sub-11025` changed from the 92.3rd to the 89.7th, whereas `sub-11038` remained at the bottom of the empirical reference distribution.

Linear age residualization remains the **v1.0 default**. This document records a **v1.1 robustness result**, not a change to the frozen baseline.

---

## 8. Artifacts

```text
results/normative/robustness_age_v11/
  scores_linear.csv
  scores_quadratic.csv
  rank_correlations.csv
  comparison_summary.json
```