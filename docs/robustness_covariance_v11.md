# Robustness: Covariance Estimator (v1.1)

**Status:** Completed experiment (does not modify NAI v1.0)  
**Script:** `scripts/34_robustness_covariance_v11.py`  
**Outputs:** `results/normative/robustness_v11/`

---

## 1. Question

Is the Neural Atypicality Index (NAI) **rank-stable** under a change of residual covariance estimator, when all other pipeline components are held fixed?

---

## 2. Design

| Component | Fixed as in v1.0 |
| :--- | :--- |
| Cohort | 41 subjects (39 TD + 2 ASD) |
| Feature blocks | $SE = 6, C = 8, G = 24, D = 16$ |
| Age correction | Linear per feature, TD-only fit |
| Distance | Block residual Mahalanobis $D_B$ |
| Composite | $\mathrm{NAI} = \frac{1}{4}(D_{\mathrm{SE}} + D_{\mathrm{C}} + D_{\mathrm{G}} + D_{\mathrm{D}})$ |
| Feature extraction | Frozen tables (no re-extraction) |                  |

**Only change:** Residual covariance estimator.

| Estimator | Description |
|:---|:---|
| **$\lambda$-ridge (v1.0 default)** | $\Sigma_\lambda = (1-\lambda)\Sigma + \lambda rac{\mathrm{tr}(\Sigma)}{p}I$, $\lambda=0.10$ |
| **Ledoit–Wolf** | Data-driven shrinkage (`sklearn.covariance.LedoitWolf`), fit on TD residuals |

ASD subjects never enter covariance or age-model fits.

---

## 3. Results

### 3.1 In-sample block / NAI distances

| Metric | mean TD ($\lambda$) | mean TD (LW) | Spearman $\rho$ | mean $\lvert\Delta\rvert$ |
| :--- | :--- | :--- | :--- | :--- |
| $D_{\mathrm{SE}}$ | 1.428 | 1.459 | 0.999 | 0.036 |
| $D_{\mathrm{C}}$ | 1.907 | 1.922 | 0.999 | 0.016 |
| $D_{\mathrm{G}}$ | 2.074 | 2.111 | 1.000 | 0.037 |
| $D_{\mathrm{D}}$ | 3.359 | 3.091 | 0.971 | 0.324 |
| **NAI** | **2.192** | **2.146** | **0.995** | **0.064** |

### 3.2 Ledoit–Wolf shrinkage intensity

| Block | λ (fixed) | LW shrinkage |
|-------|-----------|--------------|
| SE | 0.10 | 0.082 |
| C | 0.10 | 0.108 |
| G | 0.10 | 0.087 |
| D | 0.10 | **0.379** |

The dynamic block received the strongest automatic shrinkage, consistent with higher dimensionality relative to $n_{	ext{TD}}=39$.

### 3.3 Leave-one-out TD NAI

| Estimator | LOO mean | LOO P95 | Spearman ρ (LOO) |
|-----------|----------|---------|------------------|
| λ = 0.10 | 2.583 | 4.006 | |
| Ledoit–Wolf | 2.438 | 3.763 | **0.991** |

### 3.4 ASD individual scores (descriptive only)

| Subject | NAI (λ) | NAI (LW) | Emp. rank vs LOO TD |
|---------|---------|----------|---------------------|
| sub-11025 | 3.864 | 3.656 | **92.3rd** (both) |
| sub-11038 | 1.242 | 1.133 | **0th** (both) |

Empirical LOO percentiles were **identical** under both estimators.

---

## 4. Interpretation

Subject **rank ordering** of NAI was highly stable under replacement of fixed ridge shrinkage by Ledoit–Wolf covariance estimation. Absolute Mahalanobis magnitudes shifted moderately, most for the dynamic block $D_D$, where LW applied stronger shrinkage.

This supports treating **rank-based** conclusions from NAI v1.0 as robust to this particular covariance choice on the present cohort. It does **not** imply that absolute distance scales are estimator-invariant, nor that results generalize beyond $n_{	ext{TD}}=39$.

---

## 5. Limitations

1. Single cohort ($n_{	ext{TD}}=39$, $n_{	ext{ASD}}=2$).
2. Robustness to covariance only; age model, features, and graph construction unchanged.
3. Not external validation; not a group-level ASD analysis.
4. LW and $\lambda$-ridge are both shrinkage estimators; agreement does not rule out sensitivity to other covariance methods (e.g. MCD).

---

## 6. Conclusion

On the canonical NAI cohort, switching residual covariance from fixed $\lambda=0.10$ to Ledoit–Wolf left NAI rank structure essentially unchanged ($
ho_{\mathrm{NAI}}=0.995$; LOO $
ho=0.991$). Empirical LOO percentiles for the two ASD individuals were unchanged. NAI v1.0 remains the frozen baseline; this experiment is a **v1.1 robustness result**, not a change to the baseline.