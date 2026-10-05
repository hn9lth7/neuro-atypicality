# Statistical Plan — NAI v1.0

## 1. Design

- **Reference population:** TD only ($n=39$)
- **Estimand:** Individual atypicality relative to age-adjusted TD residual distribution
- **Constraint:** No ASD-label-based feature selection or weight learning

---

## 2. Primary Analysis

1. Fit age-linear models per feature on TD.
2. Residual covariance + shrinkage $\lambda=0.10$.
3. Block Mahalanobis distances $D_{SE}, D_C, D_G, D_D$.
4. Composite $NAI = \frac{1}{4}\sum D_B$.
5. **Leave-one-out** NAI for each TD subject.
6. Score ASD subjects with full TD model.
7. Report **empirical ranks** vs. LOO TD distribution.

---

## 3. Sensitivity

- Grid: $\lambda \in \{0.01, 0.05, 0.10, 0.20, 0.30, 0.50\}$
- Rank / qualitative separation of individual profiles across $\lambda$

---

## 4. Interpretation Rules

- Empirical percentiles are **descriptive ranks**, not population probabilities.
- No ASD vs. TD group hypothesis test as a primary claim ($n_{\mathrm{ASD}}=2$).
- Feature-level residual claims require stability evidence when available.

---

## 5. Not Performed in v1.0 as Primary Endpoints

- Supervised classification accuracy as main result
- Multiple-testing correction across hundreds of raw edges
- External cohort validation