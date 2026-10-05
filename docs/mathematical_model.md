# NAI v1.0 Mathematical Model

**Status:** Specification freeze (aligned with Architecture v1.0 and Validation Protocol)  
**Date:** 2026-09-20  
**Scope:** Resting-state EEG normative atypicality framework on OpenNeuro `ds006780`

---

## 1. Scientific Objective

Define a scalar **Neural Atypicality Index** ($	ext{NAI}_{	ext{v1.0}}$) that quantifies how an individual resting-state EEG feature profile deviates from an age-dependent typically developing (TD) reference, using a **four-block** residual Mahalanobis construction.

NAI is an **exploratory computational measure**, not a clinical diagnostic score.

---

## 2. Feature Space

Each subject is represented by four disjoint feature blocks:

$$\mathbf{x} =  igl[ \mathbf{x}_{SE},\; \mathbf{x}_{C},\; \mathbf{x}_{G},\; \mathbf{x}_{D}  igr]$$

| Block | Symbol | Dimension $p_B$ | Content |
|:---|:---:|:---:|:---|
| Spectral / entropy | $SE$ | 6 | Relative band powers, spectral entropy, log power ratios |
| Connectivity | $C$ | 8 | PLV mean and median across four bands |
| Static graph | $G$ | 24 | Graph metrics $	imes$ four bands |
| Dynamic | $D$ | 16 | Transition and degree-heterogeneity dynamics $	imes$ four bands |

Bands: $\{	heta,  lpha,  eta, \gamma\}$.

Feature names are fixed in `src/nai/features/blocks.py` (single source of truth).

---

## 3. Canonical Cohort

- Normative reference: **TD only**, $n_{	ext{TD}}=39$
- Scored sample: clean cohort $n=41$ (39 TD + 2 ASD)
- Excluded: extreme artifact subject(s) (e.g. `sub-10777`)
- Unit of analysis: **subject** (runs aggregated), not individual runs

ASD labels are **not** used to fit age models, residual covariances, or composite weights.

---

## 4. Age-Dependent Location Model

For each feature $j$ in block $B$, fit on TD:

$$x_j =  eta_{0j} +  eta_{1j}\, a +  arepsilon_j$$

where $a$ is age (years).

Predicted value at age $a$:

$$\hat{x}_j(a) = \hat{ eta}_{0j} + \hat{ eta}_{1j}\, a$$

Age-corrected residual:

$$r_j = x_j - \hat{x}_j(a)$$

Residual vector of block $B$:

$$\mathbf{r}_B \in \mathbb{R}^{p_B}$$

Age correction is performed **before** covariance estimation.

---

## 5. Residual Covariance

Let $R_{	ext{TD},B}$ be the matrix of TD residuals for block $B$.

Empirical covariance:

$$\Sigma_B = \operatorname{Cov}(R_{	ext{TD},B})$$

Shrinkage toward a scaled identity:

$$\Sigma_{B,\lambda} = (1-\lambda)\,\Sigma_B + \lambda\, rac{\operatorname{tr}(\Sigma_B)}{p_B}\,I_{p_B}$$

Default:

$$\lambda = 0.10$$

Sensitivity grid used in validation:

$$\lambda \in \{0.01,\,0.05,\,0.10,\,0.20,\,0.30,\,0.50\}$$

---

## 6. Block Mahalanobis Distance

$$D_B = \sqrt{ \mathbf{r}_B^{	op} \Sigma_{B,\lambda}^{-1} \mathbf{r}_B }$$

for $B \in \{SE, C, G, D\}$.

Properties:
- $D_B \ge 0$
- Incorporates feature correlations within the block
- Is **not** a sum of independent univariate $|z|$-scores

---

## 7. Neural Atypicality Index (v1.0)

Equal-weight composite:

$$ oxed{	ext{NAI}_{	ext{v1.0}} =  rac{1}{4}  igl( D_{SE} + D_C + D_G + D_D  igr)}$$

Equivalently:

$$	ext{NAI}_{	ext{v1.0}} = \sum_{B \in \{SE,C,G,D\}} w_B D_B ,\qquad w_B = 	frac{1}{4}$$

Weights are a **baseline definition**. They are not optimized on ASD scores.

---

## 8. Leave-One-Out (LOO) Scoring for TD

For each TD subject $i = 1,\ldots,n_{	ext{TD}}$:

1. Fit age models on TD $\setminus\{i\}$
2. Estimate residual covariance on TD $\setminus\{i\}$
3. Apply shrinkage $\lambda$
4. Compute $D_{SE}^{(i)}, D_C^{(i)}, D_G^{(i)}, D_D^{(i)}$ and $	ext{NAI}^{(i)}$

This yields an empirical reference distribution of out-of-sample TD atypicality scores.

ASD subjects are always scored with the **full** TD model (they never enter the fit).

---

## 9. Empirical Rank Calibration

For an observed score $s$ (e.g. NAI of an ASD subject):

$$P(s) = 	ext{empirical percentile of } s 	ext{ relative to the LOO TD score set}$$

Interpretation:
- Descriptive rank within the observed LOO TD sample ($n_{	ext{TD}}=39$)
- **Not** a population probability
- **Not** a diagnostic posterior probability of ASD

Reference LOO NAI quantiles at $\lambda=0.10$ (from frozen run):

| Quantile | Value |
|:---|:---:|
| Mean | $ pprox 2.58$ |
| P95 | $ pprox 4.01$ |
| P99 | $ pprox 4.52$ |

Example (frozen scores):

| Subject | NAI | Empirical Rank vs LOO TD |
|:---|:---:|:---|
| `sub-11025` | 3.864 | $ pprox$ 92.3rd percentile ($< 	ext{P95}$) |
| `sub-11038` | 1.242 | $ pprox$ 0th empirical rank |

---

## 10. Residual Profile (Interpretation Layer)

For feature $j$:

$$z_j =  rac{r_j}{\sqrt{(\Sigma_{B,\lambda})_{jj}}}$$

Profiles support **interpretation** of which features contribute to $D_B$. They do not replace the multivariate distance $D_B$.

Feature-level claims require **run-level stability** evidence when repeated recordings exist (documented for the dynamic block).

---

## 11. Dynamic Substructure (Interpretive Only)

Block $D$ can be conceptually partitioned as:
- $D_\Delta$: PLV transition dynamics (`mean_delta`, `cv_delta`)
- $D_H$: Degree-heterogeneity dynamics (`mean_degree_cv`, `temporal_cv_degree_cv`)

This partition is for residual analysis only and does **not** change the v1.0 composite formula.

---

## 12. What is Original vs Classical

| Component | Status |
|:---|:---|
| Linear age regression | Classical |
| Mahalanobis distance | Classical |
| Covariance shrinkage | Classical |
| PLV, graph metrics, windowed dynamics | Established EEG methodology |
| **Four-block residual construction + equal-weight NAI + LOO calibration protocol** | **Project-specific framework** |

Scientific contribution is the **architecture and protocol**, not a claim of a new elementary distance.

---

## 13. Explicit Non-Claims

v1.0 does **not** claim:
- Clinical diagnostic validity
- ASD-specific biomarkers
- Optimal or learned block weights
- Generalization beyond the present dataset, age range, and resting-state protocol
- That empirical percentiles are population probabilities

With $n_{	ext{ASD}}=2$, results are individual normative profiles relative to the TD model.

---

## 14. Implementation Mapping

| Mathematical Object | Code Location |
|:---|:---|
| Feature contracts | `src/nai/features/blocks.py` |
| Age models / residuals | `src/nai/normative/regression.py` |
| Covariance + shrinkage | `src/nai/normative/covariance.py` |
| Mahalanobis $D_B$ | `src/nai/normative/mahalanobis.py` |
| LOO / $\lambda$ helpers | `src/nai/normative/validation.py` |
| $	ext{NAI} = \mathrm{mean}(D_B)$ | `src/nai/nai/composite.py` |
| Residual $z$-profiles | `src/nai/nai/profile.py` |
| Orchestration | `scripts/30_nai_v10_unified.py` |
| Consistency audit | `scripts/31_release_audit_v10.py` |

---

## 15. Frozen formula summary

\[
\begin{aligned}
\mathbf{r}_B &= \mathbf{x}_B - \hat{\mathbf{x}}_B(a) \\
\Sigma_{B,\lambda}
&= (1-\lambda)\Sigma_B + \lambda \frac{\mathrm{tr}(\Sigma_B)}{p_B} I \\
D_B
&= \sqrt{\mathbf{r}_B^{\top}\Sigma_{B,\lambda}^{-1}\mathbf{r}_B} \\
NAI_{v1.0}
&= \frac{1}{4}(D_{SE}+D_C+D_G+D_D)
\end{aligned}
\]

with \(\lambda = 0.10\) by default, TD-only normative fit, and LOO calibration for descriptive ranking.