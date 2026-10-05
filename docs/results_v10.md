**Model:**
$$
\mathrm{NAI}_{v1.0} = \frac{1}{4}(D_{\mathrm{SE}}+D_{\mathrm{C}}+D_{\mathrm{G}}+D_{\mathrm{D}}),\quad \lambda = 0.10
$$
(TD-only normative fit)

---

## 1. Cohort

| Item | Value |
|:---|:---|
| Canonical subjects | 41 |
| TD (normative reference) | 39 |
| ASD (scored only) | 2 |
| Excluded (artifact) | `sub-10777` |

Release audit: **70 PASS / 0 FAIL / 0 WARN**.

---

## 2. Normative Reference

- Age-linear residual model per feature, TD only
- Regularized residual covariance per block
- Leave-one-out TD distribution of NAI used for descriptive ranking

### LOO TD NAI Summary

| Statistic | Value |
|:---|:---|
| Mean | 2.583 |
| P95 | 4.006 |
| P99 | 4.523 |

Figures: `results/figures/analysis_v10/nai_loo_distribution.png`, `nai_loo_ecdf.png`.

---

## 3. Individual ASD Profiles

| Subject | QC Flag | $D_{SE}$ | $D_C$ | $D_G$ | $D_D$ | **NAI** | Empirical Rank vs LOO TD |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `sub-11025` | `very_low_alpha` | 3.349 | 2.866 | 3.454 | 5.788 | **3.864** | $ pprox$ **92.3rd** percentile |
| `sub-11038` | `ok` | 0.541 | 0.848 | 1.034 | 2.545 | **1.242** | $ pprox$ **0th** rank |

**Interpretation (Strict):**

- `sub-11025` shows an **elevated** multidimensional score relative to the observed LOO TD distribution but **does not exceed** the empirical TD 95th percentile (4.006).
- `sub-11038` lies at the **low end** of the observed LOO TD ranks.
- These are **individual normative profiles**, not group evidence for ASD.

---

## 4. Block Contribution Analysis

Definition (decomposition only, not a new model):

$$C_B = rac{D_B}{D_{SE}+D_C+D_G+D_D},\qquad \sum_B C_B = 1$$

| Subject | $C_{SE}$ | $C_C$ | $C_G$ | $C_D$ |
|:---|:---:|:---:|:---:|:---:|
| `sub-11025` | 0.217 | 0.185 | 0.223 | **0.374** |
| `sub-11038` | 0.109 | 0.171 | 0.208 | **0.512** |

For both individuals, the **dynamic block** has the largest relative contribution. For `sub-11038` this occurs at a **low** total NAI; therefore high $C_D$ must not be read as an ASD biomarker.

Figures: `nai_block_contribution_11025.png`, `nai_block_contribution_11038.png`.

---

## 5. $\lambda$-Sensitivity

| $\lambda$ | TD Mean NAI | `sub-11025` | `sub-11038` |
|:---|:---:|:---:|:---:|
| 0.01 | 2.522 | 5.270 | 1.629 |
| 0.05 | 2.288 | 4.263 | 1.371 |
| 0.10 | 2.192 | 3.864 | 1.242 |
| 0.20 | 2.121 | 3.548 | 1.125 |
| 0.30 | 2.108 | 3.430 | 1.076 |
| 0.50 | 2.167 | 3.419 | 1.058 |

Absolute scores depend on shrinkage; **reference $\lambda = 0.10$** remains the frozen default. Relative separation of the two ASD profiles is preserved across the grid.

Figure: `nai_lambda_sensitivity.png`.

---

## 6. Residual Profiles

Age-corrected residual $z$-profiles ($SE$ and $D$ blocks) are stored in `residual_profiles_v10.csv` and corresponding figures under `analysis_v10/`.

Dynamic stability audit (prior stage): for `sub-11025`, `mean_delta_theta` was consistent across runs; `cv_delta_gamma` was run-sensitive. Feature-level claims remain qualified by that audit.

---

## 7. Limitations

1. $n_{	ext{ASD}}=2$ — no ASD–TD group inference
2. Single dataset / protocol / age band
3. Resting-state only
4. No external validation
5. QC flag on `sub-11025` (`very_low_alpha`)
6. Dense PLV graphs; full threshold re-extraction not required for freeze
7. Equal weights are baseline, not optimized

---

## 8. Conclusion

NAI v1.0 provides a **reproducible four-block normative atypicality score** with LOO calibration, regularization sensitivity, block decomposition, and residual profiling. It is an **exploratory computational framework**, not a clinical diagnostic model.

---

## 9. Methods Summary

Resting-state EEG from `ds006780` was preprocessed (1–45 Hz band-pass, 60 Hz notch, average reference, 64 EEG channels). Subject-level features were extracted in four blocks: spectral–entropy (6), PLV connectivity (8), static graph metrics (24), and sliding-window dynamic features (16; 10 s windows, 5 s step).  

An age-linear model was fit per feature on TD subjects only ($n=39$). Residual covariances were shrinkage-regularized ($\lambda=0.10$). Block atypicality was defined as the Mahalanobis distance of the residual vector. The Neural Atypicality Index was the equal-weight mean of the four block distances.  

TD subjects were scored with leave-one-out normative fits; ASD subjects ($n=2$) were scored with the full TD model. Empirical ranks used the LOO TD NAI distribution. Sensitivity analyses varied $\lambda \in \{0.01,\ldots,0.50\}$. Block contributions $D_B/\sum D_B$ and residual $z$-profiles were reported for interpretation only.