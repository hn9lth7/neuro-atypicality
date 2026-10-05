# NAI v1.1 — Robustness Summary

**Status:** Two controlled sensitivity experiments completed  
**Baseline:** NAI v1.0 frozen (linear age, $\lambda=0.10$, equal-weight blocks)  
**Cohort:** 41 subjects (39 TD + 2 ASD), same frozen feature tables  

**Detailed Reports:**
- `docs/robustness_covariance_v11.md`
- `docs/robustness_age_v11.md`

---

## Design Principle

Each experiment changes **one** modelling layer only:

```text
same cohort → same features → same NAI formula
         │
         ├── (1) residual covariance estimator
         └── (2) age residualization model
```

*ASD subjects never enter TD normative fits.*

---

## Results at a Glance

| Experiment | Perturbation | $\rho_{\mathrm{NAI}}$ | LOO $\rho$ | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Covariance** | $\lambda=0.10$ vs. Ledoit–Wolf | 0.995 | 0.991 | Highest stability; $D_D$ most sensitive ($
ho=0.971$) |
| **Age** | Linear vs. quadratic | 0.983 | 0.989 | Slightly larger rank movement; $D_D$ again sensitive ($
ho=0.969$) |

> Mean absolute NAI change was small in both cases ($ pprox 0.06$).

---

## ASD Individuals (Descriptive Only)

| Subject | v1.0 NAI ($\lambda$, linear) | Cov. LW NAI | Quad. Age NAI | Empirical LOO Rank (v1.0 $\rightarrow$ alt.) |
| :--- | :---: | :---: | :---: | :--- |
| `sub-11025` | 3.864 | 3.656 | 3.939 | 92.3rd $\rightarrow$ 92.3rd (LW); 92.3rd $\rightarrow$ 89.7th (quad) |
| `sub-11038` | 1.242 | 1.133 | 1.427 | 0th in all settings |

Both remain below LOO TD P95 under the v1.0 reference ($ pprox 4.01$). Rank shifts under quadratic age illustrate that absolute distance and empirical percentile are not interchangeable when the reference distribution also changes.

---

## Scientific Reading

1. **Rank-based conclusions** from NAI v1.0 are robust to the tested covariance and age-model alternatives on this cohort.
2. **Absolute scales of $D_B$** and LOO quantiles can shift; do not over-interpret small differences in raw distances across estimator choices.
3. **The dynamic block** is consistently the more sensitive component under both perturbations.
4. **These experiments are internal sensitivity analyses**, not external validation and not group-level ASD inference ($n_{\text{ASD}}=2$).

---

## What Remains the Frozen Baseline

$$NAI_{\mathrm{v1.0}} = \frac{1}{4}(D_{SE} + D_C + D_G + D_D)$$

- Linear age residualization
- Residual covariance with $\lambda = 0.10$
- TD-only normative reference
- LOO TD for empirical calibration

*v1.1 robustness does not replace this default.*

---

## Optional Next Experiments (Not Required for Closing v1.1)

| Candidate | New Question |
| :--- | :--- |
| **ILR / compositional SE** | Does removing sum-to-one constraint change ranks? |
| **Graph density threshold** | Are static graph distances density-dependent? |
| **Joint age $\times$ covariance** | Is there interaction of the two perturbations? |
| **External cohort** | True transfer / calibration |

> **Recommended default after this summary:** Pause new math unless a specific paper section needs one more controlled test.

---

## Artifact Index

```text
results/normative/robustness_v11/          # covariance experiment
results/normative/robustness_age_v11/      # age experiment
scripts/34_robustness_covariance_v11.py
scripts/35_robustness_age_v11.py
docs/robustness_covariance_v11.md
docs/robustness_age_v11.md
docs/robustness_summary_v11.md             # this file
```