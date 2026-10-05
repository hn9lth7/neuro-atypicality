# NAI v1.1 — Parity Status

**Status:** FEATURE + NORMATIVE SCORING = CLOSED  
**Date:** Architecture parity vs. frozen v1.0  
**Scope:** Library implementation in `src/nai/` (no change to NAI equations)

---

## 1. Summary

The v1.1 library implementation reproduces the frozen v1.0 feature extraction, subject-level aggregation, and normative scoring paths on the canonical cohort.

Verified status:

| Layer | Result |
| :--- | :--- |
| Run-level C / G / D / SE | PASS (`sub-10025` run-01; numerical parity) |
| Subject-level SE (mean abs $\rightarrow$ rel/log; mean entropy) | PASS 8/8 vs. `features_v032` |
| Subject-level 54-D vector | PASS 54/54 |
| Block Mahalanobis + equal-weight NAI ($\lambda = 0.10$) | PASS vs. `nai_v10.csv` |

For the canonical **41-subject** scoring set (39 TD + 2 ASD), the maximum absolute difference in NAI was **$4.441 \times 10^{-16}$**, consistent with floating-point numerical precision (not a modelling discrepancy).

---

## 2. What Was Not Changed

- Feature definitions / band limits / PLV / graph / dynamic window policy  
- Age-linear residual model  
- Shrinkage $\lambda = 0.10$  
- Equal block weights $NAI = \frac{1}{4}(D_{SE}+D_C+D_G+D_D)$  
- Canonical cohort exclusions used by frozen v1.0  

---

## 3. Contracts Fixed During v1.1

1. **Dynamics:** Bandpass on full run $\rightarrow$ windows $\rightarrow$ PLV (not bandpass per short window).  
2. **SE Subject Aggregation:** Mean absolute band powers across runs, *then* relative powers and log-ratios; spectral entropy = mean across runs.  
3. **C / G / D Subject Aggregation:** Arithmetic mean of run-level features.

---

## 4. Evidence Artifacts

- `results/features/v11_assembly_sub-10025_run-01.csv`  
- `results/features/v11_se_per_run_sub-10025.csv`  
- `results/normative/scoring_parity_v11.csv`  
- Scripts: `51`–`59` integration / parity suite  

---

## 5. Interpretation

v1.1 is a **structurally organized library implementation of the same frozen v1.0 logic**, not a new mathematical model. Further work (LOO library parity, robustness estimators, external cohorts) belongs to a separate branch and must not silently alter v1.0 formulas or cohort definitions.