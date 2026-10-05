# C0 — Supervised ASD/TD ML Protocol

**Track:** C (separate from frozen NAI v1.0)  
**Not clinical diagnostic validation until independent external cohort**

---

## 1. Scientific Question

Does the frozen 54-dimensional subject-level EEG feature representation carry discriminable information between ASD and TD labels on the development set, under leakage-safe nested cross-validation?

**Secondary (exploratory):** How much is lost under the 54-D $\rightarrow$ 1-D NAI compression for the same labels?

---

## 2. Non-Goals (C Development Phase)

- Clinical diagnostic claims
- Sensitivity/specificity as clinical performance
- Changing NAI weights, $\lambda$, or block definitions
- Using full-sample TD normative artifacts inside CV without refit

---

## 3. Development Dataset

- **Source:** OpenNeuro `ds006780` (expanded subject-level 54-D table)
- **Unit of analysis:** Subject (not run)
- **Expected class counts (verify in C1):** TD $\approx 39$, ASD $\approx 63$, Total $\approx 102$
- **Target:** $y \in \{\mathrm{TD}, \mathrm{ASD}\}$
- **Features primary:** $X \in \mathbb{R}^{n \times 54}$ (SE: 6 + C: 8 + G: 24 + D: 16)

---

## 4. Representations

| ID | Representation | Role |
| :--- | :--- | :--- |
| **R0** | Raw 54-D features | **Primary baseline** |
| **R1** | Age-residual 54-D (TD age model fit **only on training fold TD**) | Optional sensitivity |
| **R2** | Scalar NAI (frozen or fold-local) | Secondary exploratory |

*Primary reported result = **R0**.*

---

## 5. Leakage Rules

- No `fit` / `fit_transform` on full $X$ before outer CV.
- Scaler, imputer, feature selector, classifier: **train fold only**.
- If residuals: age regression coefficients estimated from **train TD only**.
- Frozen `age_models.json` / full-cohort covariance **must not** enter primary R0.
- Subject never appears in both train and test of the same outer fold.

---

## 6. Cross-Validation

- **Outer:** StratifiedKFold, **5 folds**, fixed `random_state`
- **Inner:** StratifiedKFold, **5 folds**, same seed policy
- **Nested:** Hyperparameters selected only on inner CV of the outer train set
- **Final outer metrics:** Aggregate over outer test folds only

---

## 7. Models (First Wave)

1. **Logistic Regression** (class_weight balanced or balanced subsample)
2. **Linear SVC** (or LinearSVM with probability calibration if needed)
3. **Random Forest**

*No XGBoost/LightGBM until R0 baselines are locked.*

---

## 8. Hyperparameter Search (Inner Only)

Document grids in code; selection criterion = **inner mean ROC-AUC** (or balanced accuracy if severe imbalance in a fold — fixed *a priori*: ROC-AUC).

---

## 9. Primary Metrics (Outer Test)

- ROC-AUC
- PR-AUC
- Balanced accuracy
- $F_1$ (positive class = ASD)
- Confusion matrix (summed or mean over folds)

**Reporting language:**  
> *"Cross-validated performance on ds006780 development set"* (not clinical sensitivity/specificity).

---

## 10. Ablation (After R0 Baseline)

- **E0:** SE
- **E1:** C
- **E2:** G
- **E3:** D
- **E4:** SE + C
- **E5:** SE + C + G
- **E6:** Full 54-D

*Descriptive comparison only; not biomarker ranking.*

---

## 11. Interpretability

- **Linear models:** Coefficients (after scaling)
- **Trees:** Permutation importance
- **Aggregate importance:** To blocks SE / C / G / D

*No feedback into NAI v1.0 weights.*

---

## 12. Model Freeze (C3)

After nested CV on development set, freeze:
- Chosen pipeline family
- Hyperparameters
- Preprocessing recipe
- Random seeds

Then **stop tuning** on `ds006780` before external evaluation.

---

## 13. External Validation (C4–C5)

Requires independent pediatric ASD+TD resting EEG meeting a written **ML input contract** (channels, srate, reference, feature definition).  
*Until then, no external AUC claims.*

---

## 14. Relation to NAI

- NAI remains a separate normative research product.
- ML does not redefine NAI.
- NAI-only classifier is an exploratory secondary analysis.