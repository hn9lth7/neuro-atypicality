# C3 — Development Pipeline Freeze

**Status:** FROZEN specification for Track C development phase  
*Does not claim clinical readiness*

After C2, the specification below is locked for any future **external** evaluation. Hyperparameter or feature searching on `ds006780` to artificially raise mean ROC is explicitly out of scope.

---

## 1. Locked Items

| Item | Value |
| :--- | :--- |
| **Input Representation** | R0 raw 54-D feature vector |
| **Feature Definitions** | Frozen block contract (SE/C/G/D as in C0/C1) |
| **Development Cohort** | `ds006780` subject-level ($n=102$, ASD$=63$, TD$=39$) |
| **Exclusions** | None (post C1 PASS) |
| **Outer CV** | StratifiedKFold, 5 splits, `random_state=42` |
| **Inner CV** | StratifiedKFold, 5 splits, selection metric = ROC-AUC |
| **Model Families** | Logistic Regression, Linear SVM, Random Forest |
| **Preprocessing** | `StandardScaler` for LR/SVM (train fold only); RF no scaling |
| **Class Weight** | `balanced` |
| **Primary Metric** | Outer mean ROC-AUC (with fold std) |
| **Secondary Metrics** | PR-AUC, balanced accuracy, F1 (ASD) |
| **Dataset Role** | Development only |

---

## 2. Policy & Boundaries

### Explicitly Not Frozen as Clinical Product
- A single "winning" coefficient vector or RF for deployment.
- Decision thresholds for diagnosis.
- Clinical sensitivity/specificity claims.
- Any modification to NAI v1.0.

### Allowed / Disallowed Actions

```text
ALLOWED:
  ├── C6 block ablation on the same development set (exploratory)
  ├── C7 NAI-only exploratory classifier (secondary)
  └── C4–C5 independent cohort under a written ML input contract

DISALLOWED:
  ├── Iterative feature engineering / grid expansion to maximize ROC on n=102
  └── Silent changes to the 54-D contract for the frozen benchmark