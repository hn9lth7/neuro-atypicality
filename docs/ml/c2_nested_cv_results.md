# C2 ‚Äî Nested CV Baseline (Development Set)

**Status:** CLOSED  
**Track:** C (supervised ASD/TD ML)  
*Not clinical validation*

---

## 1. Dataset Parameters

- **Source File:** `results/features/features_54d_development_c1b.csv`
- **Sample Size:** $n = 102$ (ASD $= 63$, TD $= 39$)
- **Features:** R0 raw 54-D (SE 6 + C 8 + G 24 + D 16)
- **Unit of Analysis:** Subject-level
- **Role:** **Development only** (`ds006780`)

---

## 2. Evaluation Protocol

- **Outer CV:** 5-fold StratifiedKFold (`random_state=42`)
- **Inner CV:** 5-fold StratifiedKFold (hyperparameter selection by ROC-AUC)
- **Preprocessing:** `StandardScaler` fitted on outer-train fold only (LR, Linear SVM); RF no scaling
- **Models:** Logistic Regression, Linear SVM, Random Forest (`class_weight="balanced"`)
- **Artifacts:** `results/ml/c2_outer_fold_metrics.csv`, `c2_summary.json`, `c2_confusion_sum.csv`

---

## 3. Performance Results (Outer-Fold Mean $\pm$ Std)

| Model | ROC-AUC | PR-AUC | Balanced Accuracy | F1 (ASD) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.666 $\pm$ 0.100 | 0.780 $\pm$ 0.068 | 0.588 $\pm$ 0.088 | 0.647 $\pm$ 0.112 |
| **Linear SVM** | 0.667 $\pm$ 0.098 | 0.775 $\pm$ 0.066 | 0.564 $\pm$ 0.080 | 0.599 $\pm$ 0.167 |
| **Random Forest** | 0.658 $\pm$ 0.083 | 0.771 $\pm$ 0.053 | 0.559 $\pm$ 0.120 | 0.604 $\pm$ 0.191 |

*Note:* Outer-fold ROC-AUC ranged approximately $0.52\text{--}0.82$ across folds and models.

---

## 4. Key Interpretations & Scope Limits

- **Discrimination:** Weak-to-moderate discrimination on the development set ($	ext{mean ROC} \approx 0.66\text{--}0.67$). High fold-to-fold variability; point estimates remain unstable at $n = 102$.
- **Model Complexity:** Linear models are comparable to Random Forest, showing no evidence that a complex non-linear decision boundary is required.
- **Precision-Recall:** PR-AUC is above the ASD prevalence baseline ($ pprox 0.62$), but far from ceiling performance.

### Non-Claims
- Not clinical sensitivity or specificity.
- Not external validation.
- Not justification to modify NAI v1.0 weights or $\lambda$.
- Not a frozen diagnostic product.