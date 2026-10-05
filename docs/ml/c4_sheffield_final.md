# C4 — Sheffield ORDA (Dickinson, Jeste & Milne 2022)

**Status:** CLOSED as pilot external probe — **negative-to-weak** for diagnostic transfer.  
*Not external validation of NAI as an ASD classifier.*

---

## 1. Cohort Summary

- **Sample Size:** $n = 46$ (28 ASD + 18 CTRL)
- **Paradigm & Age:** Eyes-closed resting-state; ages $\approx 1\text{--}28\text{ y}$
- **Harmonization:** 64-ch BioSemi 10–20, $512\text{ Hz}$, notch $50\text{ Hz}$, CAR
- **Duration:** $\approx 160\text{ s}$ (vs. $\approx 63\text{ s}$ discovery resting runs)

---

## 2. Experiment Summary

| Experiment | AUC | Interpretation |
| :--- | :---: | :--- |
| **NAI In-sample** (fit on all Sheffield CTRL) | **0.849** | Optimistic (CTRL in reference) |
| **NAI Exact LOO** (C5) | **0.492** | Chance performance |
| **NAI Discovery TD $\rightarrow$ Sheffield** (C6) | **0.383** | Domain shift ($	ext{NAI}_{	ext{CTRL}} > 	ext{NAI}_{	ext{ASD}}$) |
| **Supervised Nested CV** (54-D + age) (3a) | **$\approx 0.60$** | Weak, unstable across folds |

---

## 3. Harmonization Verification

Verified parameters: $512\text{ Hz}$, 64 EEG channels, `preprocess_minimal(..., notch_freqs=50.0)`, average reference (CAR).  
*Note:* API default $60\text{ Hz}$ was **not** used for Sheffield extraction.

---

## 4. Detailed Methodological Breakdowns

### C5 — Exact LOO
- **Protocol:** CTRL $i$ fitted on $\mathrm{CTRL} \setminus \{i\}$; ASD $i$ fitted on all CTRL.
- **Metrics:** AUC $0.492$; mean NAI ASD $2.96$, CTRL $3.21$.
- **In-sample Baseline:** AUC $0.849$ (ASD $2.96$, CTRL $2.04$).
- **Conclusion:** $0.849$ is in-sample optimism, not out-of-sample performance.

### C6 — Discovery TD $\rightarrow$ Sheffield
- **Fit:** Discovery TD only ($n = 39$, age $8.0\text{--}12.9$).
- **Score:** All Sheffield subjects (no Sheffield data in fit).
- **Metrics:** AUC $0.383$; mean NAI ASD $4.76$, CTRL $5.41$.
- **Conclusion:** Both groups lie far from the discovery reference $\rightarrow$ severe site/protocol shift.

### 3a — Supervised Nested CV
- **Features:** 54-D + age ($5 \times$ outer / $3 \times$ inner stratified CV).
- **Logistic Regression:** OOF AUC **0.601**, balanced accuracy $0.58$.
- **Linear SVM:** OOF AUC **0.603**, balanced accuracy $0.52$.
- **Stability:** Fold AUC range $\approx 0.40\text{--}0.75 \rightarrow$ highly unstable pilot.

---

## 5. Scientific Wording Guidelines

- **Allowed:** Pilot multi-check on one external resting cohort showed that in-sample NAI separation did not survive exact LOO, discovery-reference scoring, or supervised nested CV.
- **Not Allowed:** "NAI externally validated for ASD diagnosis"; "AUC 0.85 confirms diagnostic utility."

---

## 6. Artifacts & Implications

### Generated Artifacts
- `results/ml/c5_sheffield_exact_loo_*.csv/json`
- `results/ml/c6_sheffield_from_discovery_*.csv/json`
- `results/ml/c3a_sheffield_nested_cv_*.csv/json`
- `docs/ml/c4_sheffield_provenance.md`

### Implication for NAI v1.0
NAI remains a **normative atypicality framework** on discovery data. Sheffield does **not** support NAI (or 54-D linear ML) as a transferable resting-EEG ASD diagnostic on this sample size and protocol mismatch.