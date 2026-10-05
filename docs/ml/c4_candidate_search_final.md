# C4 — External Cohort Candidates: Final Status

## 1. Audited Candidates Overview

| Dataset | C4-A | C4-B | Verdict |
| :--- | :---: | :---: | :--- |
| **HBN R9** (`ds005514`) | BLOCKED | N/A | **NO-GO** |
| **Sheffield ORDA** (Dickinson 2022) | PASS | PASS | **RESULT AVAILABLE** |
| **SheffieldAutismBiomarkers CSV** | PASS | FAIL | **NO-GO** |
| **BCIAUT-P300** (NEMAR) | N/A | FAIL | **NO-GO** |
| **KAU** | N/A | FAIL | **NO-GO** |
| **ds005234** (MEG) | N/A | FAIL | **NO-GO** |
| **Thailand Mahidol 2025** | PENDING | FAIL (19ch) | **NO-GO** |

---

## 2. Sheffield ORDA Headline Findings

- **Cohort:** $n = 46$ (28 ASD + 18 CTRL)
- **Primary Metrics:** $	ext{AUC} = 0.849$, $95\%\text{ CI } [0.720, 0.953]$, Cohen's $d = 1.22$, Permutation $p < 10^{-4}$
- **Age-Adjusted AUC:** $0.863$
- **Harmonization Audit:** **CLOSED** ($50	ext{ Hz}$ notch, CAR, 64-ch, $512	ext{ Hz}$)
- **Limitations:** Duration mismatch vs. discovery cohort
- **LOO Stability:** Sensitivity only; exact LOO pending
- **Feature Block Behavior:** D-block dominance (exploratory)

---

## 3. Final Decision & Strategic Path

> **Sheffield ORDA Verdict:** **RESULT AVAILABLE** — Pilot negative-to-weak performance after C5/C6/3a evaluation. Not framed as external diagnostic validation.

### Project-Level Next Steps
1. Document freeze across all Track C outputs.
2. Either secure additional dataset access (NDA/LEAP/ERP) or publish/write up the negative transfer findings.
3. **Strict rule:** No further hyperparameter tuning on $n = 46$.