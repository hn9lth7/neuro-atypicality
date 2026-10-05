# B6 — Validation Synthesis (NAI v1.0)

**Status:** CLOSED (synthesis)  
**Scope:** Research / validation track only  
**Frozen model:** NAI v1.0 — equal-weight four-block residual Mahalanobis, $\lambda = 0.10$  
**Canonical cohort:** $n = 41$ ($\mathrm{TD} = 39$ normative reference; $\mathrm{ASD} = 2$ scoring only)  

This document does **not** change features, weights, covariance, $\lambda$, or any artifact under `models/nai_v1/`.

---

## 1. Purpose

B6 consolidates the validation branch:

$$\mathrm{B1 
ightarrow B2 
ightarrow B3 
ightarrow B4 
ightarrow B5 
ightarrow B6\ (Synthesis)}$$

It separates what is established on the discovery cohort, what is not established, and what remains open.

---

## 2. What Is Established (Canonical / Discovery)

### 2.1 Implementation Integrity
- Four-block NAI is implemented as a reproducible pipeline with fixed block contracts ($\mathrm{SE}=6$, $\mathrm{C}=8$, $\mathrm{G}=24$, $\mathrm{D}=16$).
- Canonical scoring table (`nai_v10.csv`) is internally consistent:

  $$\mathrm{NAI} = \mathrm{mean}(D_{\mathrm{SE}}, D_{\mathrm{C}}, D_{\mathrm{G}}, D_{\mathrm{D}})$$

  (max absolute residual on the order of machine precision).
- Release audit and product parity work treat the research core as frozen; validation analyses (B4–B5) read frozen distances only and do not refit the model.

### 2.2 Within-Cohort Stability and Uncertainty (B1–B2)

* **B1:** Normative / ranking procedures on the TD reference show substantial rank stability under the documented resampling protocol (reported mean rank stability $\approx 0.844$). This supports internal numerical and ordinal coherence of the index on the discovery TD set, not external generalization.
* **B2:** Uncertainty analyses yield wide intervals for individual scores and related quantities. Point NAI values must be read with that uncertainty; narrow clinical cut-offs are not justified by B2.

### 2.3 Block Structure (B4)
On the full canonical set ($n = 41$):

| Quantity | Approx. Value |
| :--- | :---: |
| **Mean $D_{\mathrm{SE}}, D_{\mathrm{C}}, D_{\mathrm{G}}, D_{\mathrm{D}}$** | $1.45, 1.90, 2.08, 3.39$ |
| **Mean NAI** | $2.21$ |
| **Mean $C_{\mathrm{SE}}, C_{\mathrm{C}}, C_{\mathrm{G}}, C_{\mathrm{D}}$** | $0.16, 0.21, 0.23, 0.39$ |
| **Dominant block by $\max C_B$** | $\mathrm{D}$ for $41/41$ |
| **$\max C_B \ge 0.40$** | $19/41$ |

- **Strongest inter-block rank association:** $D_{\mathrm{C}}$–$D_{\mathrm{G}}$ (Spearman $
ho  pprox 0.92$), consistent with graph metrics derived from connectivity.
- Correlations of each $D_B$ with NAI are high but partly structural (NAI is their average).
- Larger typical magnitude of $D_{\mathrm{D}}$ and universal D-dominance in $C_B$ are compatible with scale / dimensionality / regularization effects; they do not imply that dynamic features are biologically primary or diagnostically decisive.

### 2.4 Heterogeneity of Contribution Profiles (B5)
- **Profile entropy:** $H = -\sum_B C_B \log C_B$, mean $ pprox 1.31$ (upper bound $\ln 4  pprox 1.39$) $
ightarrow$ profiles are mostly multi-block.
- **Concentration of $\max C_B$:** Only $2/41$ with $\max C \ge 0.50$; most subjects in $0.30	ext{--}0.50$.
- **High-NAI subset (top quartile, $\mathrm{NAI} \ge 2.61$, $n = 11$):** Mean $C_{\mathrm{D}}  pprox 0.35$ (lower than cohort mean $ pprox 0.39$), with slightly higher mean shares for SE/C/G $
ightarrow$ high composite scores often reflect combined block elevations, not pure dynamic spikes.

### 2.5 Individual Discovery ASD Scores (Descriptive Only)
Previously reported scoring of the two discovery ASD subjects (e.g. elevated vs. low NAI relative to LOO-TD ranks) remains individual description within the same study that defined the TD norm. It is not group-level evidence and not external validation.

---

## 3. What Is Not Established

| Claim | Status |
| :--- | :---: |
| **External pediatric validation of frozen 54-D EV-A** | **No** (B3-A: no eligible independent cohort under contract) |
| **Clinical diagnostic validity for ASD** | **No** |
| **Sensitivity / specificity / ROC as clinical metrics** | **No** |
| **Inferential ASD vs. TD effects ($n_{\mathrm{ASD}} = 2$)** | **No** |
| **Generalization beyond discovery acquisition / age band** | **No** |
| **Clinical utility or decision thresholds** | **No** |
| **Optimal or data-driven block weights** | **No** (equal weights remain baseline by design) |
| **Causal primacy of the dynamic block** | **No** |

**B3-A Status (Current):**
**NO-GO** — no eligible external cohort available for frozen EV-A (independent pediatric ASD+TD rest, raw access, montage/srate/age compatible with the frozen 54-D contract without retuning). Thailand 2025 remains **PENDING** (access / metadata unresolved). Healthy-only or adult-only datasets do not satisfy the pediatric ASD EV-A contract.

---

## 4. Scientific Position of NAI v1.0 After B1–B6

NAI v1.0 is a frozen exploratory normative framework:

$$\mathrm{NAI} = rac{1}{4} igl(D_{\mathrm{SE}} + D_{\mathrm{C}} + D_{\mathrm{G}} + D_{\mathrm{D}} igr)$$

with age-corrected residuals and regularized covariance estimated on TD, applied for scoring without refitting on labels.

- **Supported:** Reproducible computation; internal TD-oriented stability and uncertainty characterization; descriptive block and profile structure on the canonical cohort.
- **Not supported:** Diagnostic biomarker status; external clinical generalization; group inference from $n = 2	ext{ ASD}$.

**Methodological Conclusion:**  
Further scientific progress that could change the strength of claims requires independent data (and/or a revised contract for a different EV design), not additional parameter tuning or repeated analyses of the same 41 subjects aimed at “confirming” NAI.

---

## 5. Roadmap After B6

| Stage | Focus Area | Status |
| :--- | :--- | :---: |
| **A** | Research core (v1.0) | **CLOSED / FROZEN** |
| **B1–B2** | Stability & uncertainty | **CLOSED** |
| **B3-A** | External EV-A cohort | **NO-GO** (unavailable) |
| **B3** | Thailand / future cohort | **PENDING** if access appears |
| **B4–B5** | Contribution structure | **CLOSED** |
| **B6** | Validation synthesis | **CLOSED** |
| **P0–P4** | Product feature-row path | **CLOSED** |
| **C** | Clinical / supervised ML | **NOT STARTED** (separate track) |

---

## 6. Artifact Index (Validation Track)

| Stage | Primary Artifacts |
| :--- | :--- |
| **B3** | `docs/b3_external_validation_contract.md`, `docs/b3_candidate_registry.md` |
| **B4** | `scripts/validation/block_contribution_b4.py`, `results/validation/b4_*`, `docs/validation/b4_block_contribution.md` |
| **B5** | `scripts/validation/heterogeneity_b5.py`, `results/validation/b5_*`, `docs/validation/b5_heterogeneity.md` |
| **B6** | `docs/validation/b6_validation_synthesis.md` (this file) |

Core frozen scoring remains: `results/normative/nai_v10.csv` and associated covariance / model summary artifacts from the v1.0 freeze.

---

## 7. Closing Statement

Validation branch B1–B6 is complete for the current evidence base.  
NAI v1.0 is frozen as an exploratory four-block normative index with documented internal structure and explicit limits. External pediatric validation and clinical claims are not part of the present result set.