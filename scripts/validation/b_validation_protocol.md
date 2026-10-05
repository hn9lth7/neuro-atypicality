# B — Scientific Validation Protocol

**Status:** ACTIVE (post A-core freeze)  
**Does not modify:** NAI v1.0 equations, λ = 0.10, equal weights, 54-D block definitions, subject aggregation contracts, canonical discovery cohort rules.

## Scope

Validate the *frozen* normative NAI framework under resampling, run stability, and (later) independent data — without retuning the index to ASD labels.

## Frozen reference (immutable)

| Element | Specification |
|--------|----------------|
| Feature vector | 54-D: SE(6) + C(8) + G(24) + D(16) |
| Preprocessing | as in frozen pipeline (notch 60 Hz, band-pass 1–45 Hz, average ref, EEG pick) |
| SE subject agg | mean abs powers → rel + log; entropy = mean over runs |
| C/G/D subject agg | mean of run-level features |
| Normative fit | **TD only** |
| Age model | linear: \(x_j = \beta_{0j} + \beta_{1j}\,age + \varepsilon_j\) |
| Residuals | \(r = x - \hat x_{\mathrm{TD}}(age)\) |
| Covariance | empirical + shrinkage \(\lambda = 0.10\) |
| Block distance | \(D_B = \sqrt{r_B^\top \Sigma_{B,\lambda}^{-1} r_B}\) |
| Composite | \(NAI = (D_{SE}+D_C+D_G+D_D)/4\) |
| Discovery cohort | canonical clean set used for v1.0 (39 TD + 2 ASD scoring) |

No changes to the above after looking at validation results.

## What is *not* allowed in B

- Optimizing weights or λ on ASD labels  
- Changing feature definitions after seeing external results  
- Mixing external subjects into the discovery TD covariance  
- Claiming diagnostic sensitivity/specificity as primary B endpoints  
- Interpreting n_ASD = 2 on discovery set as group-level ASD evidence  

## B pipeline

B0  This protocol (freeze rules)
B1  Internal robustness (incl. bootstrap)
B2  Uncertainty / stability reporting
B3  External validation (EV-A first, EV-B later)
B4  Association / block contribution
B5  Heterogeneity (profiles; exploratory)
B6  Consolidated validation report


Clinical ML (level C) only after B report.

## Primary metrics (B)

1. **Rank stability** of NAI under TD bootstrap / leave-out (Spearman ρ with frozen NAI)  
2. **Interval stability**: width of subject-level bootstrap quantiles for NAI and \(D_B\)  
3. **Run-level consistency** (subjects with ≥2 runs): within-subject dispersion of run-derived scores where applicable  
4. **External EV-A** (when data available): same *definitions*, TD normative fit **on external TD only**, score held-out external subjects  

Secondary / exploratory: block profiles, age association, sensitivity tables already run in A (λ-grid, etc.) — document, do not retune.

## External validation modes

**EV-A (first):**  
External TD → fit age + \(\Sigma_\lambda\) → score external subjects with **frozen feature pipeline**.  
Tests transfer of *concept*, not of ds006780 numerical μ/Σ.

**EV-B (later):**  
Score external subjects under **ds006780 TD** normative model.  
Only after EV-A is technically feasible; document montage/reference/age support mismatches as limitations.

**GO for a candidate external dataset**

- Resting (or clearly documented eyes-open resting-like) EEG  
- Documented age (preferably overlapping ~8–13 or explicit age model plan)  
- Usable channel set mappable to analysis pipeline  
- Clear group labels if association is claimed  
- Legal/ethical access to raw or preprocessed data  

**NO-GO:** unresolved provenance, non-comparable task, no usable TD reference, or acquisition QC failure that dominates signal.

## GO / NO-GO for continuing after B1–B2

- **GO:** bootstrap rank correlation with frozen NAI remains high on discovery TD; no numerical breakdown; uncertainties documented  
- **NO-GO for stronger claims:** if NAI ranks collapse under mild TD resampling → stop external association claims until causes are understood  

## Deliverables

| ID | Artifact |
|----|----------|
| B0 | This protocol |
| B1 | Bootstrap normative stability tables/figures |
| B2 | Subject-level CI / stability summary |
| B3 | External QC + EV-A report (when data exist) |
| B6 | `docs/validation/b_validation_report.md` |

## Relation to A

A proved: library ≡ frozen math on discovery features.  
B asks: how stable is that score, and does the *framework* transfer under independent data and uncertainty quantification — **without changing the formula**.