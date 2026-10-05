# B1–B2 Internal Normative Stability and Uncertainty

**Status:** COMPLETE  
**Core unchanged:** NAI v1.0 equations, λ = 0.10, equal weights, 54-D features, TD-only fit.

## B1 — TD bootstrap ranking stability

- Cohort scored: 41 subjects (39 TD + 2 ASD)
- Fit sample: TD only, resampled with replacement
- Replicates: B = 1000, seed = 42, λ = 0.10
- Point NAI vs frozen `nai_v10.csv`: max |Δ| = 4.44×10⁻¹⁶

**Rank stability (Spearman ρ of bootstrap NAI vs frozen full-TD NAI):**

| Statistic | Value |
|-----------|-------|
| Mean ρ | 0.8437 |
| Median ρ | 0.8500 |
| 95% interval of ρ | [0.7181, 0.9274] |

Interpretation: individual **ranking** of NAI is sensitive to which TD subjects enter the normative reference. This is expected with n_TD = 39 and high-dimensional blocks (especially G). It is **not** evidence of numerical failure of the index.

## B2 — Subject-level uncertainty

From the same bootstrap samples:

| Metric | Value |
|--------|-------|
| Median 95% CI width (all) | 1.5165 |
| Median 95% CI width (TD) | 1.5177 |
| Median 95% CI width (ASD) | 1.0127 |
| Median \|bias\| (boot mean − frozen) | 0.2776 |
| Max \|bias\| | 0.7154 |
| Frozen NAI inside bootstrap 95% CI | 41/41 (100%) |

**Discovery ASD:**

| Subject | Frozen NAI | 95% CI | Width | Bias |
|---------|------------|--------|-------|------|
| sub-11025 | 3.864 | [3.639, 5.156] | 1.517 | +0.420 |
| sub-11038 | 1.180 | [1.153, 1.662] | 0.509 | +0.195 |

The **widest** intervals were observed among **TD** subjects with high point NAI (e.g. sub-10212 width ≈ 4.13), not among ASD. Interval width is a property of score + reference configuration, not an “ASD-only” effect.

Bias is defined as E_bootstrap[NAI] − NAI_frozen. It is **not** a classical measurement-error SD; it describes systematic shift of the bootstrap mean relative to the full-TD point estimate.

## Combined interpretation (report wording)

Bootstrap resampling of the 39-subject TD normative reference preserved numerical validity of the frozen NAI but showed non-negligible dependence of individual ranking on normative-sample composition (median Spearman ρ = 0.850, 95% interval 0.718–0.927). Subject-level bootstrap 95% intervals had median width ≈ 1.52 NAI units; the frozen score lay inside the interval for all 41 subjects. These results support explicit uncertainty reporting and caution against treating individual NAI ranks as invariant to the TD reference sample.

## Non-claims

Do **not** state that B1/B2 prove clinical reliability, diagnostic validity, or that NAI is “highly robust.”  
Do **not** retune λ, weights, or features in response to these results.

## Artifacts

- `results/validation/bootstrap_nai_samples_b1.csv`
- `results/validation/bootstrap_summary_b1.csv`
- `results/validation/bootstrap_summary_b1.json`
- `results/validation/uncertainty_summary_b2.csv`
- `results/validation/uncertainty_summary_b2.json`
- `results/figures/validation_b1/`
- `results/figures/validation_b2/`

## Next

**B3 — external candidate audit → GO/NO-GO → EV-A only if GO.**  
No discovery NAI recomputation required.