# Research Question — NAI v1.0

## Primary question

How can resting-state EEG be transformed into a **quantitative, multidimensional measure of neurophysiological atypicality** relative to an age-dependent typically developing (TD) reference, without reducing the individual to a binary ASD/control label?

## Operational question

Given a subject-level EEG feature vector partitioned into spectral–entropy, connectivity, static graph, and dynamic blocks, does an equal-weight composite of age-adjusted residual Mahalanobis distances ($NAI_{\mathrm{v1.0}}$) yield a stable, interpretable normative score under leave-one-out TD calibration and regularization sensitivity?

## Secondary questions

1. Which feature blocks contribute most to individual NAI profiles?
2. Does the dynamic block add structure beyond static spectral features in individual deviation profiles?
3. Is the direction of residual deviation informative beyond the scalar NAI?

## Explicit non-questions (v1.0)

- Does NAI diagnose autism?
- What is the population distribution of NAI in ASD?
- What are optimal block weights trained on ASD labels?

With $n_{\mathrm{ASD}}=2$, v1.0 addresses **pipeline validity and individual normative profiling**, not group-level ASD inference.