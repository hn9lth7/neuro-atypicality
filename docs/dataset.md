# Dataset — NAI v1.0

## Source

- **OpenNeuro / SFARI EEG:** `ds006780`
- Modality: resting-state EEG (eyes-open blocks)
- Hardware: BioSemi ActiveTwo, **64 EEG channels**, **512 Hz**
- Age range (cohort): approximately **8–13 years**
- BIDS structure; raw EEG without project-specific preprocessing applied upstream

## Canonical analysis cohort (v1.0)

| Group | n | Role |
|-------|---|------|
| TD | 39 | Normative fit only |
| ASD | 2 | Scoring only (never enter fit) |
| **Total** | **41** | Clean subject-level sample |

### Exclusions

- **sub-10777** — extreme spectral artifact (independent QC)
- Subjects with missing essential metadata (e.g. age/group)

### Retention with flag

- **sub-11025** retained with `qc_flag = very_low_alpha`

## Recording structure

- Multiple resting-state runs per subject (typically up to ~6; subject-level means used)
- Analysis unit: **subject**, not run (except stability audits)

## Files used by the pipeline

```text
results/features/
  features_v032.csv                         # SE + demographics / QC
  connectivity_graph_subject_v0.5.csv       # C + G
  dynamic_subject_v0.6.csv                  # D