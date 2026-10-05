# Preprocessing Protocol — NAI v1.0

**Status:** Frozen for v1.0 feature extraction  
**Principle:** Preprocessing fixed before group interpretation of NAI.

## Pipeline (resting-state run)

1. Load BIDS EEG (`mne` / `mne-bids`)
2. Load data into memory
3. Channel selection: EEG only (64 channels after pick)
4. Notch filter: **60 Hz** (line noise; QC of line noise on broader channel set where applicable)
5. Band-pass: **1–45 Hz**
6. Average reference
7. Spectral / connectivity / dynamic features computed on cleaned continuous segment (~60 s typical)

## Explicitly not in v1.0 core pipeline

- ICA / ASR as mandatory steps
- Automated bad-channel interpolation as a global rule (channel QC flags recorded where computed)
- Task-related epoching (resting-state only)

## Quality control

- Amplitude / variance summary
- Line-noise ratio
- EEG channel count (expected 64 after pick)
- Subject-level flags (e.g. `very_low_alpha`, extreme artifact)

Subjects failing hard QC (e.g. catastrophic artifact) are excluded from the canonical cohort; flagged subjects may be retained with explicit `qc_flag`.

## Reproducibility note

Feature CSVs under `results/features/` are the frozen inputs to NAI v1.0 scoring. Re-running extraction with the same protocol should regenerate equivalent tables; scoring (`scripts/30_nai_v10_unified.py`) assumes these subject-level tables.