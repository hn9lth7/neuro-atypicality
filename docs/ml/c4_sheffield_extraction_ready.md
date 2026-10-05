# C4 — Sheffield ORDA Dickinson 2022: EXTRACTION READY

**Status:** C4-A PASS, C4-B PASS, age extracted

---

## 1. Cohort Overview

- **Subjects:** $n = 46$ (28 ASD + 18 CTRL; P32 dropped due to missing age in filename)
- **Age Range:** 1–28 years ($	ext{mean} = 13.5, 	ext{SD} = 8.2$)
  - *Note:* ORDA is "Dataset 1" (child cohort) of Dickinson et al. (2022). The paper's "18–68" range describes the full 3-dataset study, not this specific subset.
- **Recording:** Eyes-closed resting-state, $150	ext{ s}$, $512	ext{ Hz}$, BioSemi 64 (Cz interpolated)
- **Exclusions:** 9 CTRL excluded due to alt-montage (`A`/`B`/`C`/`D` naming)

---

## 2. Files & Paths

- **Manifest:** `data/external/sheffield_asd_metadata/manifest_final.csv`
- **Features:** `results/features/sheffield_features_v02.csv` (pending)
- **64-channel FIF:** `data/external/sheffield_asd_metadata/interpolated_64ch/*.fif`

---

## 3. Age Extraction Method

Parsed from original BDF filenames embedded in `EEG.comments`:
- **ASD:** `AA_<id>_<age>_<batch>-Deci.bdf`
- **CTRL:** `P<id>_<age>_<batch>-Deci.bdf`
- **Missing Data:** Code `99` indicates missing age (affected 3 subjects: P25, P31, P32; only P32 in final 46).

---

## 4. Next Steps

1. Run `extract_sheffield_v02.py` $
ightarrow$ 54-D features (SE block first).
2. Extend to connectivity (C), graph (G), and dynamic (D) blocks.
3. Compute NAI scoring via `extractor.score_nai_from_dataframe`.
4. Compare with the `ds006780` development set.