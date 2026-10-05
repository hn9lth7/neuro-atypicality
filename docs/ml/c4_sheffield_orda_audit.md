# C4 — Sheffield ORDA Dickinson 2022 Audit

**Status:** AUDIT COMPLETE  
**Dataset:** ORDA/Figshare DOI [10.15131/shef.data.16840351](https://doi.org/10.15131/shef.data.16840351)  
**Paper:** Dickinson, Jeste & Milne (2022), PMID 35176551, PMC11813168  
**Role:** Candidate for independent external evaluation — **REJECTED** for C4

---

## 1. C4-A — Diagnosis Audit (PASS)

- **Sample Size:** 56 subjects (28 ASD: `ASD1`..`ASD29`, missing `ASD23` + 28 CTRL: `P1`..`P60`)
- **Metadata:** Labels stored in `.set` field `EEG.setname`
- **Paradigm:** Eyes-closed resting-state, $150	ext{ s}$
- **Sampling Rate:** $512	ext{ Hz}$ (uniform)
- **Reference:** Common (likely CAR)
- **Ages:** 18–68 (per dataset description)

---

## 2. C4-B — Acquisition / Feature Compatibility (HARD FAIL)

- **Channel Count Range:** 47–63 (non-uniform)
- **Common Channels (All 56):** 0
- **Common Channels (ASD $n=28$):** 8 (`F7`, `O1`, `O2`, `P1`, `P2`, `P9`, `P10`, `POz`)
- **Common Channels (CTRL $n=28$):** 0
- **Montage Issues:** 9 subjects use non-10-20 montage with `A*`/`B*`/`C*`/`D*` channel labels:
  - `30Abby` (P1), `31Abby` (P5), `32Abby` (P6), `33Abby` (P9)
  - `39Abby` (P20), `40Abby` (P24), `41Abby` (P25), `43Abby` (P29), `44Abby` (P31)
- **Preprocessing Impact:** Per-subject cleaning (`pop_select` in `EEG.history`) removed different channels per subject, breaking montage uniformity.

---

## 3. Final Decision & Notes

> **Decision:** **NO-GO** for frozen 54-D pipeline. No manifest file `sheffield_orda.json` will be created. No feature extraction, model fitting, or external evaluation.

**Dataset Utility:**
- Independent replication of Dickinson et al. (2022).
- Development of a new input contract for variable-channel recordings.
- **Not** suitable for C4 in its raw form.