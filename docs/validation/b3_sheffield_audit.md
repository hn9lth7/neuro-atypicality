# B3 — Sheffield External Validation Audit

**Candidate:** Sheffield Autism EEG Dataset  
**ORDA / Figshare:** https://doi.org/10.15131/shef.data.16840351  
**Linked paper:** Dickinson, Jeste & Milne (2022), *Cortex* — Dataset 1  
**Source acquisition:** Milne et al. (2019) and related Sheffield BioSemi studies  

**Role:** SECONDARY external validation candidate only  
**Primary pediatric validation:** **NO** (adult cohort, age ~18–68)  
**Status:** **AUDIT COMPLETE**  
**EV-A authorization:** **NOT GO**  
**NAI scoring:** **PROHIBITED**  

---

## 1. Provenance

Public release on University of Sheffield ORDA / Figshare (item 16840351), authored by Elizabeth Milne.

Dataset is referenced as **Dataset 1** in Dickinson, Jeste & Milne (2022), *Electrophysiological signatures of brain aging in autism spectrum disorder*.

Ethical approval for collection and sharing: UK Health Research Authority, **IRAS ID 212171**.  
Only participants with signed consent for data sharing are included.

**Provenance status:** documented / **PASS** (repository verified; local copy unpacked from `16840351.zip`).

---

## 2. Cohort and Population Metadata

| Item | Reported / Resolved | Status |
|------|---------------------|--------|
| ASC / autism spectrum condition | 28 | **PASS** |
| Neurotypical controls (TD) | 28 | **PASS** |
| Total recordings | 56 | **PASS** |
| File ID mapping | 56/56 1:1 (`1…56` ↔ `*Abby_Resting.set`) | **PASS** |
| Age range (Dickinson 2022, Dataset 1) | 18.08–68.33 years | **PASS** |

Cohort is **entirely adult**.

**Population compatibility with discovery (ds006780, ages 8–13):** **NO**.  
**Validation role:** Secondary transfer / technical portability only. Must **not** be presented as independent pediatric ASD validation.

---

## 3. EEG Protocol (Protocol-Level)

| Item | Reported / Observed | Status |
|------|---------------------|--------|
| Condition | Eyes-closed resting state | Protocol diff vs discovery (eyes-open) |
| Nominal duration | ~150 s (2.5 min) | **PASS** |
| System | BioSemi ActiveTwo | **PASS** |
| Released format | EEGLAB `.set` + `.fdt` (56/56 pairs) | **PASS** |
| Local sampling rate | **512 Hz** (all 56) | **PASS** |
| Local duration | **146.89–160.00 s** | **PASS** |

---

## 4. Sampling Rate and Acquisition Parameters

| Item | Status |
|------|--------|
| Sampling rate (released files) | **512 Hz** in all 56 (**PASS**) |
| Native acquisition rate in literature | Often reported 2048 Hz; release is downsampled |
| Highpass / lowpass in EEGLAB metadata | 0.0 / 256.0 (metadata only; not full history) |
| Reference configuration | **UNKNOWN** |

---

## 5. Montage Geometry & Channel Audit

All **56/56** recordings contain **valid 3D channel locations** for every channel. Channel counts vary between **47–63** channels across recordings.

### Montage Categorization

| Scheme | $n$ | Group Composition | Sub-IDs |
|--------|-----|-------------------|---------|
| **1010-like** | 47 | 28 ASD + 19 TD | All except below |
| **ABCD-like** | 9 | **9 TD only** | 30, 31, 32, 33, 39, 40, 41, 43, 44 |

### Channel Intersections & Geometry Analysis

* **1010-like Cohort (47 subjects):** The literal channel-name intersection contains only **5 channels** (`F7, O2, P1, P10, POz`), with only 2 overlapping the core 10–20 set (`F7`, `O2`).
* **Coverage:** 21 channels are present in $\ge 45/47$ subjects; 57 channels present in $\ge 40/47$ subjects. A greedy subset yields 30 subjects sharing 20 channels.
* **Procrustes Geometry vs `10Abby_Resting.set`:**

| Recording | Matched Pairs | NN Mean | Procrustes RMS | Scale |
|-----------|---------------|---------|----------------|-------|
| 30Abby | 48 | 5.0 mm | **7.3 mm** | 0.9872 |
| 31Abby | 45 | 5.4 mm | **7.3 mm** | 0.9849 |
| 32Abby | 54 | 4.9 mm | **7.2 mm** | 0.9887 |
| 13Abby | 59 | 0.0 mm | **0.0 mm** | 1.0000 |
| 44Abby | 45 | 5.4 mm | **7.7 mm** | 0.9868 |

**Montage Conclusion:** Validated 1:1 mapping between nonstandard A/B/C/D labels and standard labels required by the frozen 54-D pipeline is **not established**. Simple nearest-neighbor renaming to `standard_1005` is **PROHIBITED** (typical errors 10–20 mm; e.g. A25 $	o$ Iz $ pprox 34	ext{ mm}$).

---

## 6. Preprocessing & Quality Control Blockers

* **Reference History:** **UNKNOWN** (Must be deterministic and non-optimized on labels).
* **Line-Noise:** **UNKNOWN** (50 Hz notch may be required as acquisition-specific preprocessing).
* **Discontinuities:** MNE reports `boundary` annotations with non-zero duration and undocumented numeric event codes (`1, 3, 4, 6, 7`). Boundary-aware QC is mandatory before any feature calculation.

---

## 7. External Validation Decision & Rationale

**Decision:** **NOT GO for Frozen EV-A / NAI v1.0 External Validation.**

### Key Reasons for NOT GO
1. **Montage Heterogeneity:** The cross-cohort intersection of the 47 `1010-like` subjects yields only 5 common channels.
2. **Channel Reduction Penalty:** The largest common subset with $\ge 20$ channels reduces the sample size to only 30 subjects.
3. **Group Confounding:** Dropping the 9 `ABCD-like` participants removes 9 TD controls, creating a strong subgroup imbalance (28 ASD vs 19 TD).
4. **No Silent Harmonization:** Neither silent renaming nor channel padding/truncation is permitted for frozen EV-A validation.
5. **Out-of-Domain Age Cohort:** The adult age range (18–68 years) lies entirely outside the frozen EV-A pediatric normative support domain.

---

## 8. Pipeline Status & Allowed Future Use

### Summary Checklist

- [x] Sheffield ID / group / age audit: **PASS**
- [x] File integrity & 1:1 mapping: **PASS**
- [ ] Montage compatibility with frozen 54-D NAI: **NOT GO**
- [ ] Preprocessing & Reference history: **UNRESOLVED**
- [ ] NAI feature extraction: **DO NOT RUN**

### Allowed Future Use (Non-EV-A)
Sheffield may be used for **Sheffield-native exploratory research**, including:
* Native TD normative models on a predefined common channel subset (e.g., 20-channel / 30-subject subset or 21-channel / 45-subject subset);
* Methodological exploration of ABCD-like to 10–10 geometric mapping.

*Any such analysis must be explicitly designated as a new Sheffield-native model and must NOT be presented as validation of the frozen NAI v1.0 pipeline.*

### Next Action
**Continue B3 external-validation search with pediatric resting-state ASD+TD datasets (e.g., ABC-CT or other eligible cohorts).**