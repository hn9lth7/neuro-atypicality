# B3 — External Validation Contract

**Status:** OPEN  
**Scope:** Frozen NAI v1 / 54-D feature path / EV-A only  
**Related Artifacts:** `models/nai_v1/`, `docs/mathematical_model.md`, product P0–P4 (feature-row path)

---

## 1. Purpose

B3 defines whether an independent external EEG cohort may be used for **external validation (EV-A)** of frozen NAI v1.

NAI scoring on an external cohort is **not performed** until a formal **GO** decision is recorded after a compatibility audit against this contract.

**EV-A means:**
- The same frozen model artifact `models/nai_v1/`;
- The same 54-dimensional feature semantics (blocks $SE$, $C$, $G$, $D$);
- No refit, no label-driven tuning, no change of $\lambda$ or weights.

---

## 2. GO / NO-GO Criteria

Each criterion is assigned one of:

| Status | Meaning |
|:---|:---|
| **GO** | Criterion satisfied with documented evidence |
| **NO-GO** | Criterion fails; cohort is not eligible for frozen 54-D EV-A |
| **UNKNOWN** | Insufficient evidence; does **not** count as GO |

**A single NO-GO is sufficient to reject the cohort for frozen 54-D EV-A.**

| # | Criterion | GO Condition |
|:---:|:---|:---|
| **1** | Independence | Different cohort / site from discovery data (not `ds006780` discovery set used to define the normative model) |
| **2** | Groups | Explicit ASD and TD/NT labels available in metadata |
| **3** | Age | Overlap with the discovery age band (8–13 years), **or** an explicit age policy documented in the EV protocol (out-of-range handling must not silently alter scoring) |
| **4** | Paradigm | Resting-state eyes-open and/or eyes-closed |
| **5** | Duration | Usable resting-state segment(s) of at least ~30–60 s after QC |
| **6** | Sampling rate | Documented; resampling only if predefined in the EV protocol |
| **7** | Montage | Channel layout compatible with the frozen 54-D extraction pipeline (or explicitly judged incompatible NO-GO for EV-A) |
| **8** | Reference | Known and reproducible (or reconstructible from documentation) |
| **9** | Preprocessing | Raw data available, **or** a fully specified preprocessing history that does not conflict with the frozen extraction contract |
| **10** | Access | Raw EEG (or contract-compatible data) actually obtainable |
| **11** | 54-D compatibility | $SE$, $C$, $G$, and $D$ features can be extracted with **unchanged** semantics relative to frozen NAI v1 |

---

## 3. Decision Rule

```text
EV-A = GO  only if criteria 1–11 contain no NO-GO  and no required criterion remains UNKNOWN
```

| Outcome | Meaning |
|:---|:---|
| **GO** | Proceed to fixed preprocessing $rightarrow$ 54-D extraction 
frozen `models/nai_v1/` scoring 
external validation report |
| **NO-GO** | Do not score under EV-A; archive compatibility memo only |
| **PENDING** | Waiting on access or missing evidence (e.g. author response) |
| **NOT PRIMARY** | Cohort may be useful for other analyses but is not a primary ASD/TD EV-A target |

Partial compatibility is not EV-A.

**Examples of separate studies (not frozen EV-A):**
- Different montage / reduced channel set;
- Reduced feature subset;
- Different sampling or preprocessing contract;
- Site harmonization methods that change feature semantics.

Such work must not modify `models/nai_v1/` or claim EV-A under this contract.

---

## 4. Candidate Registry

Fill cells with `GO` / `NO-GO` / `UNKNOWN`. Update the final column after audit.

| Cohort | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | EV-A Decision |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Mexico / Duville | NO-GO | NO-GO | — | — | — | — | — | — | — | — | — | **NO-GO** |
| Sheffield | — | — | — | — | — | — | — | — | — | — | — | **NO-GO** (for frozen 54-D EV-A) |
| HBN | — | — | — | — | — | — | — | — | — | — | — | **NOT PRIMARY** |
| ARISE | — | — | — | — | — | — | — | — | — | — | — | **NO-GO / weak** |
| Dup15q | — | — | — | — | — | — | — | — | — | — | — | **NO-GO** |
| Thailand 2025 | UNKNOWN | UNKNOWN | — | — | — | — | — | — | — | — | — | **PENDING** |

**Notes:**
- **Mexico / Duville:** Acquisition-level amplitude confound $
ightarrow$ NO-GO for EV-A.
- **Sheffield:** Unresolved harmonization / montage $	imes$ group issues $
ightarrow$ not GO for frozen 54-D EV-A.
- **HBN:** Not a dedicated ASD/TD case–control design $
ightarrow$ not primary EV target.
- **Thailand 2025:** Scientifically interesting pediatric ASD/NT candidate; raw access and montage compatibility pending author response; low channel count may imply NO-GO for 54-D as-is.

---

## 5. Frozen-Model Restrictions

While B3 is open, the following are **prohibited**:
- Refitting the TD normative model on external data;
- ASD/TD label-driven parameter tuning;
- Optimization of $\lambda$;
- Optimization of block weights;
- Feature replacement or addition;
- Substituting a reduced feature vector for the frozen 54-D set under the EV-A label;
- Montage remapping that changes feature semantics;
- Post-hoc amplitude rescaling without a predefined protocol written before scoring;
- Training ML classifiers (SVM, logistic regression, etc.) to “improve” validation under the EV-A label.

Any external cohort scored under EV-A must use the existing artifact:
```text
models/nai_v1/
```
without modification.

---

## 6. Allowed Work Before GO

**Allowed:**
- Provenance audit;
- Metadata audit;
- Channel / montage audit;
- Reference audit;
- Sampling-rate audit;
- Duration audit;
- Preprocessing-history audit;
- Raw-file accessibility check;
- Written compatibility report;
- Formal GO / NO-GO memo.

**Not allowed:**
- NAI scoring on external subjects;
- External NAI result tables presented as validation;
- ASD-vs-TD inference from NAI;
- ROC / AUC or classifier metrics framed as NAI validation;
- Model fitting of any kind aimed at external labels.

---

## 7. Required Evidence for GO

Before any external scoring, the cohort folder / memo must document:
- Independent cohort provenance;
- Group labels (ASD, TD/NT);
- Age;
- Resting-state paradigm;
- Usable duration;
- Sampling rate;
- Channel montage;
- Reference;
- Preprocessing status / history;
- Confirmed raw (or contract-compatible) access;
- A concrete plan to extract frozen 54-D features without changing the model.

Only after this evidence is complete may the cohort be marked:
```text
EV-A = GO
```

---

## 8. Workflow

```text
P0–P4 product path          CLOSED
        │
        ▼
B3 contract (this document) OPEN
        │
        ▼
Candidate acquisition / author contact
        │
        ▼
Compatibility audit (criteria 1–11)
        ├── NO-GO  → archive memo only (no NAI)
        ├── PENDING → wait for access / evidence
        └── GO
              │
              ▼
        Fixed preprocessing
              │
              ▼
        Fixed 54-D extraction
              │
              ▼
        Frozen models/nai_v1 scoring
              │
              ▼
        External validation report
```

---

## 9. Versioning

| Item | Rule |
|:---|:---|
| NAI model | Frozen v1.0 / `models/nai_v1` |
| This contract | May be updated to clarify criteria; updates must not retroactively redefine a completed GO without a new audit |
| EV-A results | Valid only if produced under an explicit GO memo referencing this document |

---

## 10. Summary

B3 is an acquisition and compatibility stage, not a scoring stage.

```text
No GO  →  no external NAI under EV-A
GO     →  frozen scoring only, unchanged model
```