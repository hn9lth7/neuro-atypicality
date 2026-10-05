# C4 — HBN-EEG Release 9 Metadata Audit

**Status:** METADATA AUDIT IN PROGRESS  
**Dataset:** OpenNeuro `ds005514` (v1.0.1) — HBN-EEG R9  
**Role:** Candidate for independent external evaluation (not approved for ML)

---

## 1. Audit Parameters & Evidence

- **Raw Data Access:** Yes (public participant files)
- **Legal Access:** Public research access (DUA required for phenotypic data)
- **Sample Size:** 295 subjects in R9 catalog ($n_{	ext{ASD}}$, $n_{	ext{TD}}$ currently unverified)
- **Paradigm:** `RestingState` present
- **Acquisition:** EGI high-density montage, 129 recorded channels reported
- **Age Range:** Broadly covers 5–21 years
- **Diagnostics:** `participants.tsv` lacks explicit ASD/TD labels; requires phenotypic DUA

---

## 2. Key Blocking Questions

1. Does R9 `participants.tsv` contain a usable subject-level ASD/TD diagnosis?
2. If absent, is there an accessible, auditable source for subject-level diagnoses and ID mapping?
3. What are the verified sampling rates, reference schemes, and channel locations across files?
4. Can the frozen feature definitions be computed without altering their underlying semantics?

---

## 3. Decision

> **C4-A / C4-B Status:** **PENDING / NO-GO.** No feature extraction, model fitting, or external evaluation until labels, legal access, and input compatibility are completely established.