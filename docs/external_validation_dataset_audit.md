# External Validation — Dataset & Access Audit

**Status:** Criteria frozen; no external scoring until GO + data in hand  
**Discovery Cohort:** OpenNeuro `ds006780` (SFARI_EEG) — **not** used as external validation  

**Related:** `docs/cohort_expansion_v12.md`, `docs/robustness_v12.md`

---

## 1. Purpose

Select **one** independent EEG cohort to apply the **frozen** NAI definition
(equal-weight block Mahalanobis; TD-only age correction; $\lambda = 0.10$)
**without** retuning weights or features on the external ASD sample.

This document records **eligibility and access**, not results.

---

## 2. Fixed Eligibility Criteria

| Criterion | Required |
|:---|:---|
| ASD + TD (or clear case/control labels) | Yes |
| Resting-state EEG | Yes |
| Age metadata | Yes |
| Independent of `ds006780` | Yes |
| Prefer child / adolescent age near 5–15 y | Strongly preferred |
| Eyes-open rest | Preferred |
| Public or obtainable raw EEG | Yes for GO |
| Channels $\ge 19$ (harmonized) or $\ge 32$ (preferred) | Preferred |
| TD $n \gtrsim 20	ext{–}30$ | Preferred |

### Validation Modes (Chosen *Before* Seeing Scores)

| Mode | Description |
|:---|:---|
| **EV-A** | Fit normative model on **external TD only**; score external ASD |
| **EV-B** | Score external subjects with **frozen `ds006780` TD model** (harder; needs age + montage compatibility) |
| **Strict 54-D** | Same channel set / construction as v1.0 blocks |
| **Harmonized** | Pre-specified common electrode subset; same formulas on reduced montage |

Default first external attempt: **EV-A + harmonized** if montage differs.

---

## 3. Decision Table

| Dataset | ASD+TD | Age | Rest | Raw Access | Decision |
|:---|:---:|:---:|:---:|:---:|:---|
| **`ds006780`** | Yes | 8–13 | EO | Public BIDS | **REJECT** (discovery) |
| **HBN EEG `ds005505`** | Not guaranteed case/control | 5–21 | RestingState | Public BIDS, 129 EGI | **NO-GO** for ASD external |
| **Thailand Sci Rep 35+35** | Yes | 5–12 | EO $\ge 5$ min | **Upon request only** | **HOLD** — request authors |
| **NIMH NDA** autism resting | Yes (multi-study) | Broad | Often EO | **NDA application** | **PRIMARY access path** |
| **EU-AIMS LEAP** | Yes | Developmental | RS EEG | Proposal / central access | **SECONDARY access** |
| **Sheffield 28+28** | Yes | **18–68** | **EC** ~2.5 min | ORDA / shared | **SECONDARY** (adult transfer only) |
| **ARISE (UNC Dataverse)** | ASD+TD files present | Check | Rest open/closed | Public Dataverse | **AUDIT row** — verify $n$/age/ch |

---

## 4. Access Status (Operational)

### 4.1 OpenNeuro ds006780
- **Role:** Discovery / internal NAI v1.0–v1.2 only  
- **Action:** None for external  

### 4.2 HBN ds005505 (+ related releases)
- **Why NO-GO:** Not a dedicated ASD vs TD case–control cohort for our protocol; clinical measures $
eq$ binary ASD/TD design  
- **Technical mismatch:** 129-ch EGI vs BioSemi 64; would force heavy harmonization without clear ASD labels  
- **Action:** Do **not** download for primary external validation  

### 4.3 Thailand (Tiawongsuwan et al., Sci Rep; DOI 10.1038/s41598-025-30971-w)
- **Scientific fit:** Strong (children 5–12, EO rest, 35 ASD + 35 NT)  
- **Acquisition:** BrainMaster Discovery 24E, **256 Hz**, **~19** electrodes (10–20)  
- **Access text:** Data available from corresponding author **upon reasonable request**  
- **Implication:** Only **harmonized / reduced-montage** NAI possible; not strict 54-D  
- **Action:** Optional formal request email; no pipeline work until files received  

### 4.4 NIMH Data Archive (NDA)
- **Scientific fit:** Strongest path to large independent autism resting EEG  
- **Access:** Institutional account + data access request / DUA  
- **Implication:** Multi-study montage heterogeneity pre-specify harmonization  
- **Action:** **Start NDA registration / access request** (recommended primary)  

### 4.5 EU-AIMS LEAP
- **Access:** Application / analytic proposal  
- **Action:** Secondary if NDA delayed  

### 4.6 Sheffield Adult Autism EEG
- **Mismatch:** Adult age; eyes-closed  
- **Use:** Optional adult **pipeline transfer**, not age-dependent EV-B  
- **Action:** Low priority  

### 4.7 ARISE Study EEG (UNC Dataverse)
- **Note:** Public `rest.open` / ASD–TD style files reported  
- **Action:** One-time metadata check ($n$, age range, channel count); add GO/NO-GO row when verified  

---

## 5. What We Will *Not* Do

- Download multiple datasets and keep only the one where NAI “works better”  
- Optimize weights or feature sets on external ASD labels  
- Treat adult eyes-closed cohorts as equivalent to child eyes-open normative NAI  
- Claim external validation before GO dataset is in hand and protocol is locked  

---

## 6. Recommended Immediate Actions

1. Commit this audit (criteria + decisions).  
2. **Primary:** Begin **NDA** access process.  
3. **Optional parallel:** Short data-request email to Thailand corresponding author.  
4. **Optional light:** ARISE metadata check (no full download until GO).  
5. Until data arrives: no external feature extraction scripts; only protocol docs if needed.  

---

## 7. GO Criteria (When Data Arrives)

A candidate becomes **GO** only if all hold:

- [ ] Independent of `ds006780`  
- [ ] Usable ASD and TD labels  
- [ ] Age available  
- [ ] Resting EEG usable after QC  
- [ ] Written decision: EV-A and/or EV-B; strict vs harmonized montage  
- [ ] Preprocessing parameters written **before** NAI scoring  

Then: extract $
$ QC $
$ score $
$ LOO/thresholds on external TD (EV-A) $
$ report overlap/tails only as descriptive.

---

## 8. Status Line for README

```text
v1.2 frozen (expanded cohort + LOO + robustness).
External validation: access-dependent (NDA primary; Thailand request-only; HBN NO-GO for ASD case-control).
```