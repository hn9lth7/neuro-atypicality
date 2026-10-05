# C4 — Independent Cohort Protocol (Track C)

**Status:** ACTIVE protocol (no external evaluation until access + compatibility PASS)  
**Depends on:** C0 protocol, C3 development freeze  
**Does not modify:** NAI v1.0 or `ds006780` development benchmark

---

## 1. Purpose

Evaluate whether a **frozen** supervised pipeline specification (C3) and/or the same 54-D feature semantics can be applied to an **independent** subject cohort without retuning on that cohort.

C4 is **not** another development fold of `ds006780`.

---

## 2. Stages

| Stage | Name | Action |
| :--- | :--- | :--- |
| **C4-A** | Cohort audit | Independence, labels, age, resting, access |
| **C4-B** | ML input compatibility | `srate`, montage, reference, channel set, EO/EC |
| **C4-C** | Extraction | Reproduce 54-D **only if** C4-B PASS |
| **C4-D** | External evaluation | Apply frozen recipe; no hyperparameter search on external data |
| **C5** | Report | External metrics + limitations |

> **Rule:** If C4-A or C4-B fails $\rightarrow$ **NO-GO** (document reason). Do not "adapt features until it works."

---

## 3. Hard Requirements (FAIL = NO-GO)

| Check | Required |
| :--- | :---: |
| Independent of `ds006780` (no shared subjects / same study reuse as "external") | PASS |
| Both ASD and TD (or ASD and NT) labels | PASS |
| Resting-state EEG | PASS |
| Raw EEG or fully documented preprocessing reproducible to raw-equivalent | PASS |
| Sampling rate known | PASS |
| Montage / channel locations known | PASS |
| Reference known (or documented recoverable) | PASS |
| Group labels at subject level | PASS |
| No subject overlap with development set | PASS |
| Legal/ethical access for research use | PASS |

---

## 4. Soft / Design Requirements

| Check | Preference |
| :--- | :--- |
| Pediatric age band overlapping $\approx 8\text{--}13$ (or documented shift) | Preferred |
| Eyes-open / eyes-closed stated | Required to record |
| Duration sufficient for spectral + PLV + windows (e.g., $\ge 1\text{--}5\text{ min}$ usable) | Preferred |
| Channel count compatible with 64-ch BioSemi-derived 54-D **or** explicit reduced-montage contract | Critical |
| Same feature definitions (SE/C/G/D) computable | Critical |

```text
NOT GO for frozen-54D external evaluation unless a new versioned ML input contract 
(e.g., C3.1 reduced montage) is written before touching the external labels for model selection.
```