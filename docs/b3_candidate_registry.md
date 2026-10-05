# B3 — External Cohort Candidate Registry

**Status:** B3.1 closed  
**Contract:** `docs/b3_external_validation_contract.md`  
**Scope:** Frozen NAI v1 / 54-D / **EV-A only**

---

## 1. Decision Summary

| Item | Status |
|:---|:---|
| **B3-A (eligible external cohort for frozen 54-D EV-A)** | **NOT AVAILABLE** |
| **External NAI scoring performed** | **No** |
| **Frozen model modified** | **No** |
| **$\lambda$ / weights / features tuned for external data** | **No** |
| **Thailand 2025** | **PENDING** (no author response; raw access unresolved) |

**B3-A Status:**  
**NO-GO — no eligible external cohort currently available under the EV-A contract.**

This does **not** permanently close B3. A future independent cohort (including Thailand, if access and compatibility are established) may reopen B3.3–B3.4 under the same contract without changing NAI v1.

---

## 2. What Was Done

1. Formal EV-A GO/NO-GO contract (11 criteria).  
2. Public search for independent pediatric ASD+TD/NT resting-state EEG with raw access.  
3. Screening of known prior candidates (Mexico, Sheffield, HBN, ARISE, Dup15q, Thailand).  
4. Explicit exclusion of discovery data (`ds006780`) from “external” status.

**Not Done:** External NAI tables, ASD-vs-TD inference on external scores, ROC/AUC, model refit.

---

## 3. Candidate Registry

**Statuses:** `GO` / `NO-GO` / `UNKNOWN` / `PENDING` / `NOT PRIMARY` / `NOT EXTERNAL`

| Candidate | Indep. | ASD+TD | Age | Rest | Raw | Montage ~64 | Access | 54-D | **EV-A** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `ds006780` (discovery) | No | Yes | 8–13 | EO | Yes | BioSemi 64 | Yes | Yes | **NOT EXTERNAL** |
| Mexico / Duville | Yes | — | — | — | — | — | — | NO-GO | **NO-GO** |
| Sheffield | Yes | Yes | Adults | EC | Yes | BioSemi | Yes | NO-GO* | **NO-GO** |
| HBN-EEG | Yes | Mixed Dx | 5–21 | Yes | Yes | ~128 | Yes | Likely NO-GO | **NOT PRIMARY** |
| ARISE (UNC) | Yes | Yes | UNKNOWN | EO/EC | `.mff` | EGI likely | Yes | UNKNOWN $
ightarrow$ likely NO-GO | **NO-GO for EV-A pending full audit; not GO** |
| Dup15q | Yes | Limited | — | — | — | Limited | — | NO-GO | **NO-GO** |
| Thailand 2025 | Yes | ASD+NT | 5–12 | EO | UNKNOWN | ~19–24 | No reply | UNKNOWN | **PENDING** |
| Healthy-only public rest (e.g. `ds003775`, `ds005385`) | Yes | No | Adults | Yes | Yes | 64 | Yes | No ASD | **NO-GO** |

*\*Sheffield: previously judged unsuitable for frozen 54-D EV-A (harmonization / montage $	imes$ group issues).*

**ARISE Note:** Public files confirm ASD/TD labels and resting open/closed `.mff`. Format indicates EGI-family acquisition, not BioSemi64. Without documented channel layout matching the frozen 54-D pipeline, EV-A remains **not GO**. A deeper pilot (`read_raw_egi`) may refine criteria 6–8 but is **not required** to hold the current B3-A decision if montage incompatibility is accepted as blocking 54-D semantics.

---

## 4. Frozen-Model Integrity Statement

During B3 work to date:

- No external NAI scoring was performed.  
- No artifact under `models/nai_v1/` was modified.  
- No $\lambda$ optimization, weight optimization, or feature substitution was performed for external validation.  
- Discovery-cohort analyses (B1–B2, product parity) remain separate from EV-A.

---

## 5. Implications for the Project Roadmap

```text
A   Research core (v1.0 + architecture)     CLOSED
B1  Bootstrap stability                     CLOSED
B2  Uncertainty                             CLOSED
B3  External validation
    ├── Contract                            CLOSED
    ├── B3.1 Candidate search               CLOSED
    ├── B3-A Eligible cohort                NOT AVAILABLE
    └── Thailand                            PENDING
P0–P4 Product feature-row path              CLOSED
C   Clinical ML                             NOT STARTED
```