# B3 — External Validation Candidate Registry

**Rule:** No NAI computation on a candidate until **GO**.  
**Mode first:** EV-A (external TD fits normative model; frozen 54-D pipeline).  
**Not first:** EV-B (score under ds006780 μ/Σ).

## Status legend

| Status | Meaning |
|--------|---------|
| DISCOVERY | Used to build frozen NAI — **NO-GO** as external |
| PENDING | Access or metadata incomplete |
| AUDIT | Technical/provenance review in progress |
| GO | Passed protocol checks; EV-A allowed |
| NO-GO | Failed provenance, design, or compatibility |
| SECONDARY | Possible transfer/adult study; not primary child EV |

## Candidates

| Candidate | Access | Population | Age | Rest | Raw EEG | Channels | Reference | Fs | Clean TD/ASD (approx.) | Provenance | Status |
|-----------|--------|------------|-----|------|---------|----------|-----------|-----|------------------------|------------|--------|
| OpenNeuro ds006780 | public | child | 8–13 | EO rest | yes | 64 BioSemi | documented | 512 | **canonical 39 TD + 2 ASD** (full release has more subjects/paradigms) | known | **NO-GO: discovery** |
| Thailand (Mahidol request) | request | child | 5–12 | EO | pending | ~19 | pending | ~256 | ~35 / ~35 (reported) | pending author | **PENDING** |
| Sheffield adult ASD/TD | public | adult | 18–68 | EC (typical) | check | BioSemi | documented | check | ~28 / ~28 (check paper) | documented | **SECONDARY** |
| HBN / related BIDS | public | mixed pediatric | 5–21 | EO/EC | yes | high-density | documented | check | not clean ASD-vs-TD case–control for primary EV | good | **NO-GO primary ASD EV** |
| Mexico / Duville-type | public | child | check | EO | yes | 24–32 | **unresolved risks** | 256/500 | check | acquisition confound risk | **NO-GO** until provenance cleared |

*Numbers outside ds006780 canonical set are approximate and must be verified at audit.*

## GO checklist (per candidate)

1. Legal/ethical access to data used for analysis  
2. Documented age (and plan if age range ≠ 8–13)  
3. Resting-state (or explicitly justified resting-like) protocol  
4. Channel montage mappable to pipeline (or documented reduced-montage adaptation **without** changing NAI formula)  
5. Usable TD reference sample size (practical minimum to be stated in audit; n_TD ≪ 39 increases uncertainty further)  
6. Group labels only for *association* claims; **not** for tuning λ/weights/features  
7. QC pass rate acceptable; no dominant site/device confound  
8. Written GO/NO-GO decision in `docs/validation/b3_<dataset>_audit.md`

## EV-A pipeline (only after GO)

```text
external raw EEG
  → provenance lock
  → frozen preprocessing contract
  → same 54-D feature contract
  → external TD → age model + Σ_λ (λ=0.10)
  → score all external subjects
  → EV-A report (no retuning)