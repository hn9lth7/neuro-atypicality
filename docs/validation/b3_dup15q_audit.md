# B3 — Dup15q / nonsyndromic ASD–TD Controls Audit

**Dataset:** Data from: *A quantitative electrophysiological biomarker of duplication 15q11.2-q13.1 syndrome*  
**DOI:** https://doi.org/10.5061/dryad.2th78  
**Paper:** Frohlich et al. (2016), *PLoS ONE*  
**Role:** SECONDARY biological / pipeline robustness cohort only  
**Not:** idiopathic pediatric EV-A replication of NAI v1.0 discovery (ds006780)  

**Status:** **AUDIT**  
**EV-A authorization:** **NOT GRANTED**  
**NAI scoring:** **PROHIBITED** until explicit GO  

---

## 1. Provenance

Public Dryad release (and Zenodo mirror). Study 1 describes spontaneous EEG in:

- Dup15q syndrome (n = 11) — **not** used in this audit arm  
- nonsyndromic ASD (n = 10 in paper)  
- typically developing TD (n = 9)  

Local audit used only:

- `ASD Controls.zip`
- `TD Controls.zip`
- `PROTOCOL FOR PROCESSING RESTING EEG.docx`
- `Matlab Scripts.zip`

**Provenance:** PASS (open access, no author gate for these files).

---

## 2. Local inventory

| Archive | Local size | EEGLAB `*_raw.set` count |
|---------|------------|---------------------------|
| ASD Controls | ~408 MB | **9** |
| TD Controls | ~165 MB | **9** |

Paper Study 1 ASD n = 10 → **one ASD recording missing** from the public ASD Controls archive (9/10).  
TD n = 9 matches paper.

Format per subject folder: `*_raw.set` + `*_raw.fdt` (EEGLAB).  
Some `.set` files store an outdated `.fdt` basename; MNE resolves to the paired file on disk.

---

## 3. Population and labels

| Item | Status |
|------|--------|
| Clinical nonsyndromic ASD | PASS (archive-level label) |
| TD controls | PASS (archive-level label) |
| Subject ID | folder / filename codes (e.g. `026s03v1`, `170s01v1`) |
| Age (years) | **UNKNOWN — BLOCKER** for age-dependent normative EV |
| Sex / IQ | not in local archives |

Group is recoverable **without authors** (zip membership).  
Age is **not** recoverable from filenames alone.

---

## 4. Protocol

| Item | Finding |
|------|---------|
| Condition | Spontaneous EEG during silent nonsocial videos (e.g. bubbles); **not** classic eyes-open rest as in ds006780 |
| Author pipeline | EEGLAB: FIR 1–50 Hz (no notch); event tags; 0–1.024 s epochs; automated bad segment/channel rules; ICA (1:124, pca=24); re-ref to channels 1–124 + 129 |
| Our use | Must **not** adopt author cleaning as frozen NAI preprocessing without a separate contract |

**Protocol difference vs discovery:** documented limitation.

---

## 5. Sampling rate, channels, duration (local probe, all files)

### ASD (9)

| ID | sfreq | n_ch | duration (s) |
|----|-------|------|--------------|
| 026s03v1 resting | 250 | 129 | 120.4 |
| 159s03v1 resting | 250 | 129 | 134.7 |
| 176s03v1 resting | 250 | 129 | 142.0 |
| 199s03v1 resting2 | 250 | 129 | 166.9 |
| 224s03v1 resting | 250 | 129 | 147.9 |
| 265s03v1 resting | 250 | 129 | 144.1 |
| **3002v1** | **500** | 129 | **662.5** |
| 303s03v1 resting | 250 | 129 | 170.2 |
| 309s03v1 resting | 250 | 129 | 138.8 |

### TD (9)

| ID | sfreq | n_ch | duration (s) |
|----|-------|------|--------------|
| 170s01v1 resting | 250 | 129 | 135.5 |
| 171s01v1 resting | 250 | 129 | 134.0 |
| 188s01v1 resting | 250 | 129 | 190.3 |
| 195s01v1 resting | 250 | 129 | 156.0 |
| 200s01v1 resting | 250 | 129 | 134.2 |
| 214s01v1 resting | 250 | 129 | 142.2 |
| 275s02v1 resting | 250 | 129 | 155.1 |
| 287s01v1 resting | 250 | 129 | 220.3 |
| 289s01v1 resting | 250 | 129 | 194.8 |

**Notes**

- Dominant rate: **250 Hz** (compatible with downsample path; discovery is 512 Hz).  
- One ASD file at **500 Hz** and much longer → QC / harmonization rule required if ever used.  
- All probed files report **boundary** events (discontinuities).

---

## 6. Montage

| Item | Status |
|------|--------|
| Channel labels | `E1`…`E128`, `Cz` (EGI-style) |
| n_channels | 129 |
| Nonzero XYZ (probe) | 123/129 |
| BioSemi 64 / standard_1005 native | **NO** |
| Mapping to frozen 54-D channel set | **NOT ESTABLISHED** |

Simple rename to discovery montage is **NO-GO**.  
Any future path needs a **documented** EGI→subset/10–10 policy.

---

## 7. Reference and preprocessing history

| Item | Status |
|------|--------|
| Raw reference in archive | not fully established from local probe alone |
| Author protocol | vertex acquisition; later re-ref to 1–124 + 129 |
| Line noise / notch | protocol: notch **off** during FIR 1–50 |
| ICA / cleaning already applied? | filenames say `*_raw`; treat as **pre-ICA raw EEGLAB export** until proven otherwise |

---

## 8. Feasibility for frozen NAI

| Requirement | Assessment |
|-------------|------------|
| Open raw + group labels | PASS |
| Resting-like spontaneous EEG | PASS with protocol caveat |
| Age for normative model | **FAIL / BLOCKER** |
| Sufficient TD for external 54-D covariance | **FAIL** (n_TD = 9) |
| Montage contract for 54-D | **FAIL until mapping** |
| Idiopathic ASD EV-A claim | **NO** (secondary biological cohort only; small n; missing 1 ASD) |

Even with age recovered, n_TD = 9 is inadequate for a stable 54-D regularized normative fit comparable to discovery (n_TD = 39).

---

## 9. Claim scope (if limited technical work is ever done)

**May support only**

- technical portability experiments (load → QC → partial features);  
- documentation of EGI vs BioSemi differences;  
- optional sensitivity analyses clearly labeled as **non-EV-A**.

**Must not claim**

- external validation of NAI v1.0;  
- pediatric idiopathic ASD replication;  
- clinical diagnostic performance.

---

## 10. GO / NO-GO

**Decision: NOT GO for EV-A.**

| Gate | Result |
|------|--------|
| Provenance | PASS |
| Raw EEG | PASS |
| ASD/TD labels | PASS (archive-level) |
| Age metadata | **BLOCKER** |
| Montage → 54-D | **BLOCKER** |
| Adequate external TD n | **BLOCKER** |
| Protocol match to discovery rest | partial only |

**NAI scoring and full 54-D extraction on Dup15q: prohibited.**

---

## 11. Parallel B3 status

| Candidate | Status |
|-----------|--------|
| Sheffield (ORDA) | AUDIT / NOT GO — waiting subject-level age/group key |
| Dup15q ASD/TD controls | AUDIT / **NOT GO** for EV-A — age + montage + n_TD |
| EEG latinus | task EEG only — not resting EV-A |
| ds006780 | discovery only — not external |

---

## 12. Next actions

1. Commit this audit; do not download Study1/Study2 volumes for NAI.  
2. Keep Sheffield metadata request active.  
3. Continue B3 search for open **resting** ASD+TD with **subject age**, denser TD sample, and mappable montage.  
4. No change to frozen NAI mathematics (λ = 0.10, equal weights, 54-D blocks).