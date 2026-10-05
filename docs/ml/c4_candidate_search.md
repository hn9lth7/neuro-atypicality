# C4 — External Candidate Search Results

## 1. Evaluated and Rejected Candidates

| Dataset | Reason for Rejection |
| :--- | :--- |
| **HBN R9** (`ds005514`) | No ASD/TD labels; 129 channels |
| **Sheffield ORDA** (Dickinson 2022) | 47–63 channels, no common montage |
| **SheffieldAutismBiomarkers CSV** | 62/125 channels, no raw EEG |
| **BCIAUT-P300** (NEMAR) | P300 task, 8 channels, ASD only |
| **KAU** | 16 channels, not public |
| **MARIE-ASD** | Task-based (speech), not resting-state |

---

## 2. Pending Evaluation Candidates

| Dataset | Size | Channels | Access Method | Notes |
| :--- | :---: | :---: | :--- | :--- |
| **LEAP** (AIMS-2-TRIALS) | 392 | Brain Vision (64) | Application | Processed data only (T1 & T2) |
| **NDA** (Dede 2021) | 395 | Various | DUA | Eyes open + closed, processed features |
| **ABIDE-II EEG** | $\approx 200$ | 64 (site-dependent) | NITRC registration | EEG only at specific sites |
| **OpenNeuro Full Search** | Unknown | Unknown | Public | GraphQL pagination pending |

---

## 3. Next Strategic Actions

1. Fix OpenNeuro GraphQL query syntax (PowerShell `$` escaping).
2. Inspect NDA collection 2021 description.
3. Check ABIDE-II EEG availability on a per-site basis.
4. Query ELIXIR catalogue for LEAP dataset access.
5. **Decision Point:** Apply for LEAP access / NDA DUA, or continue public search.