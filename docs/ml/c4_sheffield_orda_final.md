# C4 — Sheffield ORDA Dickinson 2022: Final Audit

**Status:** C4-A PASS, C4-B PASS  
**Dataset:** ORDA/Figshare DOI [10.15131/shef.data.16840351](https://doi.org/10.15131/shef.data.16840351)  
**Paper:** Dickinson, Jeste & Milne (2022), PMID 35176551  

---

## 1. C4-A — Diagnosis Audit (PASS)

- **Sample Size:** $n = 47$ processed ($28\text{ ASD} + 19\text{ CTRL}$; 9 CTRL excluded due to alt-montage)
- **Group Identifiers:**
  - **ASD:** `ASD1`..`ASD29`
  - **CTRL:** `P10`, `P12`, ..., `P60`
- **Group Labels Source:** Extracted from `EEG.setname` in `.set` files
- **Paradigm:** Eyes-closed resting-state, $150\text{ s}$

---

## 2. C4-B — Acquisition & Compatibility (PASS)

- **Montage:** BioSemi 64 (Cz excluded as reference)
- **Channel Setup:** Fixed 63-channel array post-interpolation
- **Sampling Rate:** $512\text{ Hz}$
- **Reference:** Common Average Reference (CAR)
- **Interpolation Method:** MNE spherical spline
- **Interpolation Statistics:**
  - **Mean:** $5.1\text{ channels/subject}$
  - **Maximum:** 14 channels (`P56`)
  - **Minimum:** 0 channels (`ASD13`)

---

## 3. Associated Files & Paths

- **Manifest:** `data/external/sheffield_asd_metadata/interpolated_manifest.csv`
- **Dataset Configuration:** `data/external/sheffield_asd_metadata/sheffield_orda.json`
- **Interpolated Recordings:** `data/external/sheffield_asd_metadata/interpolated/*.fif`

---

## 4. Identified Limitations

1. **Class Imbalance:** $28\text{ ASD}$ vs. $19\text{ CTRL}$
2. **Missing Channel:** Cz excluded (requires interpolation if required by frozen pipeline contract)
3. **Reference Scheme:** CAR (must verify alignment with frozen pipeline expectations)
4. **Artifact Markers:** Boundary events present from original `pop_eegrej` preprocessing

---

## 5. Final Decision

> **Status:** **READY TO RUN** — Sheffield ORDA is fully validated for external evaluation as an independent ASD/TD cohort under Track C.