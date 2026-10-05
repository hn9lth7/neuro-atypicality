# Mexico Acquisition QC Memo

**Status:** CLOSED for NAI external validation (pending provenance explanation)  
**Date:** 2026-09  
**Scope:** Open data restEO subsets (typically developing vs high autistic traits), Mexico / Duville-related releases  
**Relation to NAI:** Does **not** modify frozen NAI v1.0 (`ds006780`). Mexico is a separate technical / QC branch.

---

## 1. Purpose

Assess whether Mexico resting-state EEG (eyes open) can support **harmonized external validation** of the Neuro-Atypicality Index (NAI) after:

1. Native GDF ingestion (BioSig),
2. Channel harmonization to a common montage,
3. Sampling-rate alignment,
4. Average re-referencing,
5. Structural and amplitude QC.

The goal was **not** to optimize group separation or to retune NAI on Mexico labels.

---

## 2. Data Subsets (As Used)

| Subset | Label in Pipeline | restEO Files | Native Montage | Native sfreq |
|:---|:---|---:|:---|:---|
| Typically developing | `td` | 31 | 32 EEG channels | 256 Hz |
| High autistic traits* | `asd` (folder name only) | 31 | 24 EEG channels | 500 Hz |

*\*Terminology: Primary sources describe **children with high autistic traits**, not necessarily a clinically confirmed ASD diagnosis. Until papers/README are fully reconciled, group label in results tables is treated as **cohort folder id**, not a clinical claim.*

Age/sex were taken from the accompanying `Age_Gender.xlsx` tables where available.

---

## 3. Pipeline Stages Completed

| Stage | Script / Step | Result |
|:---|:---|:---|
| Single-file GDF npy | `40_mexico_gdf_to_raw.py` | PASS (pilot) |
| Batch native ingest | `41_mexico_batch_rest_ingest.py` | TD 31/31; ASD failed only under hard `n_ch=32` check |
| Montage header audit | `41a_mexico_gdf_header_audit.py` | PASS |
| Reference / units header audit | `41b_mexico_reference_audit.py` | PASS (no explicit Reference field) |
| Harmonized ingest | `41c_mexico_harmonized_ingest.py` | **62/62 PASS** |
| MNE structural QC | `42_mexico_harmonized_qc.py` | **62/62 PASS** |
| Amplitude triage | `42a_mexico_amplitude_triage.py` | PASS (diagnostic) |
| Pairwise scale audit | `42b_mexico_scale_audit.py` | PASS (critical finding) |
| Full native scale audit | `42c_mexico_full_scale_audit.py` | PASS (critical finding) |

**Not run (deliberately stopped):** Spectral features, connectivity, graph, dynamic block, NAI scoring, ASD–TD inference.

---

## 4. Harmonization Design (Technical)

### Common Channel Set

$$C_{\mathrm{common}} = \{\mathrm{FP1},\mathrm{FP2},\mathrm{F3},\mathrm{F4},\mathrm{F7},\mathrm{F8},\mathrm{Fz},\mathrm{T7},\mathrm{T8},\mathrm{C3},\mathrm{C4},\mathrm{Cz},\mathrm{P3},\mathrm{P4},\mathrm{P7},\mathrm{P8},\mathrm{Pz}\}$$

$$|C_{\mathrm{common}}| = 17$$

All 17 channels are present in **both** native montages (header audit).

**Channels not used in the feature montage:**
- TD-only: AF*, FC*, CP*, PO*, Oz  
- ASD-only: O1, O2, AFz, CPz, POz, **A1, A2**

### Processing Order (Per Recording)

1. Read native GDF via BioSig (short Windows temp path).  
2. Select $C_{\mathrm{common}}$ in fixed canonical order.  
3. Average reference over the 17 common channels only (A1/A2 excluded).  
4. Resample ASD **500 $
ightarrow$ 256 Hz**; TD remains 256 Hz.  
5. Store `(n_times, 17)` float64 arrays in **$\mu	ext{V}$ as reported by BioSig**, plus metadata JSON.

Output root:
```text
data/raw/mexico_duville/converted/common_17/{td,asd}/
results/mexico/qc_harmonized_common17.csv
results/mexico/qc_mne_common17.csv
```

---

## 5. Structural QC (Harmonized)

From `qc_mne_common17.csv` / script `42`:

| Check | Result |
|:---|:---|
| Files | 31 TD + 31 ASD = 62 |
| Status | All `ok` |
| Channels | 17 |
| sfreq | 256 Hz |
| Finite values | Yes |
| Zero-variance channels | 0 |
| MNE `RawArray` | 62/62 |
| Duration | ~105–163 s (mean $ pprox$ 126 s) |

**Conclusion:** I/O, montage intersection, resampling, and MNE bridge are **technically valid**.

---

## 6. Amplitude and Native Scale Findings

### 6.1 Harmonized Amplitude Triage (`42a`)

Exploratory thresholds (not optimized on group labels):
- Soft: `mean_abs` $\le 150\,\mu	ext{V}$ and P99 $\le 500\,\mu	ext{V}$ `candidate_clean`  
- Hard: `mean_abs` $\ge 500\,\mu	ext{V}$ or P99 $\ge 2000\,\mu	ext{V}$ `extreme`

| Group | `candidate_clean` | `borderline` | `extreme` |
|:---|---:|---:|---:|
| TD | 13 | 9 | 9 |
| ASD | 0 | 0 | **31** |

Median `mean_abs_uv` (harmonized):

- TD ≈ 107
- ASD ≈ 3131

Lowest TD examples are physiologically plausible (e.g. Part10 `mean_abs` ≈ 11 µV). Highest values reach tens of thousands of µV.

### 6.2 Pairwise Native Audit (`42b`)

Example: TD Part10 vs ASD Part12 **before** harmonization.

| Quantity | TD Part10 | ASD Part12 |
|:---|:---|:---|
| sfreq | 256 | 500 |
| n_channels | 32 | 24 |
| Median PhysicalMaximum | ~103 | ~4914 |
| Max PhysicalMaximum | ~222 | **187500** |
| Scaling (median) | ~0.001 | ~0.001 |
| mean_abs (native) | ~14 | ~12650 |
| Data max / P99 | Within ~222 | **At 187500** |

BioSig `scaling ≈ 0.001` is similar; **PhysicalMaximum and sample ranges are not**.

### 6.3 Full-Cohort Native Scale Audit (`42c`)

All 62 restEO GDF files readable.

**Group Medians (Native Data):**

| Metric | TD Median | ASD Median |
|:---|---:|---:|
| pmax_median | 689.14 | 3560.66 |
| pmax_max | 8439.17 | 14837.4 |
| mean_abs | **128.41** | **3692.44** |
| data_p99 | 717.87 | 9463.86 |
| data_max | 4810.04 | 10730.5 |
| frac_at_pmax | **0** | **0** |
| frac_at_pmin | 0 | 0 |
| n_ch_high_sat | 0 | 0 |

Approximate median amplitude ratio:

$$\frac{\mathrm{median}(|x|)_{\mathrm{ASD}}}{\mathrm{median}(|x|)_{\mathrm{TD}}} \approx \frac{3692.44}{128.41} \approx 28.8$$

**Saturation:** Not cohort-wide. Median fraction of samples at PhysicalMaximum is 0 in both groups. ASD maximum `frac_at_pmax` ≈ 0.042 on individual files; high-sat channel count median 0 (max 1).

**TD Heterogeneity:** TD is not uniformly “clean” (e.g. max `mean_abs` ≈ 8050; extreme PhysicalMaximum outliers exist). The pattern is:

```text
Mexico native data
 ├── TD: heterogeneous amplitude scale
 └── ASD: systematically higher amplitude scale
         + occasional (not universal) saturation
```

---

## 7. What This Does Not Justify

1. **Global scale correction** such as `ext(ASD) → ext(ASD) / 29`
   - PhysicalMaximum varies across channels and subjects.
   - Scaling field is similar while physical ranges differ.
   - Occasional ceilings (e.g. 187500) are not a single stable gain.
   - Post-hoc constants would be unprincipled relative to NAI validation goals.

2. **Calling the issue pure clipping/saturation**
   - Contradicted by median `frac_at_pmax` = 0 for both groups.

3. **Proceeding to SE / connectivity / NAI on 31+31**
   - Absolute-power features would confound acquisition scale with biology.
   - Even relative features remain exploratory while units/reference/hardware are unexplained.

4. **Strict external validation of frozen NAI v1.0**
   - NAI v1.0 was built on a different system (64-ch BioSemi-class, matched discovery protocol).
   - Mexico differs in `n_channels`, `s_freq`, and native amplitude statistics.

5. **Clinical ASD claims from the `asd` folder alone**
   - Pending full reconciliation with source papers (high autistic traits vs diagnosis).

---

## 8. What Remains Valid

| Item | Status |
|:---|:---|
| BioSig GDF read path (incl. short-path workaround) | Valid |
| Montage intersection $C_{\mathrm{common}}$ | Valid |
| Harmonized 17-ch / 256 Hz / AR store | Valid as engineering product |
| Structural MNE QC | Valid |
| Detection of acquisition-level amplitude confound **before** features | Valid scientific QC outcome |
| NAI v1.0 on `ds006780` | Unchanged / frozen |

---

## 9. Decision (Current)

| Action | Decision |
|:---|:---|
| Spectral / connectivity / graph / dynamic features on Mexico | **STOP** |
| NAI scoring on Mexico | **STOP** |
| ASD–TD statistical comparison | **STOP** |
| Global amplitude rescaling | **NO-GO** |
| Mexico as external validation cohort for NAI v1.0 | **NO-GO** until provenance/units/hardware explain the scale gap |
| Documentation of this QC branch | **GO** (this file) |

Optional later uses (only after provenance):
- Technical pilot on a **pre-registered** clean TD subset (not asymmetric “TD kept, ASD dropped” validation);  
- Exploratory relative-only analyses with explicit acquisition caveats;  
- Replacement external dataset with matched acquisition.

---

## 10. Required Provenance Follow-up

Before any reopen of Mexico for EV:

1. Dataset README / codebook for both releases.  
2. Methods in primary papers: amplifier, gain, ADC resolution, reference, filtering, units.  
3. Explicit statement why protocols differ (32 ch 256 Hz vs 24 ch 500 Hz).  
4. Confirmation of physical units returned by BioSig for these GDF 1.25 files.  
5. If needed: contact data authors regarding PhysicalMaximum / calibration.

Until then, amplitude differences are treated as **unresolved acquisition heterogeneity**, not as a neurophysiological group effect.

---

## 11. Key Artifacts

```text
results/mexico/
  qc_rest_ingest.csv
  qc_harmonized_common17.csv
  qc_mne_common17.csv
  amplitude_triage_common17.csv
  scale_saturation_audit_native.csv

data/raw/mexico_duville/converted/common_17/{td,asd}/
  *_data.npy
  *_metadata.json
```

Scripts: `40`, `41`, `41a`, `41b`, `41c`, `42`, `42a`, `42b`, `42c`.

---

## 12. One-Paragraph Summary (for Paper / Thesis)

Mexico restEO data were successfully ingested and harmonized to a 17-channel common montage at 256 Hz with average reference (62/62 structurally valid under MNE). Native GDF audits showed a large systematic amplitude-scale difference between the typically developing and high-autistic-traits subsets (median absolute amplitude ratio $\approx 29$), present before harmonization. Cohort-wide saturation was not supported (median fraction of samples at PhysicalMaximum = 0 in both groups). Global rescaling and direct cross-group NAI external validation were therefore not justified. NAI v1.0 remains defined and frozen on the discovery cohort (`ds006780`); Mexico is retained as a documented acquisition-QC case study pending provenance clarification.

---

**Document status:** Draft aligned with completed QC through `42c`. Update only after provenance audit, not after ad-hoc feature experiments.