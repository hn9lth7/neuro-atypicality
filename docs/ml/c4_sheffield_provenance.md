# Provenance — Sheffield C4 Pipeline

## Input

- **Source:** ORDA/Figshare DOI [10.15131/shef.data.16840351](https://doi.org/10.15131/shef.data.16840351)
- **Paper:** Dickinson, Jeste & Milne (2022), PMID 35176551
- **Raw:** `.set` + `.fdt` (56 subjects)
- **Labels:** Parsed from `EEG.setname` in `.set` files (`ASD1`..`ASD29`, `P1`..`P60`)
- **Age:** Parsed from BDF filenames in `EEG.comments`

---

## Harmonization Pipeline

- **Step 1:** `interpolate_missing_channels` $
ightarrow$ `interpolated/*.fif` (63 channels)
- **Step 2:** `add_cz_back` $
ightarrow$ `interpolated_64ch/*.fif` (64 channels, BioSemi 10-20)
- **Step 3:** Features
  ```python
  preprocess_minimal(raw, notch_freqs=50.0)
  ├── raw.pick(picks="eeg", exclude="bads")
  ├── raw.set_montage("biosemi64")
  ├── raw.notch_filter(freqs=[50.0])
  ├── raw.filter(l_freq=1.0, h_freq=45.0)
  └── raw.set_eeg_reference("average", projection=False)
  ```
  $
ightarrow$ SE / C / G / D feature blocks
- **Step 4:** `score_nai_from_dataframe(td_label="CTRL", lam=0.10)`

---

## Verified Parameters

| Parameter | Value |
| :--- | :--- |
| **Sampling rate** | $512	ext{ Hz}$ |
| **EEG channels** | 64 |
| **Reference** | Average (CAR) |
| **Notch** | $50	ext{ Hz}$ |
| **Bandpass** | $1	ext{--}45	ext{ Hz}$ |
| **Duration** | $ pprox 160	ext{ s}$ |

---

## Output Files

### `results/features/`
- `sheffield_features_v02.csv` (SE + QC)
- `sheffield_connectivity_graph_v05.csv` (C + G, per-band)
- `sheffield_dynamic_v06.csv` (D, per-band)
- `sheffield_features_54d.csv` (merged, wide)
- `sheffield_nai_scores.csv` (NAI + $D_{SE} / D_C / D_G / D_D$)

### `results/ml/`
- `c4_sheffield_evaluation.json`
- `c4_sheffield_orda_dickinson_2022_audit.json`