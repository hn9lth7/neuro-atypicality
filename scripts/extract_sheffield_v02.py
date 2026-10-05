from __future__ import annotations

import sys
from pathlib import Path

import mne
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.preprocessing.pipeline import preprocess_minimal
from nai.qc.signal_quality import compute_qc
from nai.spectral.power import compute_band_powers
from nai.spectral.entropy import compute_spectral_entropy

SHEFFIELD_DIR = PROJECT_ROOT / "data" / "external" / "sheffield_asd_metadata" / "interpolated_64ch"
MANIFEST = PROJECT_ROOT / "data" / "external" / "sheffield_asd_metadata" / "manifest_final.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "features"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT = RESULTS_DIR / "sheffield_features_v02.csv"

def process_one(row):
    try:
        raw = mne.io.read_raw_fif(row['output'], preload=True, verbose=False)
        qc = compute_qc(raw)
        raw_clean = preprocess_minimal(raw, notch_freqs=50.0)
        powers = compute_band_powers(raw_clean)
        entropy = compute_spectral_entropy(raw_clean)
        return {
            'participant_id': row['subject_id'],
            'age': float(row['age']),
            'group': row['group'],
            'dataset': 'sheffield_orda',
            **qc,
            **powers,
            **entropy,
        }
    except Exception as e:
        print(f'Error on {row["subject_id"]}: {e}')
        return None

def main():
    print('=' * 70)
    print('NAI v0.2 — Sheffield ORDA extraction')
    print('=' * 70)
    df = pd.read_csv(MANIFEST)
    print(f'Subjects: {len(df)}')
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Processing'):
        feat = process_one(row)
        if feat is not None:
            records.append(feat)
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT, index=False)
    print(f'Saved -> {OUT}')
    print(f'Rows: {len(out_df)}')
    print(f'ASD: {(out_df["group"]=="ASD").sum()}')
    print(f'CTRL: {(out_df["group"]=="CTRL").sum()}')

if __name__ == '__main__':
    main()
