from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.preprocessing.pipeline import preprocess_minimal
from nai.connectivity.dynamic import compute_windowed_plv
from nai.dynamics.transitions import transition_series, dynamic_summary
from nai.graph.metrics import degree_cv

MANIFEST = PROJECT_ROOT / "data" / "external" / "sheffield_asd_metadata" / "manifest_final.csv"
OUT = PROJECT_ROOT / "results" / "features" / "sheffield_dynamic_v06.csv"

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
WINDOW_SEC = 10.0
STEP_SEC = 5.0

def process_one(row):
    raw = mne.io.read_raw_fif(row['output'], preload=True, verbose=False)
    raw_clean = preprocess_minimal(raw, notch_freqs=50.0)
    data = raw_clean.get_data()
    sfreq = float(raw_clean.info['sfreq'])
    duration = float(raw_clean.times[-1])
    
    rows = []
    for band, (fmin, fmax) in BANDS.items():
        matrices = compute_windowed_plv(data, sfreq, fmin, fmax, window_sec=WINDOW_SEC, step_sec=STEP_SEC)
        n_win = len(matrices)
        deltas = transition_series(matrices)
        dsum = dynamic_summary(deltas)
        deg_cv_series = np.array([degree_cv(W) for W in matrices])
        mean_deg_cv = float(deg_cv_series.mean())
        temporal_cv_deg_cv = float(deg_cv_series.std() / (abs(deg_cv_series.mean()) + 1e-12))
        rows.append({
            'participant_id': row['subject_id'],
            'band': band,
            'age': float(row['age']),
            'group': row['group'],
            'duration_s': duration,
            'n_windows': n_win,
            'mean_delta': dsum['mean_delta'],
            'cv_delta': dsum['cv_delta'],
            'mean_degree_cv': mean_deg_cv,
            'temporal_cv_degree_cv': temporal_cv_deg_cv,
            'std_delta': dsum['std_delta'],
            'max_delta': dsum['max_delta'],
            'temporal_entropy': dsum['temporal_entropy'],
        })
    return rows

def main():
    df = pd.read_csv(MANIFEST)
    print(f'Subjects: {len(df)}')
    all_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='D'):
        try:
            all_rows.extend(process_one(row))
        except Exception as e:
            print(f'ERROR {row["subject_id"]}: {e}')
    out = pd.DataFrame(all_rows)
    out.to_csv(OUT, index=False)
    print(f'Saved -> {OUT}')
    print(f'Rows: {len(out)}')

if __name__ == '__main__':
    main()
