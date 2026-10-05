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
from nai.connectivity.phase import bandpass_filter, compute_plv
from nai.graph.metrics import (
    mean_degree, degree_cv, weighted_clustering,
    global_efficiency, mean_path_length,
)
from nai.graph.spectral import algebraic_connectivity, laplacian_entropy

MANIFEST = PROJECT_ROOT / "data" / "external" / "sheffield_asd_metadata" / "manifest_final.csv"
OUT = PROJECT_ROOT / "results" / "features" / "sheffield_connectivity_graph_v05.csv"

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

def process_one(row):
    raw = mne.io.read_raw_fif(row['output'], preload=True, verbose=False)
    raw_clean = preprocess_minimal(raw, notch_freqs=50.0)
    data = raw_clean.get_data()
    sfreq = float(raw_clean.info['sfreq'])
    n_channels = data.shape[0]
    duration_s = float(raw_clean.times[-1] - raw_clean.times[0])
    
    rows = []
    for band, (fmin, fmax) in BANDS.items():
        data_bp = bandpass_filter(data, sfreq, fmin, fmax)
        W = compute_plv(data_bp)
        W_no_diag = W.copy()
        np.fill_diagonal(W_no_diag, 0.0)
        plv_mean = float(W_no_diag.sum() / (n_channels * (n_channels - 1)))
        plv_median = float(np.median(W_no_diag[np.triu_indices(n_channels, k=1)]))
        rows.append({
            'participant_id': row['subject_id'],
            'band': band,
            'age': float(row['age']),
            'group': row['group'],
            'n_channels': n_channels,
            'duration_s': duration_s,
            'plv_mean': plv_mean,
            'plv_median': plv_median,
            'mean_degree': mean_degree(W),
            'degree_cv': degree_cv(W),
            'clustering': weighted_clustering(W),
            'global_efficiency': global_efficiency(W),
            'mean_path_length': mean_path_length(W),
            'lambda2': algebraic_connectivity(W),
            'laplacian_entropy': laplacian_entropy(W),
        })
    return rows

def main():
    df = pd.read_csv(MANIFEST)
    print(f'Subjects: {len(df)}')
    all_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='C+G'):
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
