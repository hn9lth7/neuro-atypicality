from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEAT = PROJECT_ROOT / "results" / "features"

PATH_SE = FEAT / "sheffield_features_v02.csv"
PATH_CG = FEAT / "sheffield_connectivity_graph_v05.csv"
PATH_D  = FEAT / "sheffield_dynamic_v06.csv"
OUT = FEAT / "sheffield_features_54d.csv"

SE = ["alpha_rel","beta_rel","theta_rel","spectral_entropy_mean","log_theta_alpha","log_theta_beta"]
C = [f"plv_mean_{b}" for b in ["theta","alpha","beta","gamma"]] + \
    [f"plv_median_{b}" for b in ["theta","alpha","beta","gamma"]]
G = [f"{m}_{b}" for m in ["mean_degree","degree_cv","clustering","global_efficiency","mean_path_length","laplacian_entropy"] for b in ["theta","alpha","beta","gamma"]]
D = [f"{m}_{b}" for m in ["mean_delta","cv_delta","mean_degree_cv","temporal_cv_degree_cv"] for b in ["theta","alpha","beta","gamma"]]
FEATURES_54 = SE + C + G + D
assert len(FEATURES_54) == 54, len(FEATURES_54)

def pivot_bands(df, metrics):
    out = {}
    for _, r in df.iterrows():
        pid = r['participant_id']
        band = r['band']
        if pid not in out:
            out[pid] = {'participant_id': pid}
        for m in metrics:
            out[pid][f'{m}_{band}'] = r[m]
    return pd.DataFrame(list(out.values()))

def main():
    se = pd.read_csv(PATH_SE)
    cg = pd.read_csv(PATH_CG)
    dyn = pd.read_csv(PATH_D)

    print(f'SE rows: {len(se)}')
    print(f'C+G rows: {len(cg)}')
    print(f'D rows: {len(dyn)}')

    cg_w = pivot_bands(cg, ['plv_mean','plv_median','mean_degree','degree_cv','clustering','global_efficiency','mean_path_length','laplacian_entropy'])
    d_w  = pivot_bands(dyn, ['mean_delta','cv_delta','mean_degree_cv','temporal_cv_degree_cv'])

    print(f'C+G wide: {cg_w.shape}')
    print(f'D wide: {d_w.shape}')

    m = se[['participant_id','age','group'] + SE].merge(cg_w, on='participant_id', how='inner')
    m = m.merge(d_w, on='participant_id', how='inner')
    print(f'Merged: {m.shape}')

    missing = [c for c in FEATURES_54 if c not in m.columns]
    if missing:
        print(f'MISSING {len(missing)}: {missing[:20]}')
        sys.exit(1)

    out = m[['participant_id','age','group'] + FEATURES_54].copy()
    out.to_csv(OUT, index=False)
    print(f'Saved -> {OUT}')
    print(f'Rows: {len(out)}, cols: {out.shape[1]}')
    print(out['group'].value_counts().to_string())
    print()
    print('Sample:')
    print(out[['participant_id','age','group','alpha_rel','plv_mean_alpha','clustering_alpha','mean_delta_alpha']].head().to_string())

if __name__ == '__main__':
    main()
