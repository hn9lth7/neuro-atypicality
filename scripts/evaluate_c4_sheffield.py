from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEAT = PROJECT_ROOT / "results" / "features"
OUT_DIR = PROJECT_ROOT / "results" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORES = FEAT / "sheffield_nai_scores.csv"

def main():
    df = pd.read_csv(SCORES)
    df = df.dropna(subset=['NAI'])
    
    asd = df[df['group'] == 'ASD']['NAI'].values
    ctrl = df[df['group'] == 'CTRL']['NAI'].values
    
    print('=' * 72)
    print('C4 — Sheffield ORDA external evaluation')
    print('=' * 72)
    print(f'ASD  : n={len(asd):3d}  mean={asd.mean():.4f}  std={asd.std():.4f}  median={np.median(asd):.4f}')
    print(f'CTRL : n={len(ctrl):3d}  mean={ctrl.mean():.4f}  std={ctrl.std():.4f}  median={np.median(ctrl):.4f}')
    print()
    
    # Mann-Whitney U (non-parametric)
    u_stat, u_p = stats.mannwhitneyu(asd, ctrl, alternative='greater')
    print(f'Mann-Whitney U (ASD > CTRL): U={u_stat:.1f}, p={u_p:.6f}')
    
    # Welch t-test
    t_stat, t_p = stats.ttest_ind(asd, ctrl, equal_var=False)
    print(f'Welch t-test             : t={t_stat:.3f}, p={t_p:.6f}')
    
    # Cohen d
    pooled_std = np.sqrt(((len(asd)-1)*asd.std()**2 + (len(ctrl)-1)*ctrl.std()**2) / (len(asd)+len(ctrl)-2))
    cohen_d = (asd.mean() - ctrl.mean()) / pooled_std
    print(f'Cohen d                  : {cohen_d:.3f}')
    
    # AUC
    y_true = np.array([1]*len(asd) + [0]*len(ctrl))
    y_score = np.concatenate([asd, ctrl])
    auc = roc_auc_score(y_true, y_score)
    print(f'AUC (ASD vs CTRL)        : {auc:.4f}')
    
    # Balanced accuracy at median threshold
    threshold = np.median(y_score)
    y_pred = (y_score > threshold).astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f'Balanced accuracy (median threshold = {threshold:.4f}): {bal_acc:.4f}')
    print(f'Confusion matrix:')
    print(f'  TN={cm[0,0]:2d}  FP={cm[0,1]:2d}')
    print(f'  FN={cm[1,0]:2d}  TP={cm[1,1]:2d}')
    print()
    
    # Per-block results
    print('Per-block comparison (ASD vs CTRL):')
    print(f'{"Block":8s} {"ASD mean":>10s} {"CTRL mean":>10s} {"AUC":>8s} {"p (MW)":>10s}')
    for block in ['D_SE', 'D_C', 'D_G', 'D_D']:
        a = df[df['group']=='ASD'][block].values
        c = df[df['group']=='CTRL'][block].values
        auc_b = roc_auc_score([1]*len(a) + [0]*len(c), np.concatenate([a, c]))
        _, p_b = stats.mannwhitneyu(a, c, alternative='two-sided')
        print(f'{block:8s} {a.mean():10.4f} {c.mean():10.4f} {auc_b:8.4f} {p_b:10.6f}')
    
    # Save report
    report = {
        'n_asd': int(len(asd)),
        'n_ctrl': int(len(ctrl)),
        'asd_mean_nai': float(asd.mean()),
        'ctrl_mean_nai': float(ctrl.mean()),
        'cohen_d': float(cohen_d),
        'auc': float(auc),
        'mann_whitney_u': float(u_stat),
        'mann_whitney_p': float(u_p),
        'welch_t': float(t_stat),
        'welch_p': float(t_p),
        'balanced_accuracy': float(bal_acc),
    }
    import json
    out_json = OUT_DIR / 'c4_sheffield_evaluation.json'
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print()
    print(f'Saved -> {out_json}')

if __name__ == '__main__':
    main()
