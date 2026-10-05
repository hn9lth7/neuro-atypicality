import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv('results/features/sheffield_nai_scores.csv').dropna(subset=['NAI'])
y = (df['group'] == 'ASD').astype(int).values
s = df['NAI'].values

rng = np.random.default_rng(42)
n_boot = 5000
aucs = np.empty(n_boot)
for i in range(n_boot):
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2:
        aucs[i] = np.nan
        continue
    aucs[i] = roc_auc_score(y[idx], s[idx])

aucs = aucs[~np.isnan(aucs)]
print(f'AUC: {roc_auc_score(y, s):.4f}')
print(f'Bootstrap 95% CI: [{np.percentile(aucs, 2.5):.4f}, {np.percentile(aucs, 97.5):.4f}]')
print(f'Bootstrap mean: {aucs.mean():.4f}')
