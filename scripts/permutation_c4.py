import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv('results/features/sheffield_nai_scores.csv').dropna(subset=['NAI'])
y_true = (df['group'] == 'ASD').astype(int).values
y_score = df['NAI'].values

observed_auc = roc_auc_score(y_true, y_score)

rng = np.random.default_rng(42)
n_perm = 10000
null_aucs = np.empty(n_perm)
for i in range(n_perm):
    y_perm = rng.permutation(y_true)
    null_aucs[i] = roc_auc_score(y_perm, y_score)

p_perm = (null_aucs >= observed_auc).mean()
print(f'Observed AUC: {observed_auc:.4f}')
print(f'Permutation p (one-sided): {p_perm:.6f}')
print(f'Null 95th percentile: {np.percentile(null_aucs, 95):.4f}')
print(f'Null 99th percentile: {np.percentile(null_aucs, 99):.4f}')
