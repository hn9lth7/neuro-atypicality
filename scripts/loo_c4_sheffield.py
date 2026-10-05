from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from nai.features.extractor import score_nai_from_dataframe

FEAT = PROJECT_ROOT / "results" / "features"
IN = FEAT / "sheffield_features_54d.csv"
OUT = FEAT / "sheffield_nai_scores_loo.csv"

def main():
    df = pd.read_csv(IN)
    n = len(df)
    print(f"Subjects: {n}")

    loo_scores = np.full(n, np.nan)
    for i in range(n):
        test = df.iloc[[i]]
        train = df.drop(index=i)
        # Need >= 2 CTRL in train
        n_ctrl_train = (train['group'] == 'CTRL').sum()
        if n_ctrl_train < 2:
            continue
        combined = pd.concat([train, test], ignore_index=True)
        # Fit on train (normative from CTRL), score test
        # We score each subject in combined, then take the last row (test)
        scored = score_nai_from_dataframe(
            combined, group_col='group', age_col='age',
            id_col='participant_id', td_label='CTRL', lam=0.10,
        )
        loo_scores[i] = scored['NAI'].iloc[-1]
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{n}")

    df['NAI_loo'] = loo_scores
    df.to_csv(OUT, index=False)

    y = (df['group'] == 'ASD').astype(int).values
    s = df['NAI_loo'].values
    mask = ~np.isnan(s)
    y, s = y[mask], s[mask]
    auc = roc_auc_score(y, s)
    print()
    print(f"LOO-CV AUC: {auc:.4f}")
    print(f"In-sample AUC (earlier): 0.849")

if __name__ == '__main__':
    main()
