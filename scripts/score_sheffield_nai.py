from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.extractor import score_nai_from_dataframe

FEAT = PROJECT_ROOT / "results" / "features"
PATH_SHEFFIELD = FEAT / "sheffield_features_54d.csv"
OUT = FEAT / "sheffield_nai_scores.csv"

def main():
    df = pd.read_csv(PATH_SHEFFIELD)
    print(f'Subjects: {len(df)}')
    
    result = score_nai_from_dataframe(
        df,
        group_col='group',
        age_col='age',
        id_col='participant_id',
        td_label='CTRL',
        lam=0.10,
    )
    result.to_csv(OUT, index=False)
    print(f'Saved -> {OUT}')
    print(result.to_string())

if __name__ == '__main__':
    main()
