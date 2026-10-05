import pandas as pd
import re
from pathlib import Path

df = pd.read_csv('data/external/sheffield_asd_metadata/set_comments.csv')

def parse_age(comments, group):
    if not isinstance(comments, str):
        return None, None
    # AA_1_26_1-Deci.bdf  OR  P51_3_2-Deci.bdf
    m = re.search(r'(AA|P)[\d_]+\d+_(\d+)_(\d+)-Deci\.bdf', comments)
    if not m:
        m = re.search(r'(AA|P)(\d+)_(\d+)_(\d+)-Deci\.bdf', comments)
        if m:
            return int(m.group(3)), int(m.group(4))
        return None, None
    return int(m.group(2)), int(m.group(3))

rows = []
for _, r in df.iterrows():
    x, y = parse_age(r['comments'], r['setname'])
    rows.append({'subject_id': r['setname'], 'age_raw': x, 'batch': y})

age_df = pd.DataFrame(rows)
age_df['age'] = age_df['age_raw'].apply(lambda v: None if v is None or v == 99 else v)
age_df.to_csv('data/external/sheffield_asd_metadata/age_from_filename.csv', index=False)

print(age_df.to_string())
print()
print('Missing (99):', age_df['age'].isna().sum())
print('Age range:', age_df['age'].min(), '-', age_df['age'].max())
print()
print('Age distribution:')
print(age_df['age'].describe())
