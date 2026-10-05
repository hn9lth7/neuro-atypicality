import scipy.io as sio
from pathlib import Path
import pandas as pd

rows = []
for p in sorted(Path('data/external/sheffield_asd_metadata/all_sets').glob('*.set')):
    mat = sio.loadmat(str(p), struct_as_record=False, squeeze_me=True)
    eeg = mat['EEG']
    setname = str(eeg.setname)
    comments = str(eeg.comments) if hasattr(eeg, 'comments') else ''
    filepath = str(eeg.filepath) if hasattr(eeg, 'filepath') else ''
    # Шукати будь-які поля з 'age', 'subject', 'demo'
    extra = {}
    for f in (eeg._fieldnames if hasattr(eeg, '_fieldnames') else []):
        v = getattr(eeg, f, None)
        if isinstance(v, str) and ('age' in v.lower() or 'year' in v.lower() or 'birth' in v.lower()):
            extra[f] = v
    rows.append({
        'file': p.name,
        'setname': setname,
        'comments': comments,
        'filepath': filepath,
        **extra,
    })

df = pd.DataFrame(rows)
df.to_csv('data/external/sheffield_asd_metadata/set_comments.csv', index=False)
print(df.to_string())
