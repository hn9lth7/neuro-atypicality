import json
import pandas as pd
from datetime import datetime

df = pd.read_csv('data/external/sheffield_asd_metadata/interpolated_manifest.csv')

manifest = {
    'dataset_name': 'sheffield_orda_dickinson_2022',
    'source': 'ORDA/Figshare DOI 10.15131/shef.data.16840351',
    'paper': 'Dickinson, Jeste & Milne (2022), PMID 35176551',
    'paradigm': 'eyes-closed resting-state',
    'duration_sec': 150,
    'sampling_hz': 512,
    'reference': 'common (CAR)',
    'montage': 'BioSemi 64 (Cz excluded as reference)',
    'n_channels': 63,
    'subjects': {
        'ASD': int((df['group']=='ASD').sum()),
        'CTRL': int((df['group']=='CTRL').sum()),
        'total': len(df),
    },
    'interpolation': {
        'method': 'MNE spherical spline',
        'mean_channels_interpolated': float(df['n_missing'].mean().round(2)),
        'max_channels_interpolated': int(df['n_missing'].max()),
    },
    'excluded': {
        'alt_montage_subjects': 9,
        'reason': 'Non-10-20 channel naming (A/B/C/D prefixes)',
    },
    'files': df.to_dict(orient='records'),
    'generated': datetime.utcnow().isoformat() + 'Z',
}

with open('data/external/sheffield_asd_metadata/sheffield_orda.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print('Saved: sheffield_orda.json')
print(json.dumps(manifest, indent=2)[:1500])
