import mne
import scipy.io as sio
from pathlib import Path

with open('data/external/sheffield_asd_metadata/target_montage.txt') as f:
    target_chans = [line.strip() for line in f]

biosemi64 = mne.channels.make_standard_montage('biosemi64')

def load_and_interpolate(set_path):
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    present = set(raw.ch_names)
    missing = [c for c in target_chans if c not in present]
    if missing:
        raw.set_montage(biosemi64, on_missing='ignore')
        raw.info['bads'] = missing
        raw.interpolate_bads(reset_bads=True, verbose=False)
    raw.pick_channels(target_chans, ordered=True)
    return raw, missing

# Обробити всі стандартні
out_dir = Path('data/external/sheffield_asd_metadata/interpolated')
out_dir.mkdir(parents=True, exist_ok=True)

manifest = []
for p in sorted(Path('data/external/sheffield_asd_metadata/all_sets').glob('*.set')):
    mat = sio.loadmat(str(p), struct_as_record=False, squeeze_me=True)
    eeg = mat['EEG']
    setname = str(eeg.setname)
    chans = [c.labels for c in eeg.chanlocs]
    
    # Skip alt-montage
    alt = sum(1 for c in chans if c.startswith(('A1','A2','A3','A5','A7','B1','B2','B4','C1','C2','C5','C7','D1','D2','D4')))
    if alt >= 5:
        print(f'SKIP alt-montage: {p.name} ({setname})')
        continue
    
    group = 'ASD' if setname.startswith('ASD') else 'CTRL'
    try:
        raw, missing = load_and_interpolate(p)
        out_path = out_dir / f'{setname}_interp.fif'
        raw.save(str(out_path), overwrite=True, verbose=False)
        manifest.append({
            'subject_id': setname,
            'group': group,
            'original_file': p.name,
            'interpolated_channels': ','.join(missing) if missing else '',
            'n_missing': len(missing),
            'n_channels': len(raw.ch_names),
            'output': str(out_path),
        })
        print(f'OK: {p.name} ({setname}, {group}) - interpolated {len(missing)} channels')
    except Exception as e:
        print(f'ERROR {p.name}: {e}')

import pandas as pd
df = pd.DataFrame(manifest)
df.to_csv('data/external/sheffield_asd_metadata/interpolated_manifest.csv', index=False)
print()
print('Total processed:', len(df))
print('ASD:', (df['group']=='ASD').sum())
print('CTRL:', (df['group']=='CTRL').sum())
print('Mean interpolated:', df['n_missing'].mean())
