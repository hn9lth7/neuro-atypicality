import mne
import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path

with open('data/external/sheffield_asd_metadata/target_montage.txt') as f:
    target_chans = [line.strip() for line in f]

biosemi64 = mne.channels.make_standard_montage('biosemi64')
out_dir = Path('data/external/sheffield_asd_metadata/interpolated')
out_dir.mkdir(parents=True, exist_ok=True)


def load_and_interpolate(set_path):
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    present = list(raw.ch_names)
    missing = [c for c in target_chans if c not in present]

    if not missing:
        raw.pick(target_chans)
        return raw, []

    # Спробувати встановити montage для наявних каналів
    try:
        raw.set_montage(biosemi64, on_missing='ignore', match_case=False)
    except Exception:
        pass

    # Побудувати повний 64-канальний масив
    sfreq = raw.info['sfreq']
    n_times = raw.n_times
    full_data = np.full((len(target_chans), n_times), np.nan, dtype=np.float64)

    # Скопіювати наявні канали
    for i, ch in enumerate(target_chans):
        if ch in present:
            full_data[i, :] = raw.get_data(picks=[ch])[0]

    # Побудувати info з biosemi64
    info = mne.create_info(ch_names=target_chans, sfreq=sfreq, ch_types='eeg')
    info.set_montage(biosemi64, on_missing='ignore')

    # Створити RawArray
    raw_full = mne.io.RawArray(full_data, info, verbose=False)

    # Позначити відсутні канали як bad
    raw_full.info['bads'] = missing

    # Інтерполювати
    raw_full.interpolate_bads(reset_bads=True, verbose=False)

    # Повернути тільки цільові канали
    raw_full.pick(target_chans)
    return raw_full, missing


def is_standard(chans):
    alt = sum(1 for c in chans if c.startswith(('A1','A2','A3','A5','A7','B1','B2','B4','C1','C2','C5','C7','D1','D2','D4')))
    return alt < 5


manifest = []
for p in sorted(Path('data/external/sheffield_asd_metadata/all_sets').glob('*.set')):
    mat = sio.loadmat(str(p), struct_as_record=False, squeeze_me=True)
    eeg = mat['EEG']
    setname = str(eeg.setname)
    chans = [c.labels for c in eeg.chanlocs]

    if not is_standard(chans):
        print(f'SKIP alt-montage: {p.name} ({setname})')
        continue

    group = 'ASD' if setname.startswith('ASD') else 'CTRL'
    try:
        raw, missing = load_and_interpolate(p)
        out_path = out_dir / f'{setname}_interp_raw.fif'
        raw.save(str(out_path), overwrite=True, verbose=False)
        manifest.append({
            'subject_id': setname,
            'group': group,
            'original_file': p.name,
            'n_missing': len(missing),
            'missing_channels': ','.join(missing),
            'n_channels': len(raw.ch_names),
            'output': str(out_path),
        })
        print(f'OK: {p.name} ({setname}, {group}) - interpolated {len(missing)}')
    except Exception as e:
        print(f'ERROR {p.name}: {e}')

df = pd.DataFrame(manifest)
df.to_csv('data/external/sheffield_asd_metadata/interpolated_manifest.csv', index=False)

print()
print('Total processed:', len(df))
if len(df) > 0:
    print('ASD:', (df['group']=='ASD').sum())
    print('CTRL:', (df['group']=='CTRL').sum())
    print('Channels:', df['n_channels'].unique())
    print('Mean missing:', df['n_missing'].mean())
    print('Max missing:', df['n_missing'].max())
    print('Min missing:', df['n_missing'].min())
