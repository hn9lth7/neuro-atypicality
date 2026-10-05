import mne
import pandas as pd
from pathlib import Path

biosemi64 = mne.channels.make_standard_montage('biosemi64')
in_dir = Path('data/external/sheffield_asd_metadata/interpolated')
out_dir = Path('data/external/sheffield_asd_metadata/interpolated_64ch')
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv('data/external/sheffield_asd_metadata/interpolated_manifest.csv')

manifest = []
for _, row in df.iterrows():
    src = Path(row['output'])
    if not src.exists():
        src = in_dir / f"{row['subject_id']}_interp_raw.fif"
    if not src.exists():
        print(f'SKIP (no file): {row["subject_id"]}')
        continue

    raw = mne.io.read_raw_fif(str(src), preload=True, verbose=False)

    # Встановити BioSemi 64 montage
    raw.set_montage(biosemi64, on_missing='ignore')

    # Додати Cz як канал (нульовий), потім інтерполювати
    if 'Cz' not in raw.ch_names:
        raw = mne.add_reference_channels(raw, ['Cz'], copy=False)
        raw.set_montage(biosemi64, on_missing='ignore')
        raw.info['bads'] = ['Cz']
        raw.interpolate_bads(reset_bads=True, verbose=False)

    # Переконатися, що порядок каналів — BioSemi 64
    target = [c for c in biosemi64.ch_names if c in raw.ch_names]
    raw.pick(target)

    out_path = out_dir / f"{row['subject_id']}_raw64.fif"
    raw.save(str(out_path), overwrite=True, verbose=False)

    manifest.append({
        'subject_id': row['subject_id'],
        'group': row['group'],
        'n_channels': len(raw.ch_names),
        'output': str(out_path),
    })
    print(f'OK: {row["subject_id"]} ({row["group"]}) - {len(raw.ch_names)} channels')

out_df = pd.DataFrame(manifest)
out_df.to_csv('data/external/sheffield_asd_metadata/interpolated_64ch_manifest.csv', index=False)

print()
print('Total:', len(out_df))
print('ASD:', (out_df['group']=='ASD').sum())
print('CTRL:', (out_df['group']=='CTRL').sum())
print('Channels unique:', out_df['n_channels'].unique())
