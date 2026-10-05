import mne
import numpy as np
import scipy.io as sio
from pathlib import Path

# 1. Завантажити target montage
with open('data/external/sheffield_asd_metadata/target_montage.txt') as f:
    target_chans = [line.strip() for line in f]

# 2. Завантажити BioSemi 64 montage як reference
biosemi64 = mne.channels.make_standard_montage('biosemi64')

# 3. Функція інтерполяції
def load_and_interpolate(set_path, target_chans, biosemi64):
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    present = set(raw.ch_names)
    missing = [c for c in target_chans if c not in present]
    
    if not missing:
        # Вже має всі канали — тільки переупорядкувати
        raw.pick_channels(target_chans, ordered=True)
        return raw, []
    
    # Встановити BioSemi 64 montage, потім інтерполювати відсутні
    raw.set_montage(biosemi64, on_missing='ignore')
    raw.info['bads'] = missing
    raw.interpolate_bads(reset_bads=True, verbose=False)
    raw.pick_channels(target_chans, ordered=True)
    return raw, missing

# 4. Тест на одному файлі
set_path = 'data/external/sheffield_asd_metadata/all_sets/1Abby_Resting.set'
raw, missing = load_and_interpolate(set_path, target_chans, biosemi64)
print('1Abby (ASD1)')
print('  Original channels:', len(mne.io.read_raw_eeglab(set_path, preload=False, verbose=False).ch_names))
print('  After interpolation:', len(raw.ch_names))
print('  Interpolated:', missing)
print('  ch_names:', raw.ch_names)
