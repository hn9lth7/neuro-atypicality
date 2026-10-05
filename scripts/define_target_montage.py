import mne
import numpy as np
from pathlib import Path
import scipy.io as sio
from collections import Counter

# Знайти всі унікальні канали серед 47 стандартних суб'єктів
def is_standard(chans):
    alt = sum(1 for c in chans if c.startswith(('A1','A2','A3','A5','A7','B1','B2','B4','C1','C2','C5','C7','D1','D2','D4')))
    return alt < 5

all_chans = []
subjects = []
for p in sorted(Path('data/external/sheffield_asd_metadata/all_sets').glob('*.set')):
    mat = sio.loadmat(str(p), struct_as_record=False, squeeze_me=True)
    eeg = mat['EEG']
    chans = [c.labels for c in eeg.chanlocs]
    if is_standard(chans):
        all_chans.append(chans)
        subjects.append(str(eeg.setname))

# Union всіх каналів
union = set()
for ch in all_chans:
    union.update(ch)

print('Standard subjects:', len(subjects))
print('Union of all channels:', len(union))
print(sorted(union))
print()

# Скільки суб'єктів мають кожен канал
freq = Counter()
for ch in all_chans:
    for c in ch:
        freq[c] += 1

n = len(subjects)
print('Channel frequency (top 70):')
for c, f in sorted(freq.items(), key=lambda x: -x[1])[:70]:
    print(f'  {c}: {f}/{n} ({100*f//n}%)')

# Зберегти target montage — канали, які є у >=50%
target = sorted([c for c, f in freq.items() if f/n >= 0.5])
print()
print(f'Target montage ({len(target)} channels):')
print(target)

with open('data/external/sheffield_asd_metadata/target_montage.txt', 'w') as f:
    f.write('\n'.join(target))
