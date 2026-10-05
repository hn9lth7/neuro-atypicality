import scipy.io as sio
from pathlib import Path
from collections import Counter

# Завантажити всі .set, виключити alt-монтаж
def is_standard(chans):
    alt_markers = sum(1 for c in chans if c.startswith(('A1','A2','A3','A5','A7','B1','B2','B4','C1','C2','C5','C7','D1','D2','D4')))
    return alt_markers < 5

data = []
for p in sorted(Path('data/external/sheffield_asd_metadata/all_sets').glob('*.set')):
    mat = sio.loadmat(str(p), struct_as_record=False, squeeze_me=True)
    eeg = mat['EEG']
    setname = str(eeg.setname)
    chans = [c.labels for c in eeg.chanlocs]
    if is_standard(chans):
        data.append((setname, chans))

print('Standard subjects:', len(data))

# Частота кожного каналу
freq = Counter()
for _, chans in data:
    for c in chans:
        freq[c] += 1

n = len(data)
thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
for th in thresholds:
    keep = [c for c, f in freq.items() if f/n >= th]
    print(f'Channels in >= {int(th*100)}% subjects: {len(keep)}')

# Канали у 100%
keep_100 = sorted([c for c, f in freq.items() if f == n])
print()
print('Channels in ALL standard subjects:', keep_100)
