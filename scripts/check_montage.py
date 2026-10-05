import mne
import scipy.io as sio
from pathlib import Path

# Завантажити перший файл як еталон
mat = sio.loadmat('data/external/sheffield_asd_metadata/all_sets/1Abby_Resting.set',
                  struct_as_record=False, squeeze_me=True)
eeg = mat['EEG']
ref_chans = [c.labels for c in eeg.chanlocs]
print('Reference montage:', len(ref_chans))
print(ref_chans)
