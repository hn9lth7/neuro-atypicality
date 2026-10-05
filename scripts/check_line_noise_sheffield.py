import mne
import numpy as np

raw = mne.io.read_raw_fif(
    'data/external/sheffield_asd_metadata/interpolated_64ch/ASD1_raw64.fif',
    preload=True, verbose=False
)

# PSD на сирих даних
psd = raw.compute_psd(method='welch', fmin=1, fmax=100, verbose=False)
psds, freqs = psd.get_data(return_freqs=True)
mean_psd = psds.mean(axis=0)

# Піки в діапазоні 45-65 Hz
mask = (freqs >= 45) & (freqs <= 65)
peaks = sorted(zip(freqs[mask], mean_psd[mask]), key=lambda x: -x[1])[:10]
print('Top 10 peaks 45-65 Hz (Sheffield ASD1, raw):')
for f, p in peaks:
    print(f'  {f:.1f} Hz  power={p:.2e}')
