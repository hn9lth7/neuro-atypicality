import mne
from pathlib import Path

# Знайти перший .bdf ds006780
files = list(Path('data/raw/ds006780').rglob('*.bdf'))
raw = mne.io.read_raw_bdf(str(files[0]), preload=True, verbose=False)

psd = raw.compute_psd(method='welch', fmin=1, fmax=100, verbose=False)
psds, freqs = psd.get_data(return_freqs=True)
mean_psd = psds.mean(axis=0)

mask = (freqs >= 45) & (freqs <= 65)
peaks = sorted(zip(freqs[mask], mean_psd[mask]), key=lambda x: -x[1])[:5]
print(f'ds006780 {files[0].name} — Top 5 peaks 45-65 Hz:')
for f, p in peaks:
    print(f'  {f:.1f} Hz  power={p:.2e}')
