from __future__ import annotations

from typing import Sequence

import mne

def preprocess_minimal(
    raw: mne.io.BaseRaw,
    *,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    notch_freqs: float | Sequence[float] = 60.0,
    set_montage: bool = True,
    montage_name: str = "biosemi64",
) -> mne.io.BaseRaw:
    raw = raw.copy()
    if not raw.preload:
        raw.load_data()

    try:
        raw.pick(picks="eeg", exclude="bads")
    except TypeError:
        raw.pick_types(eeg=True, stim=False, emg=False, eog=False, exclude="bads")

    if set_montage:
        try:
            raw.set_montage(montage_name, match_case=False, on_missing="ignore")
        except Exception:
            pass

    freqs = [float(notch_freqs)] if isinstance(notch_freqs, (int, float)) else list(notch_freqs)
    raw.notch_filter(freqs=freqs, fir_design="firwin", verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw

def n_eeg_channels(raw: mne.io.BaseRaw) -> int:
    return len(mne.pick_types(raw.info, eeg=True, exclude="bads"))