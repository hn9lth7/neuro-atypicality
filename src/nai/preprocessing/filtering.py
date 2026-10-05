from __future__ import annotations

from typing import Sequence

import mne

def apply_notch(
    raw: mne.io.BaseRaw,
    freqs: float | Sequence[float] = 60.0,
) -> mne.io.BaseRaw:
    f = [float(freqs)] if isinstance(freqs, (int, float)) else list(freqs)
    raw.notch_filter(freqs=f, fir_design="firwin", verbose=False)
    return raw

def apply_bandpass(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
) -> mne.io.BaseRaw:
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    return raw
