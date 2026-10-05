from __future__ import annotations

import mne
import numpy as np

def detect_bad_channels_by_std(
    raw: mne.io.BaseRaw,
    *,
    z_thresh: float = 5.0,
) -> list[str]:
    if not raw.preload:
        raw = raw.copy().load_data()
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks)
    stds = data.std(axis=1)
    med = float(np.median(stds))
    mad = float(np.median(np.abs(stds - med))) + 1e-12
    z = 0.6745 * (stds - med) / mad
    bad = [raw.ch_names[picks[i]] for i, zi in enumerate(z) if zi > z_thresh]
    return bad