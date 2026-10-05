from __future__ import annotations

import mne
import numpy as np

def channel_std_uv(raw: mne.io.BaseRaw) -> dict[str, float]:
    if not raw.preload:
        raw = raw.copy().load_data()
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks) * 1e6
    stds = data.std(axis=1)
    return {raw.ch_names[picks[i]]: float(stds[i]) for i in range(len(picks))}