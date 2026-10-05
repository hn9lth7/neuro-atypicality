from __future__ import annotations

import mne

def set_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw