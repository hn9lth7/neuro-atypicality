from __future__ import annotations

import numpy as np
import mne

from nai.spectral.bands import FMAX, FMIN

def compute_spectral_entropy(raw: mne.io.BaseRaw) -> dict[str, float]:
    raw_uv = raw.copy()
    raw_uv.apply_function(lambda x: x * 1e6, channel_wise=False)
    psd = raw_uv.compute_psd(method="welch", fmin=FMIN, fmax=FMAX, verbose=False)
    psds, _freqs = psd.get_data(return_freqs=True)
    psds = np.maximum(psds, 1e-20)
    p = psds / psds.sum(axis=1, keepdims=True)
    K = p.shape[1]
    H = -np.sum(p * np.log(p), axis=1)
    H_norm = H / np.log(K)
    return {
        "spectral_entropy_mean": float(np.mean(H_norm)),
        "spectral_entropy_std": float(np.std(H_norm)),
    }