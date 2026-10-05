from __future__ import annotations
import numpy as np
import mne

def compute_qc(raw: mne.io.BaseRaw) -> dict:
    data = raw.get_data()
    sfreq = raw.info["sfreq"]

    if np.nanmax(np.abs(data)) < 1e-2:
        data = data * 1e6

    mean_abs_uv = float(np.mean(np.abs(data)))
    std_uv = float(np.mean(np.std(data, axis=1)))

    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=50.0,
        fmax=70.0,
        n_fft=min(2048, data.shape[1]),
        verbose=False,
    )
    psd_mean = psd.mean(axis=0)

    trapz = getattr(np, "trapezoid", None) or np.trapz

    mask_narrow = (freqs >= 58) & (freqs <= 62)
    mask_wide   = (freqs >= 55) & (freqs <= 65)

    p_narrow = trapz(psd_mean[mask_narrow], freqs[mask_narrow]) if mask_narrow.any() else 0.0
    p_wide   = trapz(psd_mean[mask_wide],   freqs[mask_wide])   if mask_wide.any()   else 1e-12

    line_noise_60hz_ratio = float(p_narrow / (p_wide + 1e-12))

    n_eeg = len(mne.pick_types(raw.info, eeg=True, stim=False, emg=False, eog=False, exclude="bads"))

    return {
        "mean_abs_uv": mean_abs_uv,
        "std_uv": std_uv,
        "line_noise_60hz_ratio": line_noise_60hz_ratio,
        "n_channels": n_eeg,                   
        "duration_s": float(raw.times[-1] - raw.times[0]),
    }