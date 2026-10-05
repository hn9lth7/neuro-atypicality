from __future__ import annotations
from nai.spectral.bands import BANDS, FMAX, FMIN
import numpy as np
import mne

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

def compute_band_powers(raw: mne.io.BaseRaw) -> dict:
    raw_uv = raw.copy()
    raw_uv.apply_function(lambda x: x * 1e6, channel_wise=False)

    psd = raw_uv.compute_psd(method="welch", fmin=1.0, fmax=45.0, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)  

    mean_psd = psds.mean(axis=0)

    trapz = getattr(np, "trapezoid", None) or np.trapz

    powers = {}
    for name, (fmin, fmax) in BANDS.items():
        idx = (freqs >= fmin) & (freqs < fmax)
        powers[f"{name}_abs"] = float(trapz(mean_psd[idx], freqs[idx]))

    total = sum(powers[f"{b}_abs"] for b in BANDS) + 1e-12

    for name in BANDS:
        powers[f"{name}_rel"] = powers[f"{name}_abs"] / total

    powers["theta_alpha"] = powers["theta_abs"] / (powers["alpha_abs"] + 1e-12)
    powers["theta_beta"] = powers["theta_abs"] / (powers["beta_abs"] + 1e-12)
    powers["alpha_beta"] = powers["alpha_abs"] / (powers["beta_abs"] + 1e-12)

    eps = 1e-12
    powers["log_theta_alpha"] = float(
        np.log((powers["theta_abs"] + eps) / (powers["alpha_abs"] + eps))
    )
    powers["log_theta_beta"] = float(
        np.log((powers["theta_abs"] + eps) / (powers["beta_abs"] + eps))
    )
    return powers

def relative_power_sum(powers: dict[str, float]) -> float:
    return float(sum(powers[f"{b}_rel"] for b in BANDS))