from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nai.product.validate_input import ValidationResult, _reject

EXPECTED_SFREQ = 512.0
SFREQ_TOL = 0.5
MIN_EEG_CHANNELS = 64
MIN_DURATION_S = 30.0

@dataclass(frozen=True)
class RawValidationConfig:
    expected_sfreq: float = EXPECTED_SFREQ
    sfreq_tol: float = SFREQ_TOL
    min_eeg_channels: int = MIN_EEG_CHANNELS
    min_duration_s: float = MIN_DURATION_S

def validate_raw_object(
    raw: Any,
    *,
    config: RawValidationConfig | None = None,
) -> ValidationResult:
    cfg = config or RawValidationConfig()
    warnings: list[str] = []

    if raw is None:
        return _reject("UNSUPPORTED_FORMAT", detail="raw is None")

    try:
        sfreq = float(raw.info["sfreq"])
    except Exception as e:
        return _reject("UNSUPPORTED_FORMAT", detail=f"sfreq missing: {e}")

    if abs(sfreq - cfg.expected_sfreq) > cfg.sfreq_tol:
        return _reject(
            "UNSUPPORTED_SAMPLING_RATE",
            sfreq=sfreq,
            expected=cfg.expected_sfreq,
        )

    try:
        n_times = int(raw.n_times)
        duration_s = n_times / sfreq
    except Exception as e:
        return _reject("UNSUPPORTED_FORMAT", detail=f"duration: {e}")

    if duration_s < cfg.min_duration_s:
        return _reject(
            "INSUFFICIENT_DURATION",
            duration_s=duration_s,
            min_duration_s=cfg.min_duration_s,
        )

    try:
        import mne

        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        n_eeg = len(picks)
    except Exception:
        try:
            n_eeg = sum(1 for t in raw.get_channel_types() if t == "eeg")
        except Exception as e:
            return _reject("UNSUPPORTED_FORMAT", detail=f"channels: {e}")

    if n_eeg < cfg.min_eeg_channels:
        return _reject(
            "INSUFFICIENT_CHANNELS",
            n_eeg=n_eeg,
            min_eeg_channels=cfg.min_eeg_channels,
        )

    if n_eeg > cfg.min_eeg_channels:
        warnings.append(
            f"EXTRA_CHANNELS: n_eeg={n_eeg} (expected ≥ {cfg.min_eeg_channels})"
        )

    return ValidationResult(
        status="PASS",
        reason=None,
        details={
            "sfreq": sfreq,
            "duration_s": duration_s,
            "n_eeg": n_eeg,
        },
        warnings=warnings,
    )

def validate_raw_path(
    path: str | Path,
    *,
    config: RawValidationConfig | None = None,
) -> ValidationResult:
    path = Path(path)
    if not path.exists():
        return _reject("UNSUPPORTED_FORMAT", path=str(path), detail="not found")

    try:
        from mne_bids import BIDSPath, read_raw_bids
        import mne

        suffix = path.suffix.lower()
        if suffix in {".bdf", ".edf", ".fif", ".vhdr"}:
            raw = mne.io.read_raw(path, preload=False, verbose=False)
        else:
            return _reject(
                "UNSUPPORTED_FORMAT",
                path=str(path),
                detail=f"unsupported suffix {suffix}",
            )
    except Exception as e:
        return _reject("UNSUPPORTED_FORMAT", path=str(path), detail=str(e))

    return validate_raw_object(raw, config=config)