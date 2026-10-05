from __future__ import annotations

import argparse
import json
from pathlib import Path

import biosig
import numpy as np


CHANNELS = [
    "FP1", "FP2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "PO7", "PO3", "PO4", "PO8", "Oz",
]

SFREQ = 256.0
MIN_DURATION_S = 30.0

def convert_one(gdf_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GDF   :", gdf_path)
    print("OUTPUT:", output_dir)
    print("=" * 70)

    if not gdf_path.exists():
        raise FileNotFoundError(gdf_path)

    print("Reading GDF with BioSig...")
    data = np.asarray(biosig.data(str(gdf_path)), dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape={data.shape}")

    n_times, n_channels = data.shape
    duration_s = float(n_times / SFREQ)

    print("shape      :", data.shape)
    print("sfreq      :", SFREQ)
    print("duration_s :", duration_s)
    print("channels   :", n_channels)

    if n_channels != len(CHANNELS):
        raise ValueError(
            f"Expected {len(CHANNELS)} channels, got {n_channels}"
        )

    if duration_s < MIN_DURATION_S:
        raise ValueError(
            f"Recording unexpectedly short: {duration_s:.2f} s"
        )

    subject_name = gdf_path.stem  
    data_path = output_dir / f"{subject_name}_data.npy"
    meta_path = output_dir / f"{subject_name}_metadata.json"

    np.save(data_path, data)

    metadata = {
        "source_file": str(gdf_path.resolve()),
        "format": "GDF 1.25",
        "reader": "BioSig",
        "sampling_frequency_hz": SFREQ,
        "n_times": int(n_times),
        "n_channels": int(n_channels),
        "duration_s": duration_s,
        "data_shape": ["time", "channels"],
        "unit": "uV",
        "channel_names": CHANNELS,
        "montage": "10-20 / extended 10-20",
        "purpose": "Mexico external validation",
        "stem": subject_name,
    }

    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("SAVED:")
    print("  data:", data_path)
    print("  meta:", meta_path)
    print()
    print("VALIDATION:")
    print("  shape      =", data.shape)
    print("  duration   =", round(duration_s, 3), "s")
    print("  n_channels =", n_channels)
    print("  sfreq      =", SFREQ)
    print("  unit       = uV")
    print("=" * 70)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mexico GDF restEO → npy + metadata.json (BioSig / .venv_gdf)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to PartXX_restEO.gdf",
    )
    parser.add_argument(
        "--output",
        default="data/raw/mexico_duville/td/_converted",
        help="Output directory for .npy and .json",
    )
    args = parser.parse_args()

    convert_one(Path(args.input), Path(args.output))

if __name__ == "__main__":
    main()