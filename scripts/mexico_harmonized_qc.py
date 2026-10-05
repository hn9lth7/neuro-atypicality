from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mexico_duville"
    / "converted"
    / "common_17"
)

QC_PATH = PROJECT_ROOT / "results" / "mexico" / "qc_mne_common17.csv"

CHANNELS = [
    "FP1", "FP2",
    "F3", "F4", "F7", "F8", "Fz",
    "T7", "T8",
    "C3", "C4", "Cz",
    "P3", "P4", "P7", "P8", "Pz",
]

TARGET_SFREQ = 256.0
MIN_DURATION_S = 30.0

UV_TO_V = 1e-6

def load_metadata(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))

def check_one(data_path: Path, group: str) -> dict:
    stem = data_path.name.replace("_data.npy", "")
    meta_path = data_path.with_name(f"{stem}_metadata.json")

    row = {
        "stem": stem,
        "group": group,
        "status": "ok",
        "error": "",
        "n_channels": None,
        "n_times": None,
        "sfreq": None,
        "duration_s": None,
        "finite": False,
        "nan_count": None,
        "inf_count": None,
        "mean_abs_uv": None,
        "std_mean_uv": None,
        "min_uv": None,
        "max_uv": None,
        "p99_abs_uv": None,
        "zero_variance_channels": None,
        "extreme_abs_gt_500_uv": None,
        "mne_rawarray": False,
        "original_n_channels": None,
        "original_sfreq": None,
        "age": None,
        "sex": None,
    }

    try:
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata: {meta_path}")

        meta = load_metadata(meta_path)

        row["original_n_channels"] = meta.get("original_n_channels")
        row["original_sfreq"] = meta.get("original_sfreq")
        row["age"] = meta.get("age")
        row["sex"] = meta.get("sex")

        x = np.load(data_path)

        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={x.shape}")

        n_times, n_channels = x.shape

        row["n_channels"] = n_channels
        row["n_times"] = n_times
        row["sfreq"] = TARGET_SFREQ
        row["duration_s"] = n_times / TARGET_SFREQ

        if n_channels != len(CHANNELS):
            raise ValueError(
                f"Expected {len(CHANNELS)} channels, got {n_channels}"
            )

        if list(meta.get("common_channels", [])) != CHANNELS:
            raise ValueError(
                "Metadata common_channels do not match canonical order"
            )

        if meta.get("final_n_channels") != len(CHANNELS):
            raise ValueError("Metadata final_n_channels mismatch")

        if abs(float(meta.get("final_sfreq")) - TARGET_SFREQ) > 1e-6:
            raise ValueError("Metadata final_sfreq mismatch")

        if x.shape[1] != len(CHANNELS):
            raise ValueError("Data channel dimension mismatch")

        nan_count = int(np.isnan(x).sum())
        inf_count = int(np.isinf(x).sum())
        finite = bool(np.isfinite(x).all())

        row["nan_count"] = nan_count
        row["inf_count"] = inf_count
        row["finite"] = finite

        if not finite:
            raise ValueError(
                f"non-finite values: NaN={nan_count}, Inf={inf_count}"
            )

        duration = row["duration_s"]

        if duration < MIN_DURATION_S:
            raise ValueError(f"Recording too short: {duration:.2f}s")

        abs_x = np.abs(x)

        row["mean_abs_uv"] = float(np.mean(abs_x))
        row["std_mean_uv"] = float(np.mean(np.std(x, axis=0)))
        row["min_uv"] = float(np.min(x))
        row["max_uv"] = float(np.max(x))
        row["p99_abs_uv"] = float(np.percentile(abs_x, 99))

        channel_std = np.std(x, axis=0)
        zero_var = int(np.sum(channel_std <= 1e-12))

        row["zero_variance_channels"] = zero_var

        if zero_var > 0:
            raise ValueError(f"zero-variance channels: {zero_var}")

        extreme = int(np.sum(abs_x > 500.0))
        row["extreme_abs_gt_500_uv"] = extreme

        data_v = (x.T * UV_TO_V).astype(np.float64, copy=False)

        info = mne.create_info(
            ch_names=CHANNELS,
            sfreq=TARGET_SFREQ,
            ch_types=["eeg"] * len(CHANNELS),
        )

        raw = mne.io.RawArray(
            data_v,
            info,
            verbose=False,
        )

        if raw.info["nchan"] != len(CHANNELS):
            raise ValueError("MNE RawArray channel count mismatch")

        if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1e-6:
            raise ValueError("MNE RawArray sfreq mismatch")

        if abs(raw.times[-1] + 1.0 / TARGET_SFREQ - duration) > 1e-6:
            raise ValueError("MNE duration mismatch")

        if not np.isfinite(raw.get_data()).all():
            raise ValueError("MNE RawArray contains non-finite values")

        row["mne_rawarray"] = True

        del raw

    except Exception as e:
        row["status"] = "fail"
        row["error"] = f"{type(e).__name__}: {e}"

    return row

def main() -> None:
    print("=" * 78)
    print("Mexico harmonized MNE QC")
    print("C_common = 17 | sfreq = 256 Hz")
    print("=" * 78)

    rows: list[dict] = []

    for group in ("td", "asd"):
        group_dir = DATA_ROOT / group

        files = sorted(group_dir.glob("*_data.npy"))

        print(f"\n[{group}] {len(files)} files")

        for data_path in files:
            print(f"  → {data_path.name}")

            row = check_one(data_path, group)
            rows.append(row)

            if row["status"] != "ok":
                print(f"    FAIL: {row['error']}")

    qc = pd.DataFrame(rows)

    QC_PATH.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(QC_PATH, index=False)

    print("\n" + "=" * 78)
    print("QC SUMMARY")
    print("=" * 78)

    print(qc.groupby(["group", "status"]).size().to_string())

    ok = qc[qc["status"] == "ok"]

    if len(ok):
        print("\nFinal channels:")
        print(sorted(ok["n_channels"].dropna().unique().tolist()))

        print("\nFinal sfreq:")
        print(sorted(ok["sfreq"].dropna().unique().tolist()))

        print("\nDuration:")
        print(ok["duration_s"].describe().to_string())

        print("\nMean absolute amplitude (uV):")
        print(ok["mean_abs_uv"].describe().to_string())

        print("\nMean channel SD (uV):")
        print(ok["std_mean_uv"].describe().to_string())

        print("\nP99 absolute amplitude (uV):")
        print(ok["p99_abs_uv"].describe().to_string())

        print("\nExtreme samples |x| > 500 uV:")
        print(
            ok.groupby("group")["extreme_abs_gt_500_uv"]
            .agg(["count", "sum", "max"])
            .to_string()
        )

        print("\nZero-variance channels:")
        print(
            ok.groupby("group")["zero_variance_channels"]
            .agg(["count", "max"])
            .to_string()
        )

        print("\nMNE RawArray:")
        print(ok.groupby("group")["mne_rawarray"].sum().to_string())

    print(f"\nSaved → {QC_PATH}")
    print("=" * 78)

if __name__ == "__main__":
    main()