from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import biosig
import numpy as np
import pandas as pd
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "mexico_duville"
OUT_ROOT = PROJECT_ROOT / "data" / "raw" / "mexico_duville" / "converted" / "common_17"
QC_PATH = PROJECT_ROOT / "results" / "mexico" / "qc_harmonized_common17.csv"

C_COMMON = [
    "FP1", "FP2",
    "F3", "F4", "F7", "F8", "Fz",
    "T7", "T8",
    "C3", "C4", "Cz",
    "P3", "P4", "P7", "P8", "Pz",
]
TARGET_SFREQ = 256.0
MIN_DURATION_S = 30.0

def part_id(stem: str) -> int | None:
    m = re.match(r"Part(\d+)_restEO", stem, re.IGNORECASE)
    return int(m.group(1)) if m else None

def load_age(group: str) -> dict[int, dict]:
    hits = list((RAW_ROOT / group).rglob("Age_Gender.xlsx"))
    if not hits:
        return {}
    df = pd.read_excel(hits[0])
    colmap = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if "participant" in cl:
            colmap[c] = "participant"
        elif "age" in cl:
            colmap[c] = "age"
        elif "gender" in cl or "sex" in cl:
            colmap[c] = "sex"
    df = df.rename(columns=colmap)
    df["participant"] = pd.to_numeric(df["participant"], errors="coerce")
    out = {}
    for _, r in df.iterrows():
        if pd.isna(r.get("participant")):
            continue
        out[int(r["participant"])] = {
            "age": float(r["age"]) if "age" in r and pd.notna(r["age"]) else None,
            "sex": str(r["sex"]) if "sex" in r and pd.notna(r["sex"]) else None,
        }
    return out

def read_gdf_short(gdf_path: Path) -> tuple[np.ndarray, list[str], float]:
    tmp_dir = Path(tempfile.gettempdir()) / "nai_gdf"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"harm_{gdf_path.stem}_{abs(hash(str(gdf_path))) % 10**8}.gdf"
    shutil.copy2(gdf_path, tmp)
    try:
        hdr = biosig.header(str(tmp))
        labels = [x.strip() for x in re.findall(r'"Label"\s*:\s*"([^"]*)"', hdr)]
        m = re.search(r'"Samplingrate"\s*:\s*([0-9.]+)', hdr)
        sfreq = float(m.group(1)) if m else float("nan")
        data = np.asarray(biosig.data(str(tmp)), dtype=np.float64)
        return data, labels, sfreq
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

def pick_common(data: np.ndarray, labels: list[str]) -> np.ndarray:
    name_to_idx = {lab: i for i, lab in enumerate(labels)}
    missing = [c for c in C_COMMON if c not in name_to_idx]
    if missing:
        raise ValueError(f"Missing channels: {missing}")
    idx = [name_to_idx[c] for c in C_COMMON]
    return data[:, idx]

def average_reference(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=1, keepdims=True)

def resample_to(x: np.ndarray, sfreq: float, target: float = TARGET_SFREQ) -> np.ndarray:
    if abs(sfreq - target) < 1e-6:
        return x
    n_times, n_ch = x.shape
    n_new = int(round(n_times * target / sfreq))
    y = np.zeros((n_new, n_ch), dtype=np.float64)
    for c in range(n_ch):
        y[:, c] = signal.resample(x[:, c], n_new)
    return y

def process_one(
    gdf_path: Path,
    group: str,
    age_info: dict,
) -> dict:
    stem = gdf_path.stem
    pid = part_id(stem)
    out_dir = OUT_ROOT / group
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{stem}_data.npy"
    meta_path = out_dir / f"{stem}_metadata.json"

    row = {
        "stem": stem,
        "group": group,
        "participant": pid,
        "status": "ok",
        "error": "",
        "age": age_info.get("age"),
        "sex": age_info.get("sex"),
        "original_n_channels": None,
        "original_sfreq": None,
        "final_n_channels": len(C_COMMON),
        "final_sfreq": TARGET_SFREQ,
        "duration_s": None,
        "reference_method": "average_reference_common_17",
    }

    try:
        data, labels, sfreq = read_gdf_short(gdf_path)
        if data.ndim != 2:
            raise ValueError(f"bad shape {data.shape}")

        n_times0, n_ch0 = data.shape
        row["original_n_channels"] = n_ch0
        row["original_sfreq"] = sfreq

        x = pick_common(data, labels)
        x = average_reference(x)
        x = resample_to(x, sfreq, TARGET_SFREQ)

        duration_s = float(x.shape[0] / TARGET_SFREQ)
        if duration_s < MIN_DURATION_S:
            raise ValueError(f"short recording {duration_s:.2f}s")
        if not np.isfinite(x).all():
            raise ValueError("NaN/Inf after harmonization")

        np.save(data_path, x)

        meta = {
            "source_file": str(gdf_path.resolve()),
            "group": group,
            "participant": pid,
            "age": age_info.get("age"),
            "sex": age_info.get("sex"),
            "stem": stem,
            "original_n_channels": n_ch0,
            "original_sfreq": sfreq,
            "original_channel_labels": labels,
            "common_channels": C_COMMON,
            "reference_method": "average_reference_common_17",
            "final_sfreq": TARGET_SFREQ,
            "final_n_channels": len(C_COMMON),
            "n_times": int(x.shape[0]),
            "duration_s": duration_s,
            "data_shape": ["time", "channels"],
            "unit": "uV",
            "purpose": "Mexico harmonized external validation",
        }
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        row["duration_s"] = duration_s
        row["status"] = "ok"
    except Exception as e:
        row["status"] = "fail"
        row["error"] = f"{type(e).__name__}: {e}"

    return row

def main() -> None:
    QC_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print("=" * 72)
    print("Mexico harmonized ingest — C_common=17, AR, 256 Hz")
    print("=" * 72)

    for group in ("td", "asd"):
        age_map = load_age(group)
        gdfs = sorted((RAW_ROOT / group).rglob("*restEO*.gdf"))
        print(f"\n[{group}] {len(gdfs)} files")
        for gdf in gdfs:
            pid = part_id(gdf.stem)
            info = age_map.get(pid or -1, {})
            print(f"  → {gdf.name}")
            row = process_one(gdf, group, info)
            rows.append(row)
            if row["status"] != "ok":
                print(f"    FAIL: {row['error']}")

    qc = pd.DataFrame(rows)
    qc.to_csv(QC_PATH, index=False)

    print("\n" + "=" * 72)
    print("QC SUMMARY")
    print("=" * 72)
    print(qc.groupby(["group", "status"]).size().to_string())
    ok = qc[qc["status"] == "ok"]
    if len(ok):
        print("\nduration_s:\n", ok["duration_s"].describe().to_string())
        print("final_sfreq unique:", sorted(ok["final_sfreq"].unique().tolist()))
        print("final_n_channels unique:", sorted(ok["final_n_channels"].unique().tolist()))
        print("original_sfreq by group:")
        print(ok.groupby("group")["original_sfreq"].agg(["min", "max", "nunique"]))
    print(f"\nSaved → {QC_PATH}")
    print("=" * 72)

if __name__ == "__main__":
    main()