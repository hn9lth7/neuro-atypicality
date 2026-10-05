from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

import biosig
import numpy as np
import pandas as pd
import shutil
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "mexico_duville"
OUT_ROOT = PROJECT_ROOT / "data" / "raw" / "mexico_duville" / "converted"
QC_DIR = PROJECT_ROOT / "results" / "mexico"
QC_PATH = QC_DIR / "qc_rest_ingest.csv"

CHANNELS = [
    "FP1", "FP2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "PO7", "PO3", "PO4", "PO8", "Oz",
]
SFREQ = 256.0
MIN_DURATION_S = 30.0
EXPECTED_N_CH = 32

def part_id_from_stem(stem: str) -> int | None:
    m = re.match(r"Part(\d+)_restEO", stem, re.IGNORECASE)
    return int(m.group(1)) if m else None

def load_age_table(group: str) -> pd.DataFrame:
    hits = list((RAW_ROOT / group).rglob("Age_Gender.xlsx"))
    if not hits:
        raise FileNotFoundError(f"Age_Gender.xlsx not found under {group}")
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
    if "participant" not in df.columns:
        raise ValueError(f"No participant column in {hits[0]}: {list(df.columns)}")
    df["participant"] = pd.to_numeric(df["participant"], errors="coerce").astype("Int64")
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df

def convert_one(gdf_path: Path, out_dir: Path, group: str, age_row: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = gdf_path.stem
    data_path = out_dir / f"{stem}_data.npy"
    meta_path = out_dir / f"{stem}_metadata.json"

    row = {
        "stem": stem,
        "group": group,
        "participant": part_id_from_stem(stem),
        "source_file": str(gdf_path),
        "status": "ok",
        "error": "",
        "sfreq": SFREQ,
        "n_channels": None,
        "n_times": None,
        "duration_s": None,
        "channel_match": None,
        "age": age_row.get("age"),
        "sex": age_row.get("sex"),
        "data_path": str(data_path),
        "meta_path": str(meta_path),
    }

    tmp_path = None
    try:
        tmp_dir = Path(tempfile.gettempdir()) / "nai_gdf"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{group}_{stem}.gdf"
        shutil.copy2(gdf_path, tmp_path)

        data = np.asarray(biosig.data(str(tmp_path)), dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D, got {data.shape}")

        n_times, n_ch = data.shape
        duration_s = float(n_times / SFREQ)

        if n_ch != EXPECTED_N_CH:
            raise ValueError(f"n_channels={n_ch}, expected {EXPECTED_N_CH}")
        if duration_s < MIN_DURATION_S:
            raise ValueError(f"duration too short: {duration_s:.2f}s")

        np.save(data_path, data)

        metadata = {
            "source_file": str(gdf_path.resolve()),
            "format": "GDF 1.25",
            "reader": "BioSig",
            "group": group,
            "participant": row["participant"],
            "age": None if pd.isna(row["age"]) else float(row["age"]),
            "sex": None if pd.isna(row["sex"]) else str(row["sex"]),
            "sampling_frequency_hz": SFREQ,
            "n_times": int(n_times),
            "n_channels": int(n_ch),
            "duration_s": duration_s,
            "data_shape": ["time", "channels"],
            "unit": "uV",
            "channel_names": CHANNELS,
            "montage": "10-20 / extended 10-20",
            "purpose": "Mexico external validation",
            "stem": stem,
        }
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        row.update(
            {
                "n_channels": n_ch,
                "n_times": n_times,
                "duration_s": duration_s,
                "channel_match": True,
                "status": "ok",
            }
        )
    except Exception as e:
        row["status"] = "fail"
        row["error"] = f"{type(e).__name__}: {e}"
        row["channel_match"] = False
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return row

def main() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print("=" * 72)
    print("Mexico batch restEO ingestion + QC")
    print("=" * 72)

    for group in ("td", "asd"):
        age_df = load_age_table(group)
        age_map = {
            int(r["participant"]): {
                "age": r.get("age"),
                "sex": r.get("sex"),
            }
            for _, r in age_df.iterrows()
            if pd.notna(r["participant"])
        }

        gdfs = sorted((RAW_ROOT / group).rglob("*restEO*.gdf"))
        print(f"\n[{group}] found {len(gdfs)} restEO files")

        for gdf in gdfs:
            pid = part_id_from_stem(gdf.stem)
            age_row = age_map.get(pid, {"age": None, "sex": None})
            out_dir = OUT_ROOT / group
            print(f"  → {gdf.name} (participant={pid})")
            row = convert_one(gdf, out_dir, group, age_row)
            rows.append(row)
            if row["status"] != "ok":
                print(f"    FAIL: {row['error']}")

    qc = pd.DataFrame(rows)
    qc.to_csv(QC_PATH, index=False)

    print("\n" + "=" * 72)
    print("QC SUMMARY")
    print("=" * 72)
    print(qc.groupby(["group", "status"]).size().to_string())
    print()
    ok = qc[qc["status"] == "ok"]
    if len(ok):
        print("duration_s:", ok["duration_s"].describe().to_string())
        print("n_channels unique:", sorted(ok["n_channels"].dropna().unique().tolist()))
        print("sfreq unique:", sorted(ok["sfreq"].dropna().unique().tolist()))
        print("missing age:", int(ok["age"].isna().sum()))
    print(f"\nSaved → {QC_PATH}")
    print("=" * 72)

if __name__ == "__main__":
    main()