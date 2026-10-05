from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import biosig
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "mexico_duville"
OUT_CSV = PROJECT_ROOT / "results" / "mexico" / "scale_saturation_audit_native.csv"

def part_id(stem: str) -> int | None:
    m = re.match(r"Part(\d+)_restEO", stem, re.IGNORECASE)
    return int(m.group(1)) if m else None

def read_native(gdf_path: Path) -> tuple[str, np.ndarray]:
    tmp_dir = Path(tempfile.gettempdir()) / "nai_gdf"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"sc_{gdf_path.stem}_{abs(hash(str(gdf_path))) % 10**8}.gdf"
    shutil.copy2(gdf_path, tmp)
    try:
        hdr = biosig.header(str(tmp))
        data = np.asarray(biosig.data(str(tmp)), dtype=np.float64)
        return hdr, data
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

def floats(pattern: str, text: str) -> list[float]:
    return [float(x) for x in re.findall(pattern, text)]

def audit_one(gdf_path: Path, group: str) -> dict:
    stem = gdf_path.stem
    row = {
        "stem": stem,
        "group": group,
        "participant": part_id(stem),
        "status": "ok",
        "error": "",
        "sfreq": None,
        "n_channels": None,
        "n_times": None,
        "pmax_min": None,
        "pmax_median": None,
        "pmax_max": None,
        "pmin_min": None,
        "pmin_median": None,
        "pmin_max": None,
        "scaling_min": None,
        "scaling_median": None,
        "scaling_max": None,
        "data_min": None,
        "data_p01": None,
        "data_p50": None,
        "data_p99": None,
        "data_max": None,
        "mean_abs": None,
        "frac_at_pmax": None,
        "frac_at_pmin": None,
        "n_ch_high_sat": None,  
    }

    try:
        hdr, data = read_native(gdf_path)
        if data.ndim != 2:
            raise ValueError(f"shape={data.shape}")

        n_times, n_ch = data.shape
        m = re.search(r'"Samplingrate"\s*:\s*([0-9.]+)', hdr)
        sfreq = float(m.group(1)) if m else float("nan")

        pmax = floats(r'"PhysicalMaximum"\s*:\s*([-0-9.eE+]+)', hdr)
        pmin = floats(r'"PhysicalMinimum"\s*:\s*([-0-9.eE+]+)', hdr)
        scaling = floats(r'"scaling"\s*:\s*([-0-9.eE+]+)', hdr)

        if len(pmax) < n_ch or len(pmin) < n_ch:
            pmax = pmax if pmax else [np.nan] * n_ch
            pmin = pmin if pmin else [np.nan] * n_ch

        pmax_a = np.asarray(pmax[:n_ch], dtype=float)
        pmin_a = np.asarray(pmin[:n_ch], dtype=float)
        sc_a = np.asarray(scaling[:n_ch], dtype=float) if scaling else np.array([np.nan])

        sat_hi = 0
        sat_lo = 0
        n_ch_high = 0
        tol = 1e-6
        for c in range(n_ch):
            ch = data[:, c]
            hi = pmax_a[c] if c < len(pmax_a) and np.isfinite(pmax_a[c]) else np.nan
            lo = pmin_a[c] if c < len(pmin_a) and np.isfinite(pmin_a[c]) else np.nan
            if np.isfinite(hi):
                n_hi = int(np.sum(np.isclose(ch, hi, rtol=0, atol=max(tol, abs(hi) * 1e-9))))
                sat_hi += n_hi
                if n_hi / max(n_times, 1) > 0.01:
                    n_ch_high += 1
            if np.isfinite(lo):
                n_lo = int(np.sum(np.isclose(ch, lo, rtol=0, atol=max(tol, abs(lo) * 1e-9))))
                sat_lo += n_lo

        total = n_times * n_ch
        row.update(
            {
                "sfreq": sfreq,
                "n_channels": n_ch,
                "n_times": n_times,
                "pmax_min": float(np.nanmin(pmax_a)),
                "pmax_median": float(np.nanmedian(pmax_a)),
                "pmax_max": float(np.nanmax(pmax_a)),
                "pmin_min": float(np.nanmin(pmin_a)),
                "pmin_median": float(np.nanmedian(pmin_a)),
                "pmin_max": float(np.nanmax(pmin_a)),
                "scaling_min": float(np.nanmin(sc_a)),
                "scaling_median": float(np.nanmedian(sc_a)),
                "scaling_max": float(np.nanmax(sc_a)),
                "data_min": float(np.min(data)),
                "data_p01": float(np.percentile(data, 1)),
                "data_p50": float(np.percentile(data, 50)),
                "data_p99": float(np.percentile(data, 99)),
                "data_max": float(np.max(data)),
                "mean_abs": float(np.mean(np.abs(data))),
                "frac_at_pmax": float(sat_hi / total) if total else None,
                "frac_at_pmin": float(sat_lo / total) if total else None,
                "n_ch_high_sat": n_ch_high,
                "status": "ok",
            }
        )
    except Exception as e:
        row["status"] = "fail"
        row["error"] = f"{type(e).__name__}: {e}"

    return row

def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print("=" * 72)
    print("Mexico native scale / saturation audit (all restEO)")
    print("=" * 72)

    for group in ("td", "asd"):
        gdfs = sorted((RAW_ROOT / group).rglob("*restEO*.gdf"))
        print(f"[{group}] {len(gdfs)} files")
        for gdf in gdfs:
            print(f"  → {gdf.name}")
            row = audit_one(gdf, group)
            rows.append(row)
            if row["status"] != "ok":
                print(f"    FAIL: {row['error']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    ok = df[df["status"] == "ok"]
    print("\n" + "=" * 72)
    print("SUMMARY BY GROUP")
    print("=" * 72)
    if len(ok):
        for g, sub in ok.groupby("group"):
            print(f"\n--- {g} n={len(sub)} ---")
            for col in [
                "pmax_median",
                "pmax_max",
                "mean_abs",
                "data_p99",
                "data_max",
                "frac_at_pmax",
                "frac_at_pmin",
                "n_ch_high_sat",
            ]:
                print(
                    f"  {col}: "
                    f"med={sub[col].median():.6g}  "
                    f"max={sub[col].max():.6g}"
                )
        print("\nTD vs ASD median mean_abs:")
        print(ok.groupby("group")["mean_abs"].median().to_string())
        print("\nTD vs ASD median frac_at_pmax:")
        print(ok.groupby("group")["frac_at_pmax"].median().to_string())

    print(f"\nSaved → {OUT_CSV}")
    print("=" * 72)

if __name__ == "__main__":
    main()