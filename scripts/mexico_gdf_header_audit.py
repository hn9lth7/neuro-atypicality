from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import biosig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "mexico_duville"

def header_text(gdf_path: Path) -> str:
    tmp_dir = Path(tempfile.gettempdir()) / "nai_gdf"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"audit_{gdf_path.stem}.gdf"
    shutil.copy2(gdf_path, tmp_path)
    try:
        return biosig.header(str(tmp_path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

def extract_labels(hdr: str) -> list[str]:
    labels = re.findall(r'"Label"\s*:\s*"([^"]*)"', hdr)
    return [lab.strip() for lab in labels]

def extract_sfreq(hdr: str) -> float | None:
    m = re.search(r'"Samplingrate"\s*:\s*([0-9.]+)', hdr)
    return float(m.group(1)) if m else None

def print_audit(tag: str, gdf_path: Path) -> list[str]:
    print()
    print("=" * 78)
    print(f"{tag}: {gdf_path.name}")
    print(gdf_path)
    print("=" * 78)

    hdr = header_text(gdf_path)
    labels = extract_labels(hdr)
    sfreq = extract_sfreq(hdr)

    print(f"Samplingrate: {sfreq}")
    print(f"n_channels  : {len(labels)}")
    print()
    print("CHANNELS:")
    print("-" * 40)
    for i, lab in enumerate(labels, start=1):
        print(f"{i:2d}: {lab}")
    print("-" * 40)
    return labels

def main() -> None:
    td = next((RAW_ROOT / "td").rglob("Part01_restEO.gdf"))
    asd = next((RAW_ROOT / "asd").rglob("Part01_restEO.gdf"))

    td_labs = print_audit("TD", td)
    asd_labs = print_audit("ASD", asd)

    set_td = set(td_labs)
    set_asd = set(asd_labs)
    common = sorted(set_td & set_asd)
    only_td = sorted(set_td - set_asd)
    only_asd = sorted(set_asd - set_td)

    print()
    print("=" * 78)
    print("INTERSECTION")
    print("=" * 78)
    print(f"|C_TD|      = {len(td_labs)}")
    print(f"|C_ASD|     = {len(asd_labs)}")
    print(f"|common|    = {len(common)}")
    print(f"common      = {common}")
    print(f"only TD     = {only_td}")
    print(f"only ASD    = {only_asd}")
    print("=" * 78)

if __name__ == "__main__":
    main()