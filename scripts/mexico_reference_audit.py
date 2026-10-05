from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import biosig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "mexico_duville"

C_COMMON = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "Fz",
    "T7", "T8", "C3", "C4", "Cz",
    "P3", "P4", "P7", "P8", "Pz",
]

def header_text(gdf_path: Path) -> str:
    tmp_dir = Path(tempfile.gettempdir()) / "nai_gdf"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"ref_{gdf_path.stem}.gdf"
    shutil.copy2(gdf_path, tmp)
    try:
        return biosig.header(str(tmp))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

def labels(hdr: str) -> list[str]:
    return [x.strip() for x in re.findall(r'"Label"\s*:\s*"([^"]*)"', hdr)]

def units(hdr: str) -> list[str]:
    u = re.findall(r'"PhysicalUnit"\s*:\s*"([^"]*)"', hdr)
    if not u:
        u = re.findall(r'"PhysDim"\s*:\s*"([^"]*)"', hdr)
    return [x.strip() for x in u]

def sfreq(hdr: str) -> float | None:
    m = re.search(r'"Samplingrate"\s*:\s*([0-9.]+)', hdr)
    return float(m.group(1)) if m else None

def audit(tag: str, path: Path) -> None:
    hdr = header_text(path)
    labs = labels(hdr)
    un = units(hdr)
    fs = sfreq(hdr)

    print("=" * 72)
    print(f"{tag}: {path.name}")
    print("=" * 72)
    print(f"sfreq        : {fs}")
    print(f"n_channels   : {len(labs)}")
    print(f"labels       : {labs}")
    print(f"units unique : {sorted(set(un))}")
    print(f"has A1/A2    : {('A1' in labs) or ('A2' in labs)}")
    print(f"has O1/O2    : {('O1' in labs) or ('O2' in labs)}")
    print(f"C_common all present: {all(c in labs for c in C_COMMON)}")
    missing = [c for c in C_COMMON if c not in labs]
    print(f"C_common missing    : {missing}")
    low = hdr.lower()
    for key in ("reference", "ref.", "common mode", "cms", "drl", "mastoid"):
        if key in low:
            print(f"header contains keyword: {key!r}")
    print()

def main() -> None:
    td = next((RAW_ROOT / "td").rglob("Part01_restEO.gdf"))
    asd = next((RAW_ROOT / "asd").rglob("Part01_restEO.gdf"))
    audit("TD", td)
    audit("ASD", asd)
    print("NOTE: explicit Reference field is often absent in GDF headers.")
    print("Harmonization plan: pick C_common → average ref on 17 ch →")
    print("resample ASD 500→256 Hz → then features.")
    print("Do not put A1/A2 into the feature channel set.")

if __name__ == "__main__":
    main()