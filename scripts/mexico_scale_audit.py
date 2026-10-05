from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import biosig
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "mexico_duville"

def load(gdf: Path):
    tmp = Path(tempfile.gettempdir()) / "nai_gdf" / gdf.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gdf, tmp)
    try:
        hdr = biosig.header(str(tmp))
        data = np.asarray(biosig.data(str(tmp)), dtype=np.float64)
        return hdr, data
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

def summarize(tag: str, gdf: Path) -> None:
    hdr, data = load(gdf)
    labels = [x.strip() for x in re.findall(r'"Label"\s*:\s*"([^"]*)"', hdr)]
    pmax = [float(x) for x in re.findall(r'"PhysicalMaximum"\s*:\s*([-0-9.eE]+)', hdr)]
    scaling = [float(x) for x in re.findall(r'"scaling"\s*:\s*([-0-9.eE]+)', hdr)]
    sfreq = re.search(r'"Samplingrate"\s*:\s*([0-9.]+)', hdr)

    print("=" * 72)
    print(tag, gdf.name)
    print("sfreq", sfreq.group(1) if sfreq else None)
    print("shape", data.shape, "labels", len(labels))
    print("PhysicalMaximum: min/med/max", min(pmax), np.median(pmax), max(pmax))
    print("scaling:         min/med/max", min(scaling), np.median(scaling), max(scaling))
    print("data quantiles:", {
        "min": float(np.min(data)),
        "p01": float(np.percentile(data, 1)),
        "p50": float(np.percentile(data, 50)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
        "mean_abs": float(np.mean(np.abs(data))),
    })
    if "FP1" in labels:
        i = labels.index("FP1")
        ch = data[:, i]
        print("FP1 mean_abs", float(np.mean(np.abs(ch))), "std", float(np.std(ch)))

def main() -> None:
    td = next((RAW / "td").rglob("Part10_restEO.gdf"))
    asd = next((RAW / "asd").rglob("Part12_restEO.gdf"))
    summarize("TD clean-ish", td)
    summarize("ASD extreme", asd)

if __name__ == "__main__":
    main()