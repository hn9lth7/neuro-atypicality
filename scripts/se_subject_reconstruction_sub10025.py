from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import find_resting_state_files, load_raw_bids
from nai.preprocessing import preprocess_minimal
from nai.spectral.power import compute_band_powers
from nai.spectral.entropy import compute_spectral_entropy
from nai.features.blocks import SE_FEATURES

SUBJECT = "10025"
PARTICIPANT = f"sub-{SUBJECT}"

def resolve_bids_root() -> Path:
    root = PROJECT_ROOT / "data" / "raw" / "ds006780"
    if (root / "ds006780").is_dir():
        root = root / "ds006780"
    return root

def list_runs(bids_root: Path) -> list[str]:
    files = find_resting_state_files(bids_root)
    runs = []
    for p in files:
        if PARTICIPANT not in str(p) and SUBJECT not in p.name:
            continue
        name = p.name
        if "run-" not in name:
            continue
        run = name.split("run-")[1].split("_")[0]
        runs.append(run)
    return sorted(set(runs), key=lambda x: int(x))

def se_for_run(bids_root: Path, run: str) -> dict[str, float]:
    raw = load_raw_bids(bids_root, subject=SUBJECT, run=run)
    raw = preprocess_minimal(raw)
    powers = compute_band_powers(raw)
    ent = compute_spectral_entropy(raw)
    feat = {**powers, **ent}
    out = {k: float(feat[k]) for k in SE_FEATURES}
    eps = 1e-12
    if "theta_abs" in powers and "alpha_abs" in powers:
        out["_check_log_theta_alpha"] = float(
            np.log((powers["theta_abs"] + eps) / (powers["alpha_abs"] + eps))
        )
    return out

def main() -> None:
    bids_root = resolve_bids_root()
    runs = list_runs(bids_root)
    print("=" * 78)
    print(f"SE subject reconstruction — {PARTICIPANT}")
    print(f"BIDS root: {bids_root}")
    print(f"Runs found: {runs}")
    print("=" * 78)

    if not runs:
        raise SystemExit("No resting runs found for subject")

    rows = []
    for run in runs:
        print(f"\n--- run-{run} ---")
        try:
            se = se_for_run(bids_root, run)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        row = {"participant_id": PARTICIPANT, "run": run}
        row.update({k: se[k] for k in SE_FEATURES})
        if "_check_log_theta_alpha" in se:
            row["check_log_ta"] = se["_check_log_theta_alpha"]
            d = abs(se["log_theta_alpha"] - se["_check_log_theta_alpha"])
            print(f"  log_theta_alpha check |Δ|={d:.3e}")
        rows.append(row)
        for k in SE_FEATURES:
            print(f"  {k:28s} {se[k]:.8g}")

    if not rows:
        raise SystemExit("No successful runs")

    df = pd.DataFrame(rows)
    out_runs = (
        PROJECT_ROOT
        / "results"
        / "features"
        / "v11_se_per_run_sub-10025.csv"
    )
    out_runs.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_runs, index=False)
    print(f"\nSaved per-run → {out_runs}")

    mean = df[SE_FEATURES].mean(axis=0)
    print("\n" + "=" * 78)
    print("SUBJECT MEAN (v1.1 over successful runs)")
    print("=" * 78)
    for k in SE_FEATURES:
        print(f"  {k:28s} {mean[k]:.8g}")

    v032_path = PROJECT_ROOT / "results" / "features" / "features_v032.csv"
    print("\n" + "=" * 78)
    print(f"COMPARE mean vs {v032_path.name}")
    print("=" * 78)
    frz = pd.read_csv(v032_path)
    m = frz["participant_id"].astype(str).str.contains("10025")
    if not m.any():
        print("sub-10025 not in features_v032")
        return
    r = frz.loc[m].iloc[0]
    n_ok = 0
    for k in SE_FEATURES:
        a = float(mean[k])
        b = float(r[k]) if k in r.index and pd.notna(r[k]) else float("nan")
        if not np.isfinite(b):
            print(f"  {k:28s}  v11_mean={a:.8g}  FROZEN=MISSING")
            continue
        ad = abs(a - b)
        rd = ad / (abs(b) + 1e-12)
        ok = ad < 1e-5 or rd < 1e-5
        if ok:
            n_ok += 1
        tag = "OK" if ok else "DIFF"
        print(
            f"  {k:28s}  v11_mean={a:.8g}  frz={b:.8g}  "
            f"|Δ|={ad:.4g}  rel={rd:.4g}  {tag}"
        )
    print(f"\nSE subject parity: {n_ok}/6 OK (strict)")
    print("Also inspect per-run table: sign of log_theta_alpha across runs.")

if __name__ == "__main__":
    main()