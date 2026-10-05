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
from nai.features.subject_aggregation import aggregate_se_subject

SUBJECT = "10025"
CHECK = [
    "delta_rel",
    "theta_rel",
    "alpha_rel",
    "beta_rel",
    "gamma_rel",
    "spectral_entropy_mean",
    "log_theta_alpha",
    "log_theta_beta",
]

def resolve_bids_root() -> Path:
    root = PROJECT_ROOT / "data" / "raw" / "ds006780"
    if (root / "ds006780").is_dir():
        root = root / "ds006780"
    return root

def list_runs(root: Path) -> list[str]:
    runs = []
    for p in find_resting_state_files(root):
        if SUBJECT not in p.name:
            continue
        if "run-" in p.name:
            runs.append(p.name.split("run-")[1].split("_")[0])
    return sorted(set(runs), key=int)

def main() -> None:
    root = resolve_bids_root()
    rows = []
    for run in list_runs(root):
        raw = preprocess_minimal(load_raw_bids(root, SUBJECT, run))
        p = compute_band_powers(raw)
        e = compute_spectral_entropy(raw)
        rows.append({"run": run, **p, **e})

    run_df = pd.DataFrame(rows)
    sub = aggregate_se_subject(run_df)

    frz = pd.read_csv(PROJECT_ROOT / "results" / "features" / "features_v032.csv")
    r = frz[frz["participant_id"].astype(str).str.contains("10025")].iloc[0]

    print("SE subject aggregation parity — sub-10025")
    print("-" * 72)
    n_ok = 0
    for k in CHECK:
        a = float(sub[k])
        b = float(r[k])
        ad = abs(a - b)
        ok = ad < 1e-8 or ad / (abs(b) + 1e-12) < 1e-8
        n_ok += int(ok)
        print(f"  {k:24s}  v11={a:.10g}  frz={b:.10g}  |Δ|={ad:.3e}  {'OK' if ok else 'DIFF'}")
    print("-" * 72)
    print(f"RESULT: {n_ok}/{len(CHECK)} OK")
    if n_ok == len(CHECK):
        print("SE subject-level parity CLOSED")

if __name__ == "__main__":
    main()