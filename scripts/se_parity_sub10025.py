from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V11 = PROJECT_ROOT / "results" / "features" / "v11_assembly_sub-10025_run-01.csv"
V032 = PROJECT_ROOT / "results" / "features" / "features_v032.csv"

CANDIDATES = [
    V032,
    PROJECT_ROOT / "results" / "features" / "features_v032_TD.csv",
    PROJECT_ROOT / "results" / "features" / "participants_features_subject_v0.3_clean.csv",
]

SE_KEYS = [
    "alpha_rel",
    "beta_rel",
    "theta_rel",
    "spectral_entropy_mean",
    "log_theta_alpha",
    "log_theta_beta",
]

def main() -> None:
    v11 = pd.read_csv(V11).iloc[0]
    print("v11 (run-01):", V11.name)

    frozen_path = None
    row = None
    for p in CANDIDATES:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        m = df["participant_id"].astype(str).str.contains("10025")
        if not m.any():
            continue
        frozen_path = p
        row = df.loc[m].iloc[0]
        break

    if row is None:
        raise SystemExit("sub-10025 not found in features_v032 / clean subject CSVs")

    print("frozen:", frozen_path.name)
    print("note: frozen is usually SUBJECT mean over runs; v11 is run-01 only")
    print("-" * 70)

    n_ok = 0
    for k in SE_KEYS:
        a = float(v11[k])
        if k not in row.index or pd.isna(row[k]):
            print(f"  {k:28s}  v11={a:.8g}  FROZEN=MISSING")
            continue
        b = float(row[k])
        ad = abs(a - b)
        rd = ad / (abs(b) + 1e-12)
        tag = "OK" if (ad < 1e-6 or rd < 1e-6) else "DIFF(run≠subject?)"
        if ad < 1e-6:
            n_ok += 1
            tag = "OK"
        print(f"  {k:28s}  v11={a:.8g}  frz={b:.8g}  |Δ|={ad:.4g}  rel={rd:.4g}  {tag}")

    print("-" * 70)
    print("If DIFF only: compare run-level SE if you have it, or re-assemble")
    print("subject-level mean of all sub-10025 runs before claiming mismatch.")

if __name__ == "__main__":
    main()