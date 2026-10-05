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

SUBJECT = "10025"

def resolve_bids_root() -> Path:
    root = PROJECT_ROOT / "data" / "raw" / "ds006780"
    if (root / "ds006780").is_dir():
        root = root / "ds006780"
    return root

def list_runs(root: Path) -> list[str]:
    files = find_resting_state_files(root)
    runs = []
    for p in files:
        if SUBJECT not in p.name:
            continue
        if "run-" in p.name:
            runs.append(p.name.split("run-")[1].split("_")[0])
    return sorted(set(runs), key=int)

def main() -> None:
    root = resolve_bids_root()
    runs = list_runs(root)
    print("Runs:", runs)

    rows = []
    for run in runs:
        raw = preprocess_minimal(load_raw_bids(root, SUBJECT, run))
        p = compute_band_powers(raw)
        e = compute_spectral_entropy(raw)
        row = {"run": run, **{k: float(v) for k, v in p.items()}, **e}
        abs_keys = [k for k in p if k.endswith("_abs")]
        rel_keys = [k for k in p if k.endswith("_rel")]
        row["sum_abs"] = float(sum(p[k] for k in abs_keys))
        row["sum_rel"] = float(sum(p[k] for k in rel_keys))
        eps = 1e-12
        if "theta_abs" in p and "alpha_abs" in p:
            row["log_from_abs"] = float(
                np.log((p["theta_abs"] + eps) / (p["alpha_abs"] + eps))
            )
        rows.append(row)
        print(f"\nrun-{run} keys: {sorted(p.keys())}")
        print(f"  sum_rel={row['sum_rel']:.6f}  (expect ~1.0)")
        for k in sorted(p.keys()):
            print(f"  {k:20s} {p[k]:.8g}")

    df = pd.DataFrame(rows)
    print("\n=== AGGREGATION VARIANTS ===")
    for k in ["alpha_rel", "beta_rel", "theta_rel"]:
        if k in df.columns:
            print(f"mean({k}) = {df[k].mean():.8g}")

    if set(["theta_abs", "alpha_abs", "beta_abs"]).issubset(df.columns):
        ma = df[["delta_abs", "theta_abs", "alpha_abs", "beta_abs", "gamma_abs"]].mean()
        cols = [c for c in df.columns if c.endswith("_abs")]
        ma = df[cols].mean()
        total = ma.sum()
        print("mean_abs then rel:")
        for c in cols:
            print(f"  {c.replace('_abs','_rel')}_via_mean_abs = {ma[c]/total:.8g}")
        eps = 1e-12
        print(
            "  log_theta_alpha via mean_abs =",
            np.log((ma["theta_abs"] + eps) / (ma["alpha_abs"] + eps)),
        )
        print(
            "  mean(log_theta_alpha) =",
            df["log_from_abs"].mean() if "log_from_abs" in df else "n/a",
        )

    frz_path = PROJECT_ROOT / "results" / "features" / "features_v032.csv"
    frz = pd.read_csv(frz_path)
    r = frz[frz["participant_id"].astype(str).str.contains("10025")].iloc[0]
    print("\n=== FROZEN features_v032 sub-10025 ===")
    for k in r.index:
        if any(s in k for s in ("alpha", "beta", "theta", "delta", "gamma", "entropy", "log_")):
            try:
                print(f"  {k:28s} {float(r[k]):.8g}")
            except Exception:
                pass

    for name in [
        "participants_features_subject_v0.2.csv",
        "participants_features_subject_v0.3_clean.csv",
        "participants_features_rest_v0.1.csv",
    ]:
        p = PROJECT_ROOT / "results" / "features" / name
        if not p.exists():
            continue
        d = pd.read_csv(p)
        m = d["participant_id"].astype(str).str.contains("10025")
        if not m.any():
            continue
        print(f"\n=== {name} (sub-10025) ===")
        rr = d.loc[m]
        if "run" in rr.columns:
            print(rr[["run"] + [c for c in rr.columns if "rel" in c or "entropy" in c or "log_" in c]].to_string(index=False))
        else:
            cols = [c for c in rr.columns if any(x in c for x in ("rel", "entropy", "log_", "abs"))]
            print(rr[cols].iloc[0].to_string())

if __name__ == "__main__":
    main()