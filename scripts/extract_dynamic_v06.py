from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.io.bids import find_resting_state_files, load_raw_bids
from nai.io.metadata import load_participants
from nai.preprocessing.pipeline import preprocess_minimal
from nai.connectivity.dynamic import compute_windowed_plv
from nai.dynamics.transitions import transition_series, dynamic_summary
from nai.graph.metrics import degree_cv

BIDS_ROOT = PROJECT_ROOT / "data" / "raw" / "ds006780"
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
OUT = FEATURES_DIR / "dynamic_rest_v0.6.csv"

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

WINDOW_SEC = 10.0
STEP_SEC = 5.0

def parse_subject_run(path: Path) -> tuple[str, str]:
    subject = run = None
    for part in path.name.split("_"):
        if part.startswith("sub-"):
            subject = part.replace("sub-", "")
        elif part.startswith("run-"):
            run = part.replace("run-", "")
    if subject is None or run is None:
        raise ValueError(f"Cannot parse: {path.name}")
    return subject, run

def load_existing() -> pd.DataFrame:
    if not OUT.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(OUT)
    except Exception:
        return pd.DataFrame()

def is_done(existing: pd.DataFrame, subject: str, run: str) -> bool:
    if existing.empty:
        return False
    sub = existing[
        (existing["participant_id"] == f"sub-{subject}")
        & (existing["run"] == run)
    ]
    return set(sub["band"].astype(str)) >= set(BANDS.keys())

def save_rows(rows: list[dict]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_csv(OUT)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["participant_id", "run", "band"], keep="last"
        )
    else:
        combined = new
    combined.to_csv(OUT, index=False)

def main():
    print("=" * 72)
    print("NAI v0.6 — Full Dynamic Extraction")
    print(f"window={WINDOW_SEC}s  step={STEP_SEC}s")
    print("=" * 72)

    participants = load_participants(BIDS_ROOT)
    participants["participant_id"] = participants["participant_id"].astype(str)
    participants = participants.set_index("participant_id")

    files = find_resting_state_files(BIDS_ROOT)
    print(f"Resting-state files: {len(files)}")

    existing = load_existing()
    processed = skipped = failed = 0

    for path in tqdm(files, desc="Runs"):
        subject, run = parse_subject_run(path)
        pid = f"sub-{subject}"

        if is_done(existing, subject, run):
            skipped += 1
            continue

        try:
            raw = load_raw_bids(BIDS_ROOT, subject=subject, run=run)
            raw_clean = preprocess_minimal(raw)
            data = raw_clean.get_data()
            sfreq = float(raw_clean.info["sfreq"])
            duration = float(raw_clean.times[-1])

            age = sex = group = np.nan
            if pid in participants.index:
                p = participants.loc[pid]
                age = p.get("age", np.nan)
                sex = p.get("sex", np.nan)
                group = p.get("group", np.nan)

            run_rows = []

            for band, (fmin, fmax) in BANDS.items():
                matrices = compute_windowed_plv(
                    data, sfreq, fmin, fmax,
                    window_sec=WINDOW_SEC, step_sec=STEP_SEC
                )
                n_win = len(matrices)

                deltas = transition_series(matrices)
                dsum = dynamic_summary(deltas)

                deg_cv_series = np.array([degree_cv(W) for W in matrices])
                mean_deg_cv = float(deg_cv_series.mean())
                temporal_cv_deg_cv = float(
                    deg_cv_series.std() / (abs(deg_cv_series.mean()) + 1e-12)
                )

                rec = {
                    "participant_id": pid,
                    "run": run,
                    "band": band,
                    "age": age,
                    "sex": sex,
                    "group": group,
                    "duration_s": duration,
                    "n_windows": n_win,
                    "mean_delta": dsum["mean_delta"],
                    "cv_delta": dsum["cv_delta"],
                    "mean_degree_cv": mean_deg_cv,
                    "temporal_cv_degree_cv": temporal_cv_deg_cv,
                    "std_delta": dsum["std_delta"],
                    "max_delta": dsum["max_delta"],
                    "temporal_entropy": dsum["temporal_entropy"],
                }
                run_rows.append(rec)

            save_rows(run_rows)
            processed += 1

        except Exception as e:
            failed += 1
            print(f"\n  ERROR {pid} run-{run}: {type(e).__name__}: {e}")
            continue

    print("\n" + "=" * 72)
    print("EXTRACTION COMPLETE")
    print("=" * 72)
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")

    if OUT.exists():
        df = pd.read_csv(OUT)
        print(f"\nOutput rows     : {len(df)}")
        print(f"Unique subjects : {df['participant_id'].nunique()}")
        print(f"Unique runs     : {df[['participant_id','run']].drop_duplicates().shape[0]}")
        print("\nRows per band:")
        print(df["band"].value_counts().sort_index().to_string())
        print("\nGroup × band:")
        print(df.groupby(["group", "band"]).size().unstack(fill_value=0).to_string())
        print("\nCore metric ranges:")
        for m in ["mean_delta", "cv_delta", "mean_degree_cv", "temporal_cv_degree_cv"]:
            print(f"  {m:25s}  min={df[m].min():.4f}  max={df[m].max():.4f}  mean={df[m].mean():.4f}")

    print(f"\nSaved → {OUT}")
    print("=" * 72)

if __name__ == "__main__":
    main()