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
from nai.connectivity.phase import bandpass_filter, compute_plv
from nai.graph.metrics import (
    mean_degree,
    degree_cv,
    weighted_clustering,
    global_efficiency,
    mean_path_length,
)
from nai.graph.spectral import (
    algebraic_connectivity,
    laplacian_entropy,
)

BIDS_ROOT = PROJECT_ROOT / "data" / "raw" / "ds006780"
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

OUT = FEATURES_DIR / "connectivity_graph_rest_v0.5.csv"

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

def parse_subject_run(path: Path) -> tuple[str, str]:
    subject = None
    run = None

    for part in path.name.split("_"):
        if part.startswith("sub-"):
            subject = part.replace("sub-", "")
        elif part.startswith("run-"):
            run = part.replace("run-", "")

    if subject is None:
        raise ValueError(f"Cannot parse subject from: {path.name}")
    if run is None:
        raise ValueError(f"Cannot parse run from: {path.name}")

    return subject, run

def load_existing_output() -> pd.DataFrame:
    if not OUT.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(OUT)
        print(f"Existing output found: {len(df)} rows")
        return df
    except Exception as exc:
        print(f"Warning: could not read existing output: {exc}")
        return pd.DataFrame()

def is_already_processed(existing: pd.DataFrame, subject: str, run: str) -> bool:
    if existing.empty:
        return False

    required = existing[
        (existing["participant_id"] == f"sub-{subject}")
        & (existing["run"] == run)
    ]
    return set(required["band"].astype(str)) >= set(BANDS.keys())

def save_results(rows: list[dict]) -> None:
    if not rows:
        return

    new_df = pd.DataFrame(rows)

    if OUT.exists():
        old_df = pd.read_csv(OUT)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["participant_id", "run", "band"],
            keep="last",
        )
    else:
        combined = new_df

    combined.to_csv(OUT, index=False)

def main():
    print("=" * 78)
    print("NAI v0.5 — Full Connectivity + Graph Extraction")
    print("=" * 78)

    participants = load_participants(BIDS_ROOT)
    participants["participant_id"] = participants["participant_id"].astype(str)
    participants = participants.set_index("participant_id")

    files = find_resting_state_files(BIDS_ROOT)
    print(f"Resting-state files found: {len(files)}")

    existing = load_existing_output()

    processed_runs = 0
    skipped_runs = 0
    failed_runs = 0

    for idx, path in enumerate(tqdm(files, desc="Processing runs"), start=1):
        subject, run = parse_subject_run(path)
        participant_id = f"sub-{subject}"

        print(f"\n[{idx}/{len(files)}] {participant_id} / run-{run}")

        if is_already_processed(existing, subject, run):
            print("  SKIP — already processed")
            skipped_runs += 1
            continue

        try:
            raw = load_raw_bids(BIDS_ROOT, subject=subject, run=run)

            raw_clean = preprocess_minimal(raw)
            data = raw_clean.get_data()
            sfreq = float(raw_clean.info["sfreq"])
            n_channels, n_times = data.shape
            duration_s = float(raw_clean.times[-1] - raw_clean.times[0])

            print(f"  EEG       : {n_channels} channels")
            print(f"  Duration  : {duration_s:.1f} s")
            print(f"  sfreq     : {sfreq:.1f} Hz")

            age = np.nan
            sex = np.nan
            group = np.nan
            if participant_id in participants.index:
                p = participants.loc[participant_id]
                age = p.get("age", np.nan)
                sex = p.get("sex", np.nan)
                group = p.get("group", np.nan)

            run_rows = []

            for band, (fmin, fmax) in BANDS.items():
                print(f"  {band.upper():5s} {fmin:.0f}–{fmax:.0f} Hz ...", end=" ", flush=True)

                data_bp = bandpass_filter(data, sfreq, fmin, fmax)

                W = compute_plv(data_bp)

                W_no_diag = W.copy()
                np.fill_diagonal(W_no_diag, 0.0)

                plv_mean = float(W_no_diag.sum() / (n_channels * (n_channels - 1)))
                plv_median = float(np.median(W_no_diag[np.triu_indices(n_channels, k=1)]))

                md = mean_degree(W)
                dcv = degree_cv(W)
                clustering = weighted_clustering(W)
                efficiency = global_efficiency(W)
                path_length = mean_path_length(W)
                lambda2 = algebraic_connectivity(W)
                lap_entropy = laplacian_entropy(W)

                rec = {
                    "participant_id": participant_id,
                    "run": run,
                    "band": band,
                    "age": age,
                    "sex": sex,
                    "group": group,
                    "n_channels": n_channels,
                    "duration_s": duration_s,
                    "plv_mean": plv_mean,
                    "plv_median": plv_median,
                    "mean_degree": md,
                    "degree_cv": dcv,
                    "clustering": clustering,
                    "global_efficiency": efficiency,
                    "mean_path_length": path_length,
                    "lambda2": lambda2,
                    "laplacian_entropy": lap_entropy,
                }
                run_rows.append(rec)

                print(f"PLV={plv_mean:.3f}  λ₂={lambda2:.3f}")

            save_results(run_rows)
            processed_runs += 1
            print(f"  saved {len(run_rows)} band rows")

        except Exception as exc:
            failed_runs += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            continue

    print("\n" + "=" * 78)
    print("EXTRACTION COMPLETE")
    print("=" * 78)
    print(f"Processed runs : {processed_runs}")
    print(f"Skipped runs   : {skipped_runs}")
    print(f"Failed runs    : {failed_runs}")

    if OUT.exists():
        final_df = pd.read_csv(OUT)
        print(f"\nOutput rows    : {len(final_df)}")
        print(f"Unique subjects: {final_df['participant_id'].nunique()}")
        print(f"Unique runs    : {final_df[['participant_id', 'run']].drop_duplicates().shape[0]}")

        print("\nRows per band:")
        print(final_df["band"].value_counts().sort_index().to_string())

        print("\nGroup × band:")
        print(final_df.groupby(["group", "band"]).size().unstack(fill_value=0).to_string())

        print("\nBasic metric ranges:")
        for metric in [
            "plv_mean", "mean_degree", "degree_cv", "clustering",
            "global_efficiency", "mean_path_length", "lambda2", "laplacian_entropy"
        ]:
            print(
                f"  {metric:22s} "
                f"min={final_df[metric].min():.4f} "
                f"max={final_df[metric].max():.4f} "
                f"mean={final_df[metric].mean():.4f}"
            )

    print(f"\nSaved → {OUT}")
    print("=" * 78)

if __name__ == "__main__":
    main()