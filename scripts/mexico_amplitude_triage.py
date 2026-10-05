from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QC_IN = PROJECT_ROOT / "results" / "mexico" / "qc_mne_common17.csv"
QC_OUT = PROJECT_ROOT / "results" / "mexico" / "amplitude_triage_common17.csv"

MEAN_ABS_SOFT = 150.0   
MEAN_ABS_HARD = 500.0
P99_SOFT = 500.0
P99_HARD = 2000.0

def flag_row(r: pd.Series) -> str:
    if r["mean_abs_uv"] <= MEAN_ABS_SOFT and r["p99_abs_uv"] <= P99_SOFT:
        return "candidate_clean"
    if r["mean_abs_uv"] >= MEAN_ABS_HARD or r["p99_abs_uv"] >= P99_HARD:
        return "extreme"
    return "borderline"

def main() -> None:
    df = pd.read_csv(QC_IN)
    df = df[df["status"] == "ok"].copy()
    df["amp_flag"] = df.apply(flag_row, axis=1)

    df = df.sort_values(["group", "mean_abs_uv"], ascending=[True, True])
    df.to_csv(QC_OUT, index=False)

    print("=" * 72)
    print("AMPLITUDE TRIAGE (exploratory thresholds)")
    print(f"  mean_abs soft/hard: {MEAN_ABS_SOFT}/{MEAN_ABS_HARD} uV")
    print(f"  p99      soft/hard: {P99_SOFT}/{P99_HARD} uV")
    print("=" * 72)
    print(df.groupby(["group", "amp_flag"]).size().to_string())
    print()
    print("--- lowest mean_abs (5) ---")
    print(
        df.nsmallest(5, "mean_abs_uv")[
            ["stem", "group", "mean_abs_uv", "p99_abs_uv", "std_mean_uv", "amp_flag"]
        ].to_string(index=False)
    )
    print()
    print("--- highest mean_abs (5) ---")
    print(
        df.nlargest(5, "mean_abs_uv")[
            ["stem", "group", "mean_abs_uv", "p99_abs_uv", "std_mean_uv", "amp_flag"]
        ].to_string(index=False)
    )
    print()
    print("--- by group: mean_abs median ---")
    print(df.groupby("group")["mean_abs_uv"].median().to_string())
    print(f"\nSaved → {QC_OUT}")

if __name__ == "__main__":
    main()