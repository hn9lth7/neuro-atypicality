from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"

SRC = FEATURES_DIR / "dynamic_rest_v0.6.csv"
OUT = FEATURES_DIR / "dynamic_subject_v0.6.csv"

CORE = [
    "mean_delta",
    "cv_delta",
    "mean_degree_cv",
    "temporal_cv_degree_cv",
]

BANDS = ["theta", "alpha", "beta", "gamma"]

def main():
    print("=" * 70)
    print("NAI v0.6 — Subject-level Dynamic Aggregation")
    print("=" * 70)

    df = pd.read_csv(SRC)
    print(f"Run-level rows : {len(df)}")
    print(f"Unique subjects: {df['participant_id'].nunique()}")

    meta = (
        df.groupby("participant_id")[["age", "sex", "group"]]
        .first()
        .reset_index()
    )

    agg = (
        df.groupby(["participant_id", "band"])[CORE]
        .mean()
        .reset_index()
    )

    n_runs = (
        df.groupby(["participant_id", "band"])["run"]
        .nunique()
        .reset_index(name="n_runs")
    )

    subject_band = agg.merge(n_runs, on=["participant_id", "band"])
    subject_band = subject_band.merge(meta, on="participant_id", how="left")

    parts = []
    for metric in CORE:
        pivot = subject_band.pivot(
            index="participant_id", columns="band", values=metric
        )
        pivot.columns = [f"{metric}_{b}" for b in pivot.columns]
        parts.append(pivot)

    wide = pd.concat(parts, axis=1)
    n_runs_subj = (
        subject_band.groupby("participant_id")["n_runs"].max().rename("n_runs")
    )
    wide = wide.join(n_runs_subj)
    wide = wide.join(meta.set_index("participant_id"))
    wide = wide.reset_index()

    front = ["participant_id", "age", "sex", "group", "n_runs"]
    metric_cols = sorted([c for c in wide.columns if c not in front])
    wide = wide[front + metric_cols]

    print(f"Subject-level rows: {len(wide)}")
    print(wide["group"].value_counts(dropna=False))
    print("\nFeature columns:", len(metric_cols))
    print(metric_cols[:8], "...")

    wide.to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}")
    print("=" * 70)

if __name__ == "__main__":
    main()