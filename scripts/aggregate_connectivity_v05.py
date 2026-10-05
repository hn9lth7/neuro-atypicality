from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"

SRC = FEATURES_DIR / "connectivity_graph_rest_v0.5.csv"
OUT = FEATURES_DIR / "connectivity_graph_subject_v0.5.csv"

METRICS = [
    "plv_mean",
    "plv_median",
    "mean_degree",
    "degree_cv",
    "clustering",
    "global_efficiency",
    "mean_path_length",
    "lambda2",
    "laplacian_entropy",
]

BANDS = ["theta", "alpha", "beta", "gamma"]

def main():
    print("=" * 70)
    print("NAI v0.5 — Subject-level Connectivity + Graph Aggregation")
    print("=" * 70)

    df = pd.read_csv(SRC)
    print(f"Run-level rows : {len(df)}")
    print(f"Unique subjects: {df['participant_id'].nunique()}")
    print(f"Unique runs    : {df[['participant_id','run']].drop_duplicates().shape[0]}")

    group_cols = ["participant_id", "band"]
    meta_cols = ["age", "sex", "group"]

    meta = (
        df.groupby("participant_id")[meta_cols]
        .first()
        .reset_index()
    )

    agg = (
        df.groupby(group_cols)[METRICS]
        .mean()
        .reset_index()
    )

    n_runs = (
        df.groupby(group_cols)["run"]
        .nunique()
        .reset_index(name="n_runs")
    )

    subject_band = agg.merge(n_runs, on=group_cols)
    subject_band = subject_band.merge(meta, on="participant_id", how="left")

    print(f"\nSubject × band rows: {len(subject_band)}")

    wide_parts = []

    for metric in METRICS:
        pivot = subject_band.pivot(
            index="participant_id",
            columns="band",
            values=metric
        )
        pivot.columns = [f"{metric}_{b}" for b in pivot.columns]
        wide_parts.append(pivot)

    wide = pd.concat(wide_parts, axis=1)

    n_runs_subj = (
        subject_band.groupby("participant_id")["n_runs"]
        .max()
        .rename("n_runs")
    )
    wide = wide.join(n_runs_subj)
    wide = wide.join(meta.set_index("participant_id"))

    wide = wide.reset_index()

    front = ["participant_id", "age", "sex", "group", "n_runs"]
    metric_cols = [c for c in wide.columns if c not in front]
    wide = wide[front + sorted(metric_cols)]

    print(f"Subject-level rows: {len(wide)}")
    print("\nGroup counts:")
    print(wide["group"].value_counts(dropna=False))

    print("\nFirst columns:")
    print(wide.columns[:12].tolist())

    wide.to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}")

    print("\nSample metrics (alpha):")
    cols = [c for c in wide.columns if c.endswith("_alpha")]
    print(wide[["participant_id"] + cols].head(3).round(3).to_string(index=False))

    print("=" * 70)

if __name__ == "__main__":
    main()