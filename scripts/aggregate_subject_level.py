from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"

RUN_LEVEL_CSV = FEATURES_DIR / "participants_features_rest_v0.1.csv"
SUBJECT_LEVEL_CSV = FEATURES_DIR / "participants_features_subject_v0.2.csv"

def main():
    print("=" * 60)
    print("NAI v0.2 — Subject-level aggregation")
    print("=" * 60)

    df = pd.read_csv(RUN_LEVEL_CSV)
    print(f"Run-level rows: {len(df)}")
    print(f"Unique subjects: {df['participant_id'].nunique()}")
    print("\nGroup counts (runs):")
    print(df["group"].value_counts(dropna=False))

    feature_cols = [
        c for c in df.columns
        if c.endswith(("_abs", "_rel")) or c in ("theta_alpha", "theta_beta", "alpha_beta", "duration_s", "n_channels")
    ]

    group_cols = ["participant_id", "age", "sex", "group"]

    agg_dict = {col: "mean" for col in feature_cols}
    agg_dict["run"] = "count"  

    df_subj = (
        df.groupby(group_cols, dropna=False)
        .agg(agg_dict)
        .reset_index()
        .rename(columns={"run": "n_runs"})
    )

    print("\n" + "-" * 60)
    print("Subject-level summary")
    print("-" * 60)
    print(f"Subjects: {len(df_subj)}")
    print("\nGroup counts (subjects):")
    print(df_subj["group"].value_counts(dropna=False))
    print("\nRuns per subject:")
    print(df_subj["n_runs"].describe())

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df_subj.to_csv(SUBJECT_LEVEL_CSV, index=False)
    print(f"\nSaved → {SUBJECT_LEVEL_CSV}")

    print("\nFirst rows:")
    print(df_subj.head().to_string())

if __name__ == "__main__":
    main()