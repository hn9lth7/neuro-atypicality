from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"

RUN_CSV = FEATURES_DIR / "participants_features_rest_v0.2.csv"
OUT_CSV = FEATURES_DIR / "participants_features_subject_v0.2.csv"

def main():
    print("=" * 65)
    print("NAI v0.2 — Subject-level aggregation")
    print("=" * 65)

    df = pd.read_csv(RUN_CSV)
    print(f"Run-level rows     : {len(df)}")
    print(f"Unique subjects    : {df['participant_id'].nunique()}")

    numeric_cols = [
        c for c in df.columns
        if c not in ("participant_id", "age", "sex", "group", "run")
    ]

    group_cols = ["participant_id", "age", "sex", "group"]

    agg = {col: "mean" for col in numeric_cols}
    agg["run"] = "count"

    df_subj = (
        df.groupby(group_cols, dropna=False)
        .agg(agg)
        .reset_index()
        .rename(columns={"run": "n_runs"})
    )

    abs_cols = ["delta_abs", "theta_abs", "alpha_abs", "beta_abs", "gamma_abs"]
    if all(c in df_subj.columns for c in abs_cols):
        total = df_subj[abs_cols].sum(axis=1) + 1e-12
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            df_subj[f"{band}_rel"] = df_subj[f"{band}_abs"] / total

        df_subj["theta_alpha"] = df_subj["theta_abs"] / (df_subj["alpha_abs"] + 1e-12)
        df_subj["theta_beta"]  = df_subj["theta_abs"] / (df_subj["beta_abs"]  + 1e-12)
        df_subj["alpha_beta"]  = df_subj["alpha_abs"] / (df_subj["beta_abs"]  + 1e-12)

    print("\nSubject-level summary")
    print("-" * 40)
    print(f"Subjects           : {len(df_subj)}")
    print("\nGroup counts:")
    print(df_subj["group"].value_counts(dropna=False))
    print("\nRuns per subject:")
    print(df_subj["n_runs"].describe().round(2))

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df_subj.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")

    print("\nFirst rows:")
    print(df_subj.head(3).to_string())

if __name__ == "__main__":
    main()