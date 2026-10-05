from pathlib import Path
import pandas as pd

FEATURES_DIR = Path(__file__).resolve().parents[1] / "results" / "features"
SRC = FEATURES_DIR / "dynamic_rest_v0.6.csv"

def main():
    df = pd.read_csv(SRC)
    print("=" * 70)
    print("NAI v0.6.2 — Dynamic Feature Audit")
    print("=" * 70)
    print(f"Rows: {len(df)}  |  Subjects: {df['participant_id'].nunique()}")

    cols = ["participant_id", "run", "band", "age", "group",
            "duration_s", "n_windows",
            "mean_delta", "cv_delta", "mean_degree_cv", "temporal_cv_degree_cv"]

    print("\n--- TOP 10 cv_delta ---")
    print(df.nlargest(10, "cv_delta")[cols].round(4).to_string(index=False))

    print("\n--- TOP 10 temporal_cv_degree_cv ---")
    print(df.nlargest(10, "temporal_cv_degree_cv")[cols].round(4).to_string(index=False))

    print("\n--- BOTTOM 10 mean_delta ---")
    print(df.nsmallest(10, "mean_delta")[cols].round(4).to_string(index=False))

    print("\n--- Descriptive (core) ---")
    print(df[["mean_delta", "cv_delta", "mean_degree_cv", "temporal_cv_degree_cv"]]
          .describe().round(4).to_string())

    print("=" * 70)

if __name__ == "__main__":
    main()