from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "results" / "features"

SRC = FEATURES_DIR / "participants_features_subject_v0.2.csv"
OUT_FULL = FEATURES_DIR / "participants_features_subject_v0.3_full.csv"
OUT_CLEAN = FEATURES_DIR / "participants_features_subject_v0.3_clean.csv"

def main():
    print("=" * 65)
    print("NAI v0.3.1 — Clean dataset")
    print("=" * 65)

    df = pd.read_csv(SRC)
    print(f"Original subjects: {len(df)}")

    df["qc_flag"] = "ok"

    df.loc[df["participant_id"] == "sub-10777", "qc_flag"] = "extreme_artifact"

    df.loc[df["participant_id"] == "sub-11025", "qc_flag"] = "very_low_alpha"

    df.loc[df["age"].isna() | df["group"].isna(), "qc_flag"] = "missing_metadata"

    print("\nQC flags:")
    print(df["qc_flag"].value_counts())

    df.to_csv(OUT_FULL, index=False)
    print(f"\nSaved FULL  → {OUT_FULL.name}")

    clean = df[df["qc_flag"].isin(["ok", "very_low_alpha"])].copy()

    print(f"\nClean subjects: {len(clean)}")
    print(clean["group"].value_counts(dropna=False))
    print(clean["qc_flag"].value_counts())

    clean.to_csv(OUT_CLEAN, index=False)
    print(f"\nSaved CLEAN → {OUT_CLEAN.name}")

    print("\nDone.")

if __name__ == "__main__":
    main()