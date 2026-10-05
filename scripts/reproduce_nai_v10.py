from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nai.features.blocks import C_FEATURES, D_FEATURES, G_FEATURES, SE_FEATURES
from nai.features.extractor import score_nai_from_dataframe

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "results" / "features"
REF = ROOT / "results" / "normative" / "nai_v10.csv"

def load_merged() -> pd.DataFrame:
    spectral = pd.read_csv(FEAT / "features_v032.csv")
    conn = pd.read_csv(FEAT / "connectivity_graph_subject_v0.5.csv")
    dyn = pd.read_csv(FEAT / "dynamic_subject_v0.6.csv")
    for extra in (conn, dyn):
        drop = [c for c in ("age", "sex", "group") if c in extra.columns and c in spectral.columns]
        extra.drop(columns=drop, inplace=True, errors="ignore")
    df = spectral.merge(conn, on="participant_id", how="inner").merge(dyn, on="participant_id", how="inner")
    if "qc_flag" in df.columns:
        df = df[~df["qc_flag"].isin(["extreme_artifact", "missing_metadata"])].copy()
    df = df[df["group"].isin(["TD", "ASD"])].dropna(subset=["age"]).copy()
    return df.reset_index(drop=True)

def main() -> None:
    df = load_merged()
    scored = score_nai_from_dataframe(df)
    ref = pd.read_csv(REF)

    m = scored.merge(
        ref[["participant_id", "NAI_v10", "D_SE", "D_C", "D_G", "D_D"]].rename(
            columns={
                "NAI_v10": "NAI_ref",
                "D_SE": "D_SE_ref",
                "D_C": "D_C_ref",
                "D_G": "D_G_ref",
                "D_D": "D_D_ref",
            }
        ),
        on="participant_id",
        how="inner",
    )
    print(f"Matched subjects: {len(m)}")
    for a, b in [
        ("NAI", "NAI_ref"),
        ("D_SE", "D_SE_ref"),
        ("D_C", "D_C_ref"),
        ("D_G", "D_G_ref"),
        ("D_D", "D_D_ref"),
    ]:
        diff = np.abs(m[a].values - m[b].values)
        print(f"  max|{a}-{b}| = {diff.max():.3e}  mean = {diff.mean():.3e}")

    for pid in ["sub-11025", "sub-11038"]:
        row = m[m["participant_id"] == pid]
        if len(row):
            print(
                f"  {pid}: NAI_lib={row['NAI'].iloc[0]:.6f}  "
                f"NAI_ref={row['NAI_ref'].iloc[0]:.6f}"
            )

if __name__ == "__main__":
    main()