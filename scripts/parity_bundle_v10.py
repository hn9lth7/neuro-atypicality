from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nai.product.model_bundle import load_bundle
from nai.product.score import score_row

FEATURES_DIR = ROOT / "results" / "features"
NORM_DIR = ROOT / "results" / "normative"
BUNDLE_DIR = ROOT / "models" / "nai_v1"
REF_CSV = NORM_DIR / "nai_v10.csv"
OUT_CSV = NORM_DIR / "parity_bundle_v10.csv"

TOL = 1e-6

def _assert_unique_subjects(df: pd.DataFrame, name: str) -> None:
    n = len(df)
    n_unique = df["participant_id"].nunique()
    if n != n_unique:
        raise ValueError(
            f"{name}: expected 1 row per subject, got {n} rows / {n_unique} ids"
        )

def load_merged() -> pd.DataFrame:
    spectral = pd.read_csv(FEATURES_DIR / "features_v032.csv")
    conn = pd.read_csv(FEATURES_DIR / "connectivity_graph_subject_v0.5.csv")
    dyn = pd.read_csv(FEATURES_DIR / "dynamic_subject_v0.6.csv")

    _assert_unique_subjects(spectral, "features_v032")
    _assert_unique_subjects(conn, "connectivity_graph_subject_v0.5")
    _assert_unique_subjects(dyn, "dynamic_subject_v0.6")

    df = spectral.merge(
        conn.drop(columns=["age", "sex", "group"], errors="ignore"),
        on="participant_id",
        how="inner",
    )
    df = df.merge(
        dyn.drop(columns=["age", "sex", "group", "n_runs"], errors="ignore"),
        on="participant_id",
        how="inner",
    )

    if "qc_flag" in df.columns:
        df = df[df["qc_flag"].isin(["ok", "very_low_alpha"])].copy()
    df = df[df["participant_id"] != "sub-10777"].copy()
    df = df.dropna(subset=["age", "group"]).copy()

    _assert_unique_subjects(df, "merged canonical")
    return df.reset_index(drop=True)

def main() -> None:
    print("=" * 72)
    print("NAI v1 — Bundle parity vs nai_v10.csv")
    print("=" * 72)

    bundle = load_bundle(BUNDLE_DIR)
    print(f"Bundle: {BUNDLE_DIR}")
    print(f"lambda: {bundle.lambda_}")

    df = load_merged()
    ref = pd.read_csv(REF_CSV)

    rows = []
    for _, row in df.iterrows():
        sc = score_row(row, bundle)
        rows.append(
            {
                "participant_id": row["participant_id"],
                "age": row["age"],
                "group": row["group"],
                "D_SE": sc["D_SE"],
                "D_C": sc["D_C"],
                "D_G": sc["D_G"],
                "D_D": sc["D_D"],
                "NAI": sc["NAI"],
            }
        )

    scored = pd.DataFrame(rows)
    merged = scored.merge(
        ref[
            [
                "participant_id",
                "D_SE",
                "D_C",
                "D_G",
                "D_D",
                "NAI_v10",
            ]
        ],
        on="participant_id",
        how="inner",
        suffixes=("_bundle", "_ref"),
    )

    if "D_SE_ref" not in merged.columns and "D_SE_y" in merged.columns:
        merged = merged.rename(
            columns={
                "D_SE_x": "D_SE_bundle",
                "D_C_x": "D_C_bundle",
                "D_G_x": "D_G_bundle",
                "D_D_x": "D_D_bundle",
                "D_SE_y": "D_SE_ref",
                "D_C_y": "D_C_ref",
                "D_G_y": "D_G_ref",
                "D_D_y": "D_D_ref",
            }
        )
    elif "D_SE_bundle" not in merged.columns:
        merged = merged.rename(
            columns={
                "D_SE": "D_SE_bundle",
                "D_C": "D_C_bundle",
                "D_G": "D_G_bundle",
                "D_D": "D_D_bundle",
            }
        )
        if "NAI_v10" in merged.columns and "D_SE_ref" not in merged.columns:
            pass

    ref2 = ref.rename(
        columns={
            "D_SE": "D_SE_ref",
            "D_C": "D_C_ref",
            "D_G": "D_G_ref",
            "D_D": "D_D_ref",
            "NAI_v10": "NAI_ref",
        }
    )
    scored2 = scored.rename(
        columns={
            "D_SE": "D_SE_bundle",
            "D_C": "D_C_bundle",
            "D_G": "D_G_bundle",
            "D_D": "D_D_bundle",
            "NAI": "NAI_bundle",
        }
    )
    m = scored2.merge(
        ref2[
            [
                "participant_id",
                "D_SE_ref",
                "D_C_ref",
                "D_G_ref",
                "D_D_ref",
                "NAI_ref",
            ]
        ],
        on="participant_id",
        how="inner",
    )

    print(f"Matched subjects: {len(m)}")
    if len(m) != len(ref):
        print(f"WARNING: ref has {len(ref)} rows, matched {len(m)}")

    pairs = [
        ("D_SE_bundle", "D_SE_ref"),
        ("D_C_bundle", "D_C_ref"),
        ("D_G_bundle", "D_G_ref"),
        ("D_D_bundle", "D_D_ref"),
        ("NAI_bundle", "NAI_ref"),
    ]

    print("-" * 72)
    ok = True
    for a, b in pairs:
        delta = (m[a] - m[b]).abs()
        mx, mean = float(delta.max()), float(delta.mean())
        label = a.replace("_bundle", "")
        print(f"  {label:8s}  max|Δ|={mx:.6e}  mean|Δ|={mean:.6e}")
        if label == "NAI" and mx > TOL:
            ok = False
        if label != "NAI" and mx > TOL:
            ok = False

    print("-" * 72)
    for pid in ("sub-11025", "sub-11038", "sub-10025"):
        sub = m[m["participant_id"] == pid]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        print(
            f"  {pid}: NAI bundle={r['NAI_bundle']:.6f}  "
            f"ref={r['NAI_ref']:.6f}  |Δ|={abs(r['NAI_bundle']-r['NAI_ref']):.3e}"
        )

    m.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}")

    if ok:
        print(f"PARITY PASS  (all |Δ| ≤ {TOL})")
    else:
        print(f"PARITY FAIL  (threshold {TOL})")
        raise SystemExit(1)

    print("=" * 72)

if __name__ == "__main__":
    main()