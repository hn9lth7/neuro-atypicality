from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nai.features.blocks import ALL_BLOCKS
from nai.normative.regression import fit_age_models

FEATURES_DIR = ROOT / "results" / "features"
OUT = ROOT / "models" / "nai_v1" / "age_models.json"

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

def models_to_dict(feature_names: list[str], models: list) -> dict:
    out = {}
    for name, m in zip(feature_names, models, strict=True):
        out[name] = {
            "intercept": float(np.asarray(m.intercept_).ravel()[0]),
            "slope": float(np.asarray(m.coef_).ravel()[0]),
        }
    return out

def main() -> None:
    df = load_merged()
    td = df[df["group"] == "TD"].copy().reset_index(drop=True)
    if len(td) != 39:
        raise SystemExit(f"expected 39 TD, got {len(td)}")

    age = td["age"].to_numpy(dtype=float)
    payload = {
        "version": "v1.0",
        "n_td": 39,
        "age_model": "TD-only linear",
        "source": "export_age_models_v10.py / same fit as nai_v10_unified",
        "blocks": {},
    }

    for block, feats in ALL_BLOCKS.items():
        missing = [f for f in feats if f not in td.columns]
        if missing:
            raise SystemExit(f"{block} missing columns: {missing}")
        X = td[feats].to_numpy(dtype=float)
        models = fit_age_models(X, age)
        payload["blocks"][block] = models_to_dict(feats, models)
        print(f"[{block}] {len(feats)} models")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved → {OUT}")

if __name__ == "__main__":
    main()