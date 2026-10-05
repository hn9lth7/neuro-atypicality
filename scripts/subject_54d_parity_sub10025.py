from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.blocks import SE_FEATURES, C_FEATURES, G_FEATURES, D_FEATURES
from nai.features.subject_aggregation import aggregate_se_subject, mean_columns

SUBJECT = "sub-10025"
FEAT = PROJECT_ROOT / "results" / "features"

def _norm_run(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)

def load_subject_runs_conn() -> pd.DataFrame:
    df = pd.read_csv(FEAT / "connectivity_graph_rest_v0.5.csv")
    m = df["participant_id"].astype(str).str.contains("10025")
    df = df.loc[m].copy()
    if "band" in df.columns:
        rows = {}
        for _, r in df.iterrows():
            b = str(r["band"]).lower()
            key = str(r.get("run", "01"))
            rows.setdefault(key, {"run": key})
            for c in df.columns:
                if c in ("participant_id", "run", "band", "group", "age", "sex"):
                    continue
                try:
                    rows[key][f"{c}_{b}"] = float(r[c])
                except (TypeError, ValueError):
                    pass
        return pd.DataFrame(list(rows.values()))
    return df

def load_subject_runs_dyn() -> pd.DataFrame:
    df = pd.read_csv(FEAT / "dynamic_rest_v0.6.csv")
    m = df["participant_id"].astype(str).str.contains("10025")
    df = df.loc[m].copy()
    if "band" in df.columns:
        rows = {}
        for _, r in df.iterrows():
            b = str(r["band"]).lower()
            key = str(r.get("run", "01"))
            rows.setdefault(key, {"run": key})
            for c in ("mean_delta", "cv_delta", "mean_degree_cv", "temporal_cv_degree_cv"):
                if c in r.index:
                    rows[key][f"{c}_{b}"] = float(r[c])
        return pd.DataFrame(list(rows.values()))
    return df

def load_se_runs() -> pd.DataFrame:
    p = FEAT / "participants_features_rest_v0.1.csv"
    df = pd.read_csv(p)
    m = df["participant_id"].astype(str).str.contains("10025")
    df = df.loc[m].copy()
    if "run" in df.columns:
        df["run"] = _norm_run(df["run"])
        df = df.drop_duplicates(subset=["run"])
    ent = FEAT / "v11_se_per_run_sub-10025.csv"
    if ent.exists() and "spectral_entropy_mean" not in df.columns:
        e = pd.read_csv(ent)
        e["run"] = _norm_run(e["run"])
        df["run"] = _norm_run(df["run"]) if "run" in df.columns else "01"
        df = df.merge(e[["run", "spectral_entropy_mean"]], on="run", how="left")
    return df

def compare(block: str, names, got: dict, frozen: pd.Series):
    print(f"\n=== {block} ===")
    n_ok = 0
    for k in names:
        a = float(got[k])
        if k not in frozen.index or pd.isna(frozen[k]):
            print(f"  {k:32s} MISSING in frozen")
            continue
        b = float(frozen[k])
        ad = abs(a - b)
        ok = ad < 1e-8 or ad / (abs(b) + 1e-12) < 1e-8
        n_ok += int(ok)
        print(f"  {k:32s} |Δ|={ad:.3e}  {'OK' if ok else 'DIFF'}")
    print(f"  → {n_ok}/{len(names)}")
    return n_ok

def main():
    se_runs = load_se_runs()
    se = aggregate_se_subject(se_runs)

    conn = load_subject_runs_conn()
    dyn = load_subject_runs_dyn()
    c = mean_columns(conn, C_FEATURES)
    g = mean_columns(conn, G_FEATURES)
    d = mean_columns(dyn, D_FEATURES)

    vec = {**{k: se[k] for k in SE_FEATURES}, **c, **g, **d}
    assert len(vec) == 54

    frz_se = pd.read_csv(FEAT / "features_v032.csv")
    frz_se = frz_se[frz_se["participant_id"].astype(str).str.contains("10025")].iloc[0]

    frz_cg = pd.read_csv(FEAT / "connectivity_graph_subject_v0.5.csv")
    frz_cg = frz_cg[frz_cg["participant_id"].astype(str).str.contains("10025")].iloc[0]

    frz_d = pd.read_csv(FEAT / "dynamic_subject_v0.6.csv")
    frz_d = frz_d[frz_d["participant_id"].astype(str).str.contains("10025")].iloc[0]

    n = 0
    n += compare("SE", SE_FEATURES, vec, frz_se)
    n += compare("C", C_FEATURES, vec, frz_cg)
    n += compare("G", G_FEATURES, vec, frz_cg)
    n += compare("D", D_FEATURES, vec, frz_d)
    print(f"\nTOTAL OK features checked (max 54): review per-block counts")
    print("If C/G/D all OK → subject 54-D aggregation CLOSED")

if __name__ == "__main__":
    main()