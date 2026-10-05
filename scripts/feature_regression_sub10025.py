from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.blocks import SE_FEATURES, C_FEATURES, G_FEATURES, D_FEATURES

V11 = PROJECT_ROOT / "results" / "features" / "v11_assembly_sub-10025_run-01.csv"
CONN = PROJECT_ROOT / "results" / "features" / "connectivity_graph_rest_v0.5.csv"
DYN = PROJECT_ROOT / "results" / "features" / "dynamic_rest_v0.6.csv"

SE_CANDIDATES = [
    PROJECT_ROOT / "results" / "features" / "participants_features_rest_v0.1.csv",
]

def row_v11() -> pd.Series:
    df = pd.read_csv(V11)
    return df.iloc[0]

def pick_frozen(path: Path, bands_as_rows: bool = False) -> dict[str, float]:
    df = pd.read_csv(path)
    
    if "run" in df.columns:
        df["run"] = df["run"].astype(str).str.replace(r"^0", "", regex=True)
        df["run"] = df["run"].astype(str).str.zfill(2)
    m = df["participant_id"].astype(str).str.contains("10025")
    if "run" in df.columns:
        m &= df["run"].astype(str).isin(["01", "1"])
    sub = df.loc[m]
    if sub.empty:
        raise RuntimeError(f"no sub-10025 run-01 in {path.name}")

    if bands_as_rows and "band" in sub.columns:
        out = {}
        for _, r in sub.iterrows():
            b = str(r["band"]).lower()
            for col in sub.columns:
                if col in ("participant_id", "run", "band", "group", "age", "sex", "duration_s", "n_windows"):
                    continue
                if col in D_FEATURES or col.startswith(("mean_delta", "cv_delta", "mean_degree", "temporal")):
  
                    key = f"{col}_{b}" if not col.endswith(f"_{b}") else col
                    if key in D_FEATURES or any(col.startswith(p) for p in ("mean_delta", "cv_delta", "mean_degree_cv", "temporal_cv")):
                        out[f"{col}_{b}" if col in ("mean_delta", "cv_delta", "mean_degree_cv", "temporal_cv_degree_cv") else key] = float(r[col])

        out = {}
        metric_cols = [c for c in sub.columns if c not in (
            "participant_id", "run", "band", "group", "age", "sex", "duration_s", "n_windows", "n_channels"
        )]
        for _, r in sub.iterrows():
            b = str(r["band"]).lower()
            for c in metric_cols:
                out[f"{c}_{b}"] = float(r[c])
        return out

    r = sub.iloc[0]
    return {c: float(r[c]) for c in r.index if c not in (
        "participant_id", "run", "group", "age", "sex", "duration_s", "n_channels", "n_runs"
    ) and isinstance(r[c], (int, float, np.floating)) and np.isfinite(r[c])}

def compare(block: str, names: list[str], v11: pd.Series, frozen: dict[str, float], atol=1e-5, rtol=1e-4):
    print(f"\n=== {block} ===")
    rows = []
    for k in names:
        a = float(v11[k])
        if k not in frozen:
            print(f"  {k:32s}  v11={a:.6g}  FROZEN=MISSING")
            rows.append((k, a, np.nan, np.nan, np.nan))
            continue
        b = float(frozen[k])
        ad = abs(a - b)
        rd = ad / (abs(b) + 1e-12)
        ok = "OK" if (ad <= atol or rd <= rtol) else "DIFF"
        print(f"  {k:32s}  v11={a:.6g}  frz={b:.6g}  |Δ|={ad:.3g}  rel={rd:.3g}  {ok}")
        rows.append((k, a, b, ad, rd))
    return rows

def main() -> None:
    v11 = row_v11()
    print("v11 file:", V11)

    if CONN.exists():
        fr_cg = pick_frozen(CONN, bands_as_rows=False)
        df = pd.read_csv(CONN)
        m = df["participant_id"].astype(str).str.contains("10025")
        if "run" in df.columns:
            runs = df.loc[m, "run"].astype(str)
            m2 = m & df["run"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2).isin(["01", "1"])
            if m2.any():
                m = m2
        sub = df.loc[m]
        fr_cg = {}
        if "band" in sub.columns:
            for _, r in sub.iterrows():
                b = str(r["band"]).lower()
                for c in sub.columns:
                    if c in ("participant_id", "run", "band", "group", "age", "sex", "duration_s", "n_channels", "n_runs"):
                        continue
                    try:
                        fr_cg[f"{c}_{b}"] = float(r[c])
                    except (TypeError, ValueError):
                        pass
        else:
            r = sub.iloc[0]
            for c in sub.columns:
                try:
                    fr_cg[c] = float(r[c])
                except (TypeError, ValueError):
                    pass
        compare("C", C_FEATURES, v11, fr_cg)
        compare("G", G_FEATURES, v11, fr_cg)
    else:
        print("MISSING", CONN)

    if DYN.exists():
        df = pd.read_csv(DYN)
        m = df["participant_id"].astype(str).str.contains("10025")
        if "run" in df.columns:
            m = m & df["run"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2).isin(["01", "1"])
        sub = df.loc[m]
        fr_d = {}
        if "band" in sub.columns:
            for _, r in sub.iterrows():
                b = str(r["band"]).lower()
                for c in ("mean_delta", "cv_delta", "mean_degree_cv", "temporal_cv_degree_cv"):
                    if c in r.index:
                        fr_d[f"{c}_{b}"] = float(r[c])
        compare("D", D_FEATURES, v11, fr_d)
    else:
        print("MISSING", DYN)

    for path in SE_CANDIDATES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        m = df["participant_id"].astype(str).str.contains("10025")
        if "run" in df.columns:
            m = m & df["run"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2).isin(["01", "1"])
        sub = df.loc[m]
        if sub.empty:
            continue
        r = sub.iloc[0]
        fr_se = {k: float(r[k]) for k in SE_FEATURES if k in r.index}
        compare("SE", SE_FEATURES, v11, fr_se)
        break
    else:
        print("\nSE frozen run-level: not found in candidates (compare manually to features_v032 subject mean later)")

    print("\nDone. Review DIFF lines; OK within atol/rtol is expected match.")

if __name__ == "__main__":
    main()