from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_CSVS = [
    PROJECT_ROOT / "results" / "features" / "features_merged_54d_v10.csv",
    PROJECT_ROOT / "results" / "features" / "features_v032.csv",
    PROJECT_ROOT / "results" / "ml" / "features_54d_development.csv",
    PROJECT_ROOT / "results" / "normative" / "features_merged_54d_v10.csv",
]

OUT_DIR = PROJECT_ROOT / "results" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "participant_id"
GROUP_COL = "group"
AGE_COL = "age"

SE_FEATURES = [
    "alpha_rel",
    "beta_rel",
    "theta_rel",
    "spectral_entropy_mean",
    "log_theta_alpha",
    "log_theta_beta",
]
C_FEATURES = [
    "plv_mean_theta",
    "plv_mean_alpha",
    "plv_mean_beta",
    "plv_mean_gamma",
    "plv_median_theta",
    "plv_median_alpha",
    "plv_median_beta",
    "plv_median_gamma",
]
G_FEATURES = [
    f"{m}_{b}"
    for m in [
        "mean_degree",
        "degree_cv",
        "clustering",
        "global_efficiency",
        "mean_path_length",
        "laplacian_entropy",
    ]
    for b in ["theta", "alpha", "beta", "gamma"]
] 
D_FEATURES = [
    f"{m}_{b}"
    for m in [
        "mean_delta",
        "cv_delta",
        "mean_degree_cv",
        "temporal_cv_degree_cv",
    ]
    for b in ["theta", "alpha", "beta", "gamma"]
]  

EXPECTED_54 = SE_FEATURES + C_FEATURES + G_FEATURES + D_FEATURES
assert len(EXPECTED_54) == 54, len(EXPECTED_54)

META_ALLOWED = {
    ID_COL,
    GROUP_COL,
    AGE_COL,
    "sex",
    "qc_flag",
    "n_runs",
    "handedness",
}

def resolve_csv(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    for p in CANDIDATE_CSVS:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "No development CSV found. Pass --csv PATH. Tried:\n"
        + "\n".join(str(p) for p in CANDIDATE_CSVS)
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="C1 ML dataset audit")
    parser.add_argument("--csv", default=None, help="Path to subject×54 feature CSV")
    args = parser.parse_args()

    csv_path = resolve_csv(args.csv)
    df = pd.read_csv(csv_path)

    lines: list[str] = []
    issues: list[str] = []
    warnings: list[str] = []

    def log(msg: str) -> None:
        lines.append(msg)
        print(msg)

    log("=" * 72)
    log("C1 — ML development dataset audit")
    log("=" * 72)
    log(f"CSV: {csv_path}")
    log(f"shape: {df.shape[0]} rows × {df.shape[1]} columns")

    if ID_COL not in df.columns:
        issues.append(f"missing column: {ID_COL}")
        log(f"FAIL  missing {ID_COL}")
    else:
        n = len(df)
        n_unique = df[ID_COL].nunique(dropna=False)
        n_dup = int(n - n_unique)
        log(f"rows: {n}")
        log(f"unique participant_id: {n_unique}")
        if n_dup:
            issues.append(f"duplicate participant_id: {n_dup}")
            dups = df[df[ID_COL].duplicated(keep=False)][ID_COL].tolist()
            log(f"FAIL  duplicate ids ({n_dup}): {dups[:20]}...")
        else:
            log("PASS  unique participant_id")

    if GROUP_COL not in df.columns:
        issues.append(f"missing column: {GROUP_COL}")
        log(f"FAIL  missing {GROUP_COL}")
        group_counts = {}
    else:
        g = df[GROUP_COL].astype(str).str.strip().str.upper()
        map_lab = {
            "TD": "TD",
            "CONTROL": "TD",
            "NT": "TD",
            "ASD": "ASD",
            "AUTISM": "ASD",
        }
        mapped = g.map(lambda x: map_lab.get(x, x))
        group_counts = mapped.value_counts(dropna=False).to_dict()
        log(f"group counts (raw→mapped): {group_counts}")
        allowed = {"TD", "ASD"}
        bad = set(mapped.unique()) - allowed
        if bad:
            issues.append(f"invalid group labels: {sorted(bad)}")
            log(f"FAIL  invalid labels: {sorted(bad)}")
        else:
            log("PASS  labels ⊆ {TD, ASD}")
        n_td = int((mapped == "TD").sum())
        n_asd = int((mapped == "ASD").sum())
        log(f"TD={n_td}  ASD={n_asd}  total_labeled={n_td + n_asd}")
        if n_td + n_asd != len(df):
            issues.append("not all rows have TD/ASD after mapping")
        if n_td != 39:
            warnings.append(f"TD count is {n_td}, expected ~39")
        if n_asd != 63:
            warnings.append(f"ASD count is {n_asd}, expected ~63")
        if len(df) != 102:
            warnings.append(f"n subjects is {len(df)}, expected ~102")

    age_summary = None
    if AGE_COL not in df.columns:
        warnings.append(f"missing optional column: {AGE_COL}")
        log(f"WARN  missing {AGE_COL}")
    else:
        age = pd.to_numeric(df[AGE_COL], errors="coerce")
        n_age_na = int(age.isna().sum())
        age_summary = {
            "n_missing": n_age_na,
            "min": float(age.min()) if age.notna().any() else None,
            "max": float(age.max()) if age.notna().any() else None,
            "mean": float(age.mean()) if age.notna().any() else None,
            "median": float(age.median()) if age.notna().any() else None,
        }
        log(f"age: missing={n_age_na}  min={age_summary['min']}  "
            f"max={age_summary['max']}  mean={age_summary['mean']}")
        if n_age_na:
            issues.append(f"age missing for {n_age_na} subjects")
            log(f"FAIL  age missing: {n_age_na}")
        else:
            log("PASS  age complete")

    present_54 = [c for c in EXPECTED_54 if c in df.columns]
    missing_54 = [c for c in EXPECTED_54 if c not in df.columns]
    extra = [
        c
        for c in df.columns
        if c not in EXPECTED_54 and c not in META_ALLOWED
    ]

    log(f"expected 54 features present: {len(present_54)} / 54")
    if missing_54:
        issues.append(f"missing feature columns ({len(missing_54)})")
        log(f"FAIL  missing features ({len(missing_54)}): {missing_54[:15]}...")
    else:
        log("PASS  all 54 expected feature names present")

    if extra:
        warnings.append(f"unexpected columns: {extra}")
        log(f"WARN  unexpected columns ({len(extra)}): {extra[:20]}")

    if present_54:
        X = df[present_54].apply(pd.to_numeric, errors="coerce")
        n_na = int(X.isna().sum().sum())
        n_inf = int(np.isinf(X.to_numpy(dtype=float, copy=True)).sum())
        log(f"feature NaN count: {n_na}")
        log(f"feature Inf count: {n_inf}")
        if n_na:
            per_col = X.isna().sum()
            bad_cols = per_col[per_col > 0].sort_values(ascending=False)
            issues.append(f"feature NaN cells: {n_na}")
            log(f"FAIL  NaN by column (top): {bad_cols.head(10).to_dict()}")
            na_rows = X.isna().any(axis=1)
            if ID_COL in df.columns:
                log(
                    "rows with any feature NaN: "
                    + str(df.loc[na_rows, ID_COL].tolist()[:30])
                )
        else:
            log("PASS  no feature NaN")
        if n_inf:
            issues.append(f"feature Inf cells: {n_inf}")
            log("FAIL  Inf present in features")
        else:
            log("PASS  no feature Inf")

        std = X.std(axis=0, ddof=0)
        zero_var = std[std == 0].index.tolist()
        if zero_var:
            warnings.append(f"zero-variance features: {zero_var}")
            log(f"WARN  zero-variance ({len(zero_var)}): {zero_var}")
        else:
            log("PASS  no zero-variance features among present")

        nunique = X.nunique(dropna=True)
        almost_const = nunique[nunique <= 1].index.tolist()
        if almost_const and almost_const != zero_var:
            warnings.append(f"≤1 unique value: {almost_const}")
            log(f"WARN  ≤1 unique: {almost_const}")

    log("-" * 72)
    status = "PASS" if not issues else "FAIL"
    log(f"ISSUES ({len(issues)}):")
    for i in issues:
        log(f"  - {i}")
    log(f"WARNINGS ({len(warnings)}):")
    for w in warnings:
        log(f"  - {w}")
    log(f"RESULT: {status}")
    log(
        "Note: C1 does not drop rows. Any exclusion list for C2 must be "
        "written explicitly after reviewing this audit."
    )
    log("=" * 72)

    report = {
        "csv": str(csv_path),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "group_counts": group_counts if GROUP_COL in df.columns else None,
        "age_summary": age_summary,
        "n_expected_features_present": len(present_54),
        "missing_features": missing_54,
        "unexpected_columns": extra,
        "issues": issues,
        "warnings": warnings,
        "status": status,
        "expected_n_subjects_hint": 102,
        "expected_td_hint": 39,
        "expected_asd_hint": 63,
        "block_dims": {"SE": 6, "C": 8, "G": 24, "D": 16},
    }

    out_json = OUT_DIR / "c1_dataset_audit.json"
    out_txt = OUT_DIR / "c1_dataset_audit.txt"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved → {out_json}")
    print(f"Saved → {out_txt}")

if __name__ == "__main__":
    main()