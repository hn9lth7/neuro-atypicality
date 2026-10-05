from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.features.blocks import ALL_BLOCKS, BLOCK_DIMS, SE_FEATURES, C_FEATURES, G_FEATURES, D_FEATURES

FEATURES_DIR = PROJECT_ROOT / "results" / "features"
NORM_DIR = PROJECT_ROOT / "results" / "normative"

NAI_CSV = NORM_DIR / "nai_v10.csv"
LOO_CSV = NORM_DIR / "loo_nai_v10.csv"
SUMMARY_JSON = NORM_DIR / "model_summary_v10.json"

FEATURE_FILES = {
    "SE": FEATURES_DIR / "features_v032.csv",
    "C_G": FEATURES_DIR / "connectivity_graph_subject_v0.5.csv",
    "D": FEATURES_DIR / "dynamic_subject_v0.6.csv",
}

COV_FILES = {
    "SE": NORM_DIR / "covariance_SE_v10.npy",
    "C": NORM_DIR / "covariance_C_v10.npy",
    "G": NORM_DIR / "covariance_G_v10.npy",
    "D": NORM_DIR / "covariance_D_v10.npy",
}

EXPECTED = {
    "n_total": 41,
    "n_td": 39,
    "n_asd": 2,
    "lambda": 0.10,
    "dims": {"SE": 6, "C": 8, "G": 24, "D": 16},
    "sub_11025_nai": 3.864,
    "sub_11038_nai": 1.242,
    "sub_11025_tol": 0.02,
    "sub_11038_tol": 0.02,
    "excluded": {"sub-10777"},
}

class Audit:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.passed += 1
        msg = f"  PASS  {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)

    def fail(self, name: str, detail: str = "") -> None:
        self.failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)

    def warn(self, name: str, detail: str = "") -> None:
        self.warnings += 1
        msg = f"  WARN  {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)

def _finite_matrix(A: np.ndarray) -> bool:
    return bool(np.isfinite(A).all())

def _is_symmetric(A: np.ndarray, tol: float = 1e-8) -> bool:
    return np.allclose(A, A.T, atol=tol)

def _is_pd(A: np.ndarray, tol: float = 1e-10) -> bool:
    try:
        np.linalg.cholesky(A + tol * np.eye(A.shape[0]))
        return True
    except np.linalg.LinAlgError:
        return False

def main() -> None:
    print("=" * 72)
    print("NAI v1.0 — RELEASE AUDIT")
    print("=" * 72)
    a = Audit()

    print("\n[1] ARTIFACTS")
    for path in [NAI_CSV, LOO_CSV, SUMMARY_JSON, *FEATURE_FILES.values(), *COV_FILES.values()]:
        if path.exists():
            a.ok(f"exists: {path.relative_to(PROJECT_ROOT)}")
        else:
            a.fail(f"missing: {path.relative_to(PROJECT_ROOT)}")

    if a.failed:
        print("\nCritical files missing — aborting further checks.")
        print(f"PASS={a.passed}  FAIL={a.failed}  WARN={a.warnings}")
        sys.exit(1)

    nai = pd.read_csv(NAI_CSV)
    loo = pd.read_csv(LOO_CSV)
    with open(SUMMARY_JSON, encoding="utf-8") as f:
        summary = json.load(f)

    print("\n[2] CANONICAL COHORT")
    n = len(nai)
    n_td = int((nai["group"] == "TD").sum())
    n_asd = int((nai["group"] == "ASD").sum())

    if n == EXPECTED["n_total"]:
        a.ok("n_total", str(n))
    else:
        a.fail("n_total", f"got {n}, expected {EXPECTED['n_total']}")

    if n_td == EXPECTED["n_td"]:
        a.ok("n_TD", str(n_td))
    else:
        a.fail("n_TD", f"got {n_td}, expected {EXPECTED['n_td']}")

    if n_asd == EXPECTED["n_asd"]:
        a.ok("n_ASD", str(n_asd))
    else:
        a.fail("n_ASD", f"got {n_asd}, expected {EXPECTED['n_asd']}")

    if nai["participant_id"].nunique() == n:
        a.ok("unique participant_id in nai_v10")
    else:
        a.fail("duplicate participant_id in nai_v10")

    if set(EXPECTED["excluded"]).isdisjoint(set(nai["participant_id"])):
        a.ok("excluded subjects absent", str(EXPECTED["excluded"]))
    else:
        a.fail("excluded subject still present")

    for pid in ("sub-11025", "sub-11038"):
        if pid in set(nai["participant_id"]):
            a.ok(f"ASD present: {pid}")
        else:
            a.fail(f"ASD missing: {pid}")

    print("\n[3] NAI TABLE INTEGRITY")
    score_cols = ["D_SE", "D_C", "D_G", "D_D", "NAI_v10"]
    for col in score_cols:
        if col not in nai.columns:
            a.fail(f"column missing: {col}")
            continue
        s = nai[col]
        if s.isna().any():
            a.fail(f"NaN in {col}", f"count={int(s.isna().sum())}")
        elif not np.isfinite(s.astype(float)).all():
            a.fail(f"non-finite in {col}")
        else:
            a.ok(f"finite: {col}")

    if nai["age"].isna().any():
        a.fail("NaN in age")
    else:
        a.ok("age complete")

    print("\n[4] FEATURE SCHEMA")
    if BLOCK_DIMS != EXPECTED["dims"]:
        a.fail("BLOCK_DIMS mismatch", str(BLOCK_DIMS))
    else:
        a.ok("BLOCK_DIMS", str(BLOCK_DIMS))

    feat_se = pd.read_csv(FEATURE_FILES["SE"])
    feat_cg = pd.read_csv(FEATURE_FILES["C_G"])
    feat_d = pd.read_csv(FEATURE_FILES["D"])

    for name, path_df, cols in [
        ("SE", feat_se, SE_FEATURES),
        ("C", feat_cg, C_FEATURES),
        ("G", feat_cg, G_FEATURES),
        ("D", feat_d, D_FEATURES),
    ]:
        missing = [c for c in cols if c not in path_df.columns]
        if missing:
            a.fail(f"{name} schema", f"missing {missing[:5]}...")
        else:
            a.ok(f"{name} schema ({len(cols)} cols)")

    for label, df in [
        ("features_v032", feat_se),
        ("connectivity_graph_subject_v0.5", feat_cg),
        ("dynamic_subject_v0.6", feat_d),
    ]:
        if df["participant_id"].nunique() != len(df):
            a.fail(f"duplicate IDs in {label}")
        else:
            a.ok(f"unique IDs in {label}")

    print("\n[5] COVARIANCE MATRICES")
    for name, path in COV_FILES.items():
        S = np.load(path)
        p = EXPECTED["dims"][name]
        if S.shape != (p, p):
            a.fail(f"Σ_{name} shape", f"{S.shape} != ({p},{p})")
        else:
            a.ok(f"Σ_{name} shape {S.shape}")

        if not _finite_matrix(S):
            a.fail(f"Σ_{name} non-finite")
        else:
            a.ok(f"Σ_{name} finite")

        if not _is_symmetric(S):
            a.fail(f"Σ_{name} not symmetric")
        else:
            a.ok(f"Σ_{name} symmetric")

        if _is_pd(S):
            a.ok(f"Σ_{name} positive definite (reg)")
        else:
            a.fail(f"Σ_{name} not PD")

        ev = np.linalg.eigvalsh(S)
        if (ev > 0).all():
            a.ok(f"Σ_{name} all eigenvalues > 0", f"min={ev.min():.2e}")
        else:
            a.fail(f"Σ_{name} non-positive eigenvalue", f"min={ev.min():.2e}")

    print("\n[6] MODEL SUMMARY JSON")
    lam = summary.get("lambda", summary.get("lambda_ref"))
    if lam is None:
        a.fail("lambda missing in summary JSON")
    elif abs(float(lam) - EXPECTED["lambda"]) < 1e-9:
        a.ok("lambda", str(lam))
    else:
        a.fail("lambda", f"got {lam}, expected {EXPECTED['lambda']}")

    if summary.get("n_td") == EXPECTED["n_td"]:
        a.ok("summary n_td")
    else:
        a.warn("summary n_td", str(summary.get("n_td")))

    if summary.get("n_asd") == EXPECTED["n_asd"]:
        a.ok("summary n_asd")
    else:
        a.warn("summary n_asd", str(summary.get("n_asd")))

    print("\n[7] NAI FORMULA CONSISTENCY")
    recomputed = nai[["D_SE", "D_C", "D_G", "D_D"]].mean(axis=1)
    diff = (recomputed - nai["NAI_v10"]).abs()
    max_diff = float(diff.max())
    if max_diff < 1e-8:
        a.ok("NAI = mean(D_SE,D_C,D_G,D_D)", f"max|Δ|={max_diff:.2e}")
    else:
        a.fail("NAI formula mismatch", f"max|Δ|={max_diff:.2e}")

    print("\n[8] CONTROL SUBJECT SCORES")
    for pid, key, tol in [
        ("sub-11025", "sub_11025_nai", "sub_11025_tol"),
        ("sub-11038", "sub_11038_nai", "sub_11038_tol"),
    ]:
        row = nai.loc[nai["participant_id"] == pid]
        if row.empty:
            a.fail(f"{pid} not in nai_v10")
            continue
        val = float(row["NAI_v10"].iloc[0])
        exp = EXPECTED[key]
        if abs(val - exp) <= EXPECTED[tol]:
            a.ok(f"{pid} NAI", f"{val:.3f} ≈ {exp:.3f}")
        else:
            a.fail(f"{pid} NAI", f"got {val:.3f}, expected ~{exp:.3f}")

    print("\n[9] LOO TABLE")
    if len(loo) != EXPECTED["n_td"]:
        a.fail("LOO n_rows", f"{len(loo)} != {EXPECTED['n_td']}")
    else:
        a.ok("LOO n_rows", str(len(loo)))

    if loo["participant_id"].nunique() != len(loo):
        a.fail("LOO duplicate IDs")
    else:
        a.ok("LOO unique IDs")

    loo_cols = [c for c in loo.columns if c.startswith("D_") or c == "NAI_LOO"]
    if not loo_cols and "NAI_LOO" not in loo.columns:
        loo_cols = [c for c in loo.columns if "LOO" in c or c.startswith("D_")]
    for col in loo_cols:
        s = loo[col]
        if s.isna().any() or not np.isfinite(s.astype(float)).all():
            a.fail(f"LOO non-finite: {col}")
        else:
            a.ok(f"LOO finite: {col}")

    loo_ids = set(loo["participant_id"])
    nai_td_ids = set(nai.loc[nai["group"] == "TD", "participant_id"])
    if loo_ids <= nai_td_ids:
        a.ok("LOO subjects ⊆ TD")
    else:
        a.fail("LOO contains non-TD subjects", str(loo_ids - nai_td_ids))

    print("\n[10] SCORE DOMAIN")
    for col in score_cols:
        if col not in nai.columns:
            continue
        if (nai[col] >= 0).all():
            a.ok(f"{col} ≥ 0")
        else:
            a.fail(f"{col} has negative values")

    print("\n" + "=" * 72)
    print("RELEASE AUDIT SUMMARY")
    print("=" * 72)
    print(f"  PASS : {a.passed}")
    print(f"  FAIL : {a.failed}")
    print(f"  WARN : {a.warnings}")

    if a.failed == 0:
        print("\n  RESULT: PASS — NAI v1.0 artifacts are consistent.")
        print("  Safe to proceed to mathematical_model.md and final figures.")
        status = 0
    else:
        print("\n  RESULT: FAIL — fix issues before final documentation freeze.")
        status = 1

    print("=" * 72)
    sys.exit(status)

if __name__ == "__main__":
    main()