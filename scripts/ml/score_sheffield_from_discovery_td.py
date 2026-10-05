from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT_DIR = PROJECT_ROOT / "results" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHEFFIELD_PATH = PROJECT_ROOT / "results" / "features" / "sheffield_features_54d.csv"
OUT_SCORES = OUT_DIR / "c6_sheffield_from_discovery_scores.csv"
OUT_JSON = OUT_DIR / "c6_sheffield_from_discovery_summary.json"

LAMBDA = 0.10

SE_FEATURES = [
    "alpha_rel", "beta_rel", "theta_rel",
    "spectral_entropy_mean", "log_theta_alpha", "log_theta_beta",
]
C_FEATURES = [
    "plv_mean_theta", "plv_mean_alpha", "plv_mean_beta", "plv_mean_gamma",
    "plv_median_theta", "plv_median_alpha", "plv_median_beta", "plv_median_gamma",
]
G_FEATURES = [
    f"{m}_{b}"
    for m in (
        "mean_degree", "degree_cv", "clustering",
        "global_efficiency", "mean_path_length", "laplacian_entropy",
    )
    for b in ("theta", "alpha", "beta", "gamma")
]
D_FEATURES = [
    f"{m}_{b}"
    for m in (
        "mean_delta", "cv_delta", "mean_degree_cv", "temporal_cv_degree_cv",
    )
    for b in ("theta", "alpha", "beta", "gamma")
]
ALL_BLOCKS = {"SE": SE_FEATURES, "C": C_FEATURES, "G": G_FEATURES, "D": D_FEATURES}

def _load_discovery_td() -> pd.DataFrame:
    feat_dir = PROJECT_ROOT / "results" / "features"

    candidates_se = [
        feat_dir / "features_v032_TD.csv",
        feat_dir / "features_v032.csv",
        feat_dir / "participants_features_subject_v0.3_clean.csv",
        feat_dir / "participants_features_subject_v0.2.csv",
    ]
    se_path = next((p for p in candidates_se if p.exists()), None)
    if se_path is None:
        raise SystemExit(f"No discovery SE table found in {feat_dir}")

    se = pd.read_csv(se_path)
    if "group" in se.columns:
        se = se[se["group"].isin(["TD", "CTRL"])].copy()
        se["group"] = "TD"

    conn_path = feat_dir / "connectivity_graph_subject_v0.5.csv"
    dyn_path = feat_dir / "dynamic_subject_v0.6.csv"
    if not conn_path.exists() or not dyn_path.exists():
        raise SystemExit("Need connectivity_graph_subject_v0.5.csv and dynamic_subject_v0.6.csv")

    conn = pd.read_csv(conn_path)
    dyn = pd.read_csv(dyn_path)

    exclude = {"sub-10777", "sub-10950"}
    for d in (se, conn, dyn):
        if "participant_id" in d.columns:
            d.drop(index=d.index[d["participant_id"].isin(exclude)], inplace=True, errors="ignore")

    df = se.merge(conn, on="participant_id", how="inner", suffixes=("", "_conn"))
    df = df.merge(dyn, on="participant_id", how="inner", suffixes=("", "_dyn"))

    if "age" not in df.columns and "age_x" in df.columns:
        df["age"] = df["age_x"]
    if "group" not in df.columns:
        df["group"] = "TD"

    df = df[df["group"].isin(["TD", "CTRL"])].copy()
    df["group"] = "TD"
    df = df.dropna(subset=["age"]).reset_index(drop=True)

    df = df.drop_duplicates(subset=["participant_id"], keep="first")
    return df

def fit_age_models(X: np.ndarray, age: np.ndarray) -> list[LinearRegression]:
    models = []
    for j in range(X.shape[1]):
        m = LinearRegression()
        m.fit(age.reshape(-1, 1), X[:, j])
        models.append(m)
    return models

def residuals(X: np.ndarray, age: np.ndarray, models: list[LinearRegression]) -> np.ndarray:
    R = np.zeros_like(X, dtype=float)
    a = age.reshape(-1, 1)
    for j, m in enumerate(models):
        R[:, j] = X[:, j] - m.predict(a)
    return R

def regularize_cov(R: np.ndarray, lam: float) -> np.ndarray:
    n, p = R.shape
    if n < 2:
        return np.eye(p)
    S = np.cov(R, rowvar=False)
    if np.ndim(S) == 0:
        S = np.array([[float(S)]])
    target = np.eye(S.shape[0]) * (np.trace(S) / max(S.shape[0], 1) + 1e-12)
    return (1.0 - lam) * S + lam * target

def mahalanobis(r: np.ndarray, Sigma: np.ndarray) -> float:
    inv = np.linalg.pinv(Sigma)
    val = float(r @ inv @ r)
    return float(np.sqrt(max(val, 0.0)))

def fit_block_models(td: pd.DataFrame, feats: list[str], lam: float):
    feats = [f for f in feats if f in td.columns]
    X = td[feats].to_numpy(dtype=float)
    age = td["age"].to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(age)
    X, age = X[mask], age[mask]
    models = fit_age_models(X, age)
    R = residuals(X, age, models)
    Sigma = regularize_cov(R, lam)
    return feats, models, Sigma

def score_row(row: pd.Series, block_fits: dict) -> dict:
    out = {
        "participant_id": row["participant_id"],
        "group": row["group"],
        "age": float(row["age"]),
    }
    ds = []
    for name, (feats, models, Sigma) in block_fits.items():
        x = row[feats].to_numpy(dtype=float).reshape(1, -1)
        if not np.isfinite(x).all():
            out[f"D_{name}"] = np.nan
            continue
        r = residuals(x, np.array([float(row["age"])]), models)[0]
        d = mahalanobis(r, Sigma)
        out[f"D_{name}"] = d
        ds.append(d)
    out["NAI"] = float(np.nanmean(ds)) if ds else np.nan
    return out

def main() -> None:
    print("=" * 70)
    print("C6 — Discovery TD normative → score Sheffield")
    print("=" * 70)

    td = _load_discovery_td()
    print(f"Discovery TD n = {len(td)}")
    print(f"Age range: {td['age'].min():.1f} – {td['age'].max():.1f}")

    missing = []
    for name, feats in ALL_BLOCKS.items():
        present = [f for f in feats if f in td.columns]
        print(f"  [{name}] {len(present)}/{len(feats)} features on discovery")
        if len(present) < len(feats):
            miss = set(feats) - set(present)
            missing.append((name, miss))
            print(f"       missing: {sorted(miss)[:8]}...")

    if any(len(m[1]) == len(ALL_BLOCKS[m[0]]) for m in missing):
        raise SystemExit("A whole block is missing on discovery — cannot score.")

    block_fits = {}
    for name, feats in ALL_BLOCKS.items():
        present = [f for f in feats if f in td.columns]
        if len(present) < 2:
            continue
        block_fits[name] = fit_block_models(td, present, LAMBDA)
        print(f"  fitted {name}: p={len(block_fits[name][0])}")

    sh = pd.read_csv(SHEFFIELD_PATH)
    sh = sh.dropna(subset=["age", "group"]).copy()
    sh = sh[sh["group"].isin(["ASD", "CTRL"])].reset_index(drop=True)
    print(f"Sheffield n = {len(sh)}  ASD={(sh.group=='ASD').sum()}  CTRL={(sh.group=='CTRL').sum()}")

    for name, (feats, _, _) in block_fits.items():
        miss = [f for f in feats if f not in sh.columns]
        if miss:
            raise SystemExit(f"Sheffield missing {name} cols: {miss[:10]}")

    rows = [score_row(sh.iloc[i], block_fits) for i in range(len(sh))]
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT_SCORES, index=False)

    y = (scores["group"] == "ASD").astype(int).to_numpy()
    s = scores["NAI"].to_numpy(dtype=float)
    ok = np.isfinite(s)
    auc = float(roc_auc_score(y[ok], s[ok])) if ok.sum() > 2 else float("nan")

    means = scores.groupby("group")["NAI"].agg(["mean", "std", "median"]).to_dict()
    summary = {
        "n_discovery_td": int(len(td)),
        "n_sheffield": int(len(scores)),
        "n_asd": int((scores.group == "ASD").sum()),
        "n_ctrl": int((scores.group == "CTRL").sum()),
        "lambda": LAMBDA,
        "auc_discovery_to_sheffield": auc,
        "mean_nai_asd": float(scores.loc[scores.group == "ASD", "NAI"].mean()),
        "mean_nai_ctrl": float(scores.loc[scores.group == "CTRL", "NAI"].mean()),
        "block_means_asd": {
            f"D_{b}": float(scores.loc[scores.group == "ASD", f"D_{b}"].mean())
            for b in block_fits
        },
        "block_means_ctrl": {
            f"D_{b}": float(scores.loc[scores.group == "CTRL", f"D_{b}"].mean())
            for b in block_fits
        },
        "note": (
            "Normative model fitted only on discovery TD; "
            "no Sheffield subject used in fit."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("-" * 70)
    print(json.dumps(summary, indent=2))
    print("Saved", OUT_SCORES)
    print("Saved", OUT_JSON)
    print("=" * 70)

if __name__ == "__main__":
    main()