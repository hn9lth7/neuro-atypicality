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

FEATURES_PATH = PROJECT_ROOT / "results" / "features" / "sheffield_features_54d.csv"
OUT_DIR = PROJECT_ROOT / "results" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SCORES = OUT_DIR / "c5_sheffield_exact_loo_scores.csv"
OUT_INSAMPLE = OUT_DIR / "c5_sheffield_insample_scores.csv"
OUT_JSON = OUT_DIR / "c5_sheffield_exact_loo_summary.json"

LAMBDA = 0.10
TD_LABEL = "CTRL"
ASD_LABEL = "ASD"

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
    for m in (
        "mean_degree",
        "degree_cv",
        "clustering",
        "global_efficiency",
        "mean_path_length",
        "laplacian_entropy",
    )
    for b in ("theta", "alpha", "beta", "gamma")
]
D_FEATURES = [
    f"{m}_{b}"
    for m in (
        "mean_delta",
        "cv_delta",
        "mean_degree_cv",
        "temporal_cv_degree_cv",
    )
    for b in ("theta", "alpha", "beta", "gamma")
]

def _block_lists(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "SE": [c for c in SE_FEATURES if c in df.columns],
        "C": [c for c in C_FEATURES if c in df.columns],
        "G": [c for c in G_FEATURES if c in df.columns],
        "D": [c for c in D_FEATURES if c in df.columns],
    }

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

def score_one(
    row: pd.Series,
    fit_df: pd.DataFrame,
    blocks: dict[str, list[str]],
    lam: float,
) -> dict:
    age_fit = fit_df["age"].to_numpy(dtype=float)
    out = {
        "participant_id": row["participant_id"],
        "group": row["group"],
        "age": float(row["age"]),
    }
    distances = []
    for name, feats in blocks.items():
        feats = [f for f in feats if f in fit_df.columns]
        if len(feats) < 1:
            out[f"D_{name}"] = np.nan
            continue
        X_fit = fit_df[feats].to_numpy(dtype=float)
        mask = np.isfinite(X_fit).all(axis=1) & np.isfinite(age_fit)
        X_ok = X_fit[mask]
        age_ok = age_fit[mask]
        if len(X_ok) < 3:
            out[f"D_{name}"] = np.nan
            continue
        models = fit_age_models(X_ok, age_ok)
        R_fit = residuals(X_ok, age_ok, models)
        Sigma = regularize_cov(R_fit, lam)
        x = row[feats].to_numpy(dtype=float).reshape(1, -1)
        if not np.isfinite(x).all():
            out[f"D_{name}"] = np.nan
            continue
        r = residuals(x, np.array([float(row["age"])]), models)[0]
        d = mahalanobis(r, Sigma)
        out[f"D_{name}"] = d
        distances.append(d)
    out["NAI_LOO"] = float(np.nanmean(distances)) if distances else np.nan
    return out

def auc_from_scores(scores_df: pd.DataFrame, score_col: str = "NAI_LOO") -> float:
    y = (scores_df["group"] == ASD_LABEL).astype(int).to_numpy()
    s = scores_df[score_col].to_numpy(dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))

def main() -> None:
    if not FEATURES_PATH.exists():
        raise SystemExit(f"Missing file: {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH)
    df = df.dropna(subset=["age", "group"]).copy()
    df = df[df["group"].isin([ASD_LABEL, TD_LABEL])].reset_index(drop=True)

    blocks = _block_lists(df)
    print("Blocks:", {k: len(v) for k, v in blocks.items()})
    print(
        "n subjects:",
        len(df),
        "ASD:",
        int((df["group"] == ASD_LABEL).sum()),
        "CTRL:",
        int((df["group"] == TD_LABEL).sum()),
    )

    print("-" * 60)
    print("EXACT LOO")
    rows = []
    for i in range(len(df)):
        test = df.iloc[i]
        if test["group"] == TD_LABEL:
            fit_df = df[(df["group"] == TD_LABEL) & (df.index != df.index[i])]
        else:
            fit_df = df[df["group"] == TD_LABEL]
        if len(fit_df) < 5:
            print("WARN: too few CTRL, skip", test["participant_id"])
            continue
        rows.append(score_one(test, fit_df, blocks, LAMBDA))
        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(df)}")

    loo = pd.DataFrame(rows)
    loo.to_csv(OUT_SCORES, index=False)
    auc_loo = auc_from_scores(loo)

    summary = {
        "n": int(len(loo)),
        "n_asd": int((loo["group"] == ASD_LABEL).sum()),
        "n_ctrl": int((loo["group"] == TD_LABEL).sum()),
        "lambda": LAMBDA,
        "auc_exact_loo": auc_loo,
        "nai_loo_mean_asd": float(loo.loc[loo["group"] == ASD_LABEL, "NAI_LOO"].mean()),
        "nai_loo_mean_ctrl": float(loo.loc[loo["group"] == TD_LABEL, "NAI_LOO"].mean()),
        "blocks": {k: len(v) for k, v in blocks.items()},
        "note": (
            "Exact LOO: CTRL left-out excluded from normative fit; "
            "ASD scored against full Sheffield CTRL set."
        ),
    }
    print(json.dumps({k: summary[k] for k in ("auc_exact_loo", "nai_loo_mean_asd", "nai_loo_mean_ctrl")}, indent=2))
    print("Saved", OUT_SCORES)

    print("-" * 60)
    print("IN-SAMPLE (fit on all CTRL, score all)")
    fit_all = df[df["group"] == TD_LABEL]
    rows_is = [score_one(df.iloc[i], fit_all, blocks, LAMBDA) for i in range(len(df))]
    is_df = pd.DataFrame(rows_is)
    is_df = is_df.rename(columns={"NAI_LOO": "NAI_INSAMPLE"})
    for c in list(is_df.columns):
        if c.startswith("D_"):
            is_df = is_df.rename(columns={c: c.replace("D_", "D_") + "_IS" if False else c})
    is_df.to_csv(OUT_INSAMPLE, index=False)

    tmp = is_df.rename(columns={"NAI_INSAMPLE": "NAI_LOO"})
    auc_is = auc_from_scores(tmp)
    means = is_df.groupby("group")["NAI_INSAMPLE"].mean().to_dict()
    print(f"in-sample AUC (same scorer): {auc_is:.4f}")
    print(f"mean NAI ASD/CTRL: {means}")
    print("Saved", OUT_INSAMPLE)

    summary["auc_insample_same_scorer"] = auc_is
    summary["nai_insample_mean_asd"] = float(means.get(ASD_LABEL, float("nan")))
    summary["nai_insample_mean_ctrl"] = float(means.get(TD_LABEL, float("nan")))
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("-" * 60)
    print(json.dumps(summary, indent=2))
    print("Saved", OUT_JSON)

if __name__ == "__main__":
    main()