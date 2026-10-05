from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "results" / "features" / "sheffield_features_54d.csv"
OUT_DIR = PROJECT_ROOT / "results" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "c3a_sheffield_nested_cv_summary.json"
OUT_FOLDS = OUT_DIR / "c3a_sheffield_nested_cv_folds.csv"

SE = [
    "alpha_rel", "beta_rel", "theta_rel",
    "spectral_entropy_mean", "log_theta_alpha", "log_theta_beta",
]
C = [
    "plv_mean_theta", "plv_mean_alpha", "plv_mean_beta", "plv_mean_gamma",
    "plv_median_theta", "plv_median_alpha", "plv_median_beta", "plv_median_gamma",
]
G = [
    f"{m}_{b}"
    for m in (
        "mean_degree", "degree_cv", "clustering",
        "global_efficiency", "mean_path_length", "laplacian_entropy",
    )
    for b in ("theta", "alpha", "beta", "gamma")
]
D = [
    f"{m}_{b}"
    for m in (
        "mean_delta", "cv_delta", "mean_degree_cv", "temporal_cv_degree_cv",
    )
    for b in ("theta", "alpha", "beta", "gamma")
]
FEATURE_54 = SE + C + G + D

N_OUTER = 5
N_INNER = 3
RANDOM_STATE = 42

def load_xy(include_age: bool = True):
    df = pd.read_csv(FEATURES_PATH)
    df = df.dropna(subset=["age", "group"]).copy()
    df = df[df["group"].isin(["ASD", "CTRL"])].reset_index(drop=True)
    feats = [c for c in FEATURE_54 if c in df.columns]
    if len(feats) != 54:
        raise SystemExit(f"Expected 54 features, found {len(feats)}")
    X = df[feats].to_numpy(dtype=float)
    if include_age:
        age = df["age"].to_numpy(dtype=float).reshape(-1, 1)
        X = np.hstack([X, age])
        feat_names = feats + ["age"]
    else:
        feat_names = feats
    y = (df["group"] == "ASD").astype(int).to_numpy()
    ids = df["participant_id"].astype(str).to_numpy()
    return X, y, ids, feat_names, df

def make_logistic():
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs"],
    }
    return pipe, grid

def make_linear_svm():
    base = LinearSVC(
        class_weight="balanced",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(base, method="sigmoid", cv=3),
            ),
        ]
    )
    grid = {}
    return pipe, grid

def nested_cv_logistic(X, y, ids):
    outer = StratifiedKFold(
        n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE
    )
    rows = []
    oof_score = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1)

    for fold, (tr, te) in enumerate(outer.split(X, y)):
        pipe, grid = make_logistic()
        inner = StratifiedKFold(
            n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE
        )
        gs = GridSearchCV(
            pipe,
            grid,
            cv=inner,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X[tr], y[tr])
        proba = gs.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        oof_score[te] = proba
        oof_pred[te] = pred

        auc = roc_auc_score(y[te], proba)
        bacc = balanced_accuracy_score(y[te], pred)
        rows.append(
            {
                "model": "logistic",
                "fold": fold,
                "n_test": int(len(te)),
                "auc": float(auc),
                "balanced_acc": float(bacc),
                "best_C": float(gs.best_params_["clf__C"]),
            }
        )
        print(
            f"  logistic fold {fold}: AUC={auc:.3f}  bAcc={bacc:.3f}  C={gs.best_params_['clf__C']}"
        )

    overall_auc = float(roc_auc_score(y, oof_score))
    overall_bacc = float(balanced_accuracy_score(y, oof_pred))
    return rows, overall_auc, overall_bacc, oof_score

def nested_cv_svm(X, y):
    outer = StratifiedKFold(
        n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE
    )
    rows = []
    oof_score = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1)
    C_grid = [0.01, 0.1, 1.0, 10.0]

    for fold, (tr, te) in enumerate(outer.split(X, y)):
        best_auc_inner = -1.0
        best_C = 1.0
        inner = StratifiedKFold(
            n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE
        )
        for C in C_grid:
            inner_scores = []
            for itr, iva in inner.split(X[tr], y[tr]):
                tr_i = tr[itr]
                va_i = tr[iva]
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            CalibratedClassifierCV(
                                LinearSVC(
                                    C=C,
                                    class_weight="balanced",
                                    max_iter=10000,
                                    random_state=RANDOM_STATE,
                                ),
                                method="sigmoid",
                                cv=3,
                            ),
                        ),
                    ]
                )
                pipe.fit(X[tr_i], y[tr_i])
                p = pipe.predict_proba(X[va_i])[:, 1]
                if len(np.unique(y[va_i])) > 1:
                    inner_scores.append(roc_auc_score(y[va_i], p))
            mean_inner = float(np.mean(inner_scores)) if inner_scores else -1.0
            if mean_inner > best_auc_inner:
                best_auc_inner = mean_inner
                best_C = C

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(
                            C=best_C,
                            class_weight="balanced",
                            max_iter=10000,
                            random_state=RANDOM_STATE,
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        )
        pipe.fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        oof_score[te] = proba
        oof_pred[te] = pred
        auc = roc_auc_score(y[te], proba)
        bacc = balanced_accuracy_score(y[te], pred)
        rows.append(
            {
                "model": "linear_svm",
                "fold": fold,
                "n_test": int(len(te)),
                "auc": float(auc),
                "balanced_acc": float(bacc),
                "best_C": float(best_C),
            }
        )
        print(
            f"  svm fold {fold}: AUC={auc:.3f}  bAcc={bacc:.3f}  C={best_C}"
        )

    overall_auc = float(roc_auc_score(y, oof_score))
    overall_bacc = float(balanced_accuracy_score(y, oof_pred))
    return rows, overall_auc, overall_bacc

def main():
    print("=" * 70)
    print("3a — Nested CV supervised on Sheffield 54-D + age")
    print("=" * 70)
    X, y, ids, names, df = load_xy(include_age=True)
    print(f"n={len(y)}  ASD={int(y.sum())}  CTRL={int((y==0).sum())}  p={X.shape[1]}")

    all_rows = []

    print("-" * 70)
    print("LogisticRegression (balanced)")
    rows_lr, auc_lr, bacc_lr, _ = nested_cv_logistic(X, y, ids)
    all_rows.extend(rows_lr)
    print(f"  OOF AUC={auc_lr:.4f}  OOF balanced_acc={bacc_lr:.4f}")

    print("-" * 70)
    print("Linear SVM + calibration")
    rows_svm, auc_svm, bacc_svm = nested_cv_svm(X, y)
    all_rows.extend(rows_svm)
    print(f"  OOF AUC={auc_svm:.4f}  OOF balanced_acc={bacc_svm:.4f}")

    folds_df = pd.DataFrame(all_rows)
    folds_df.to_csv(OUT_FOLDS, index=False)

    summary = {
        "n": int(len(y)),
        "n_asd": int(y.sum()),
        "n_ctrl": int((y == 0).sum()),
        "n_features": int(X.shape[1]),
        "features": "54-D + age",
        "outer_folds": N_OUTER,
        "inner_folds": N_INNER,
        "logistic_oof_auc": auc_lr,
        "logistic_oof_balanced_acc": bacc_lr,
        "logistic_fold_auc_mean": float(folds_df.loc[folds_df.model == "logistic", "auc"].mean()),
        "logistic_fold_auc_std": float(folds_df.loc[folds_df.model == "logistic", "auc"].std()),
        "svm_oof_auc": auc_svm,
        "svm_oof_balanced_acc": bacc_svm,
        "svm_fold_auc_mean": float(folds_df.loc[folds_df.model == "linear_svm", "auc"].mean()),
        "svm_fold_auc_std": float(folds_df.loc[folds_df.model == "linear_svm", "auc"].std()),
        "note": (
            "Nested stratified CV on Sheffield only. "
            "Pilot n=46; not external validation of NAI. "
            "No discovery labels used for training."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print("Saved", OUT_FOLDS)
    print("Saved", OUT_JSON)

if __name__ == "__main__":
    main()