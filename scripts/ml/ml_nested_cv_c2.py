from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV = PROJECT_ROOT / "results" / "features" / "features_54d_development_c1b.csv"
OUT_DIR = PROJECT_ROOT / "results" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_OUTER = 5
N_INNER = 5

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
    for m in [
        "mean_degree", "degree_cv", "clustering",
        "global_efficiency", "mean_path_length", "laplacian_entropy",
    ]
    for b in ["theta", "alpha", "beta", "gamma"]
]
D = [
    f"{m}_{b}"
    for m in [
        "mean_delta", "cv_delta", "mean_degree_cv", "temporal_cv_degree_cv",
    ]
    for b in ["theta", "alpha", "beta", "gamma"]
]
FEATURES = SE + C + G + D
assert len(FEATURES) == 54

def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))

def get_search_spaces():
    lr = Pipeline(
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
    lr_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs"],
    }

    svm_base = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=10000,
                    dual="auto",
                ),
            ),
        ]
    )
    svm_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}

    rf = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False, with_std=False)),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 3],
    }

    return {
        "logreg": (lr, lr_grid),
        "linear_svm": (svm_base, svm_grid),
        "rf": (rf, rf_grid),
    }

def scores_from_model(model, X, name: str) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    
    return model.predict(X).astype(float)

def main() -> None:
    print("=" * 72)
    print("C2 — Nested CV (R0, 54-D development)")
    print("=" * 72)

    df = pd.read_csv(CSV)
    y = (df["group"].astype(str).str.upper() == "ASD").astype(int).to_numpy()
    X = df[FEATURES].to_numpy(dtype=float)
    ids = df["participant_id"].astype(str).to_numpy()

    print(f"n={len(y)}  ASD={int(y.sum())}  TD={int((y == 0).sum())}  p={X.shape[1]}")

    outer = StratifiedKFold(
        n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE
    )
    spaces = get_search_spaces()

    rows = []
    confusions = {name: np.zeros((2, 2), dtype=int) for name in spaces}

    for fold_i, (tr, te) in enumerate(outer.split(X, y), start=1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        print(f"\n--- Outer fold {fold_i}/{N_OUTER}  "
              f"train={len(tr)} test={len(te)} ASD_test={int(y_te.sum())}")

        inner = StratifiedKFold(
            n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE + fold_i
        )

        for name, (pipe, grid) in spaces.items():
            gs = GridSearchCV(
                pipe,
                grid,
                scoring="roc_auc",
                cv=inner,
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X_tr, y_tr)
            best = gs.best_estimator_

            if name == "linear_svm":
                model = best
            else:
                model = best

            y_score = scores_from_model(model, X_te, name)
            y_pred = model.predict(X_te)

            try:
                roc = float(roc_auc_score(y_te, y_score))
            except ValueError:
                roc = float("nan")
            try:
                pra = pr_auc(y_te, y_score)
            except ValueError:
                pra = float("nan")

            bal = float(balanced_accuracy_score(y_te, y_pred))
            f1 = float(f1_score(y_te, y_pred, pos_label=1, zero_division=0))
            acc = float(accuracy_score(y_te, y_pred))
            cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
            confusions[name] += cm

            row = {
                "model": name,
                "outer_fold": fold_i,
                "n_test": int(len(te)),
                "n_asd_test": int(y_te.sum()),
                "roc_auc": roc,
                "pr_auc": pra,
                "balanced_accuracy": bal,
                "f1_asd": f1,
                "accuracy": acc,
                "best_params": json.dumps(gs.best_params_),
                "best_inner_roc_auc": float(gs.best_score_),
            }
            rows.append(row)
            print(
                f"  {name:12s}  ROC={roc:.3f}  PR={pra:.3f}  "
                f"bal={bal:.3f}  F1={f1:.3f}  inner={gs.best_score_:.3f}"
            )

    metrics = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "c2_outer_fold_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    summary = {"n": int(len(y)), "n_asd": int(y.sum()), "n_td": int((y == 0).sum()), "models": {}}
    for name in spaces:
        sub = metrics[metrics["model"] == name]
        summary["models"][name] = {
            "roc_auc_mean": float(sub["roc_auc"].mean()),
            "roc_auc_std": float(sub["roc_auc"].std(ddof=1)),
            "pr_auc_mean": float(sub["pr_auc"].mean()),
            "pr_auc_std": float(sub["pr_auc"].std(ddof=1)),
            "balanced_accuracy_mean": float(sub["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(sub["balanced_accuracy"].std(ddof=1)),
            "f1_asd_mean": float(sub["f1_asd"].mean()),
            "f1_asd_std": float(sub["f1_asd"].std(ddof=1)),
        }

    summary_path = OUT_DIR / "c2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    cm_rows = []
    for name, cm in confusions.items():
        cm_rows.append(
            {
                "model": name,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )
    cm_df = pd.DataFrame(cm_rows)
    cm_path = OUT_DIR / "c2_confusion_sum.csv"
    cm_df.to_csv(cm_path, index=False)

    print("\n" + "=" * 72)
    print("SUMMARY (outer-fold means ± std)")
    print("=" * 72)
    for name, s in summary["models"].items():
        print(
            f"{name:12s}  ROC {s['roc_auc_mean']:.3f}±{s['roc_auc_std']:.3f}  "
            f"PR {s['pr_auc_mean']:.3f}±{s['pr_auc_std']:.3f}  "
            f"bal {s['balanced_accuracy_mean']:.3f}±{s['balanced_accuracy_std']:.3f}  "
            f"F1 {s['f1_asd_mean']:.3f}±{s['f1_asd_std']:.3f}"
        )
    print(f"\nSaved → {metrics_path}")
    print(f"Saved → {summary_path}")
    print(f"Saved → {cm_path}")
    print("Reporting: cross-validated performance on ds006780 development set only.")
    print("=" * 72)

if __name__ == "__main__":
    main()