from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEAT = PROJECT_ROOT / "results" / "features"
NORM = PROJECT_ROOT / "results" / "normative"
OUT = NORM / "robustness_v12"
OUT.mkdir(parents=True, exist_ok=True)

EXTREMES = ["sub-11936", "sub-2713"]
AGE_EXTRA = ["sub-11325"]  

def load() -> tuple[pd.DataFrame, float, float]:
    nai = pd.read_csv(NORM / "nai_v12_expanded.csv")
    se = pd.read_csv(FEAT / "participants_features_subject_v0.3_clean.csv")[
        ["participant_id", "n_runs", "qc_flag"]
    ]
    df = nai.merge(se, on="participant_id", how="left")

    loo = pd.read_csv(NORM / "loo_nai_v12_td.csv")
    p95 = float(np.percentile(loo["NAI_LOO"], 95))
    p99 = float(np.percentile(loo["NAI_LOO"], 99))
    return df, p95, p99

def asd_stats(asd: pd.DataFrame, p95: float, p99: float, label: str) -> dict:
    n = len(asd)
    if n == 0:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": n,
        "median_NAI": float(asd["NAI"].median()),
        "mean_NAI": float(asd["NAI"].mean()),
        "max_NAI": float(asd["NAI"].max()),
        "n_above_LOO_P95": int((asd["NAI"] > p95).sum()),
        "pct_above_LOO_P95": float(100.0 * (asd["NAI"] > p95).mean()),
        "n_above_LOO_P99": int((asd["NAI"] > p99).sum()),
        "pct_above_LOO_P99": float(100.0 * (asd["NAI"] > p99).mean()),
    }

def main() -> None:
    print("=" * 72)
    print("NAI v1.2 — Extremes + run-count sensitivity")
    print("=" * 72)

    df, p95, p99 = load()
    print(f"LOO P95={p95:.3f}  P99={p99:.3f}")

    asd = df[df["group"] == "ASD"].copy()
    td = df[df["group"] == "TD"].copy()

    asd["flag_extreme"] = asd["participant_id"].isin(EXTREMES)
    asd["flag_age_extrapolation"] = asd["participant_id"].isin(AGE_EXTRA)

    rows = [
        asd_stats(asd, p95, p99, "all_ASD"),
        asd_stats(asd[~asd["participant_id"].isin(["sub-11936"])], p95, p99, "wo_11936"),
        asd_stats(asd[~asd["participant_id"].isin(["sub-2713"])], p95, p99, "wo_2713"),
        asd_stats(asd[~asd["participant_id"].isin(EXTREMES)], p95, p99, "wo_11936_2713"),
        asd_stats(
            asd[~asd["participant_id"].isin(EXTREMES + AGE_EXTRA)],
            p95,
            p99,
            "wo_extremes_and_age_extra",
        ),
    ]
    ext = pd.DataFrame(rows)
    ext_path = OUT / "extreme_sensitivity.csv"
    ext.to_csv(ext_path, index=False)
    print("\n--- Extreme sensitivity ---")
    print(ext.to_string(index=False))
    print(f"Saved → {ext_path}")

    run_rows = []
    for label, mask in [
        ("n_runs==1", asd["n_runs"] == 1),
        ("n_runs>=2", asd["n_runs"] >= 2),
        ("n_runs>=3", asd["n_runs"] >= 3),
        ("n_runs>=5", asd["n_runs"] >= 5),
    ]:
        run_rows.append(asd_stats(asd[mask], p95, p99, f"ASD_{label}"))
    for label, mask in [
        ("n_runs==1", td["n_runs"] == 1),
        ("n_runs>=2", td["n_runs"] >= 2),
        ("n_runs>=3", td["n_runs"] >= 3),
    ]:
        sub = td[mask]
        run_rows.append(
            {
                "label": f"TD_{label}",
                "n": len(sub),
                "median_NAI": float(sub["NAI"].median()) if len(sub) else np.nan,
                "mean_NAI": float(sub["NAI"].mean()) if len(sub) else np.nan,
                "max_NAI": float(sub["NAI"].max()) if len(sub) else np.nan,
                "n_above_LOO_P95": int((sub["NAI"] > p95).sum()) if len(sub) else 0,
                "pct_above_LOO_P95": float(100.0 * (sub["NAI"] > p95).mean())
                if len(sub)
                else np.nan,
                "n_above_LOO_P99": int((sub["NAI"] > p99).sum()) if len(sub) else 0,
                "pct_above_LOO_P99": float(100.0 * (sub["NAI"] > p99).mean())
                if len(sub)
                else np.nan,
            }
        )
    runs = pd.DataFrame(run_rows)
    runs_path = OUT / "run_count_sensitivity.csv"
    runs.to_csv(runs_path, index=False)
    print("\n--- Run-count sensitivity ---")
    print(runs.to_string(index=False))
    print(f"Saved → {runs_path}")

    one = asd[asd["n_runs"] == 1][
        ["participant_id", "age", "NAI", "D_SE", "D_C", "D_G", "D_D", "n_runs"]
    ]
    print("\n--- ASD with n_runs == 1 ---")
    print(one.sort_values("NAI", ascending=False).to_string(index=False))
    one.to_csv(OUT / "asd_single_run.csv", index=False)

    summary = {
        "loo_p95": p95,
        "loo_p99": p99,
        "extreme_ids": EXTREMES,
        "age_extrapolation_ids": AGE_EXTRA,
        "extreme_table": rows,
    }
    with open(OUT / "extremes_runs_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {OUT / 'extremes_runs_summary.json'}")
    print("=" * 72)

if __name__ == "__main__":
    main()