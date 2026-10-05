from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VAL = PROJECT_ROOT / "results" / "validation"
NORM = PROJECT_ROOT / "results" / "normative"
FIG = PROJECT_ROOT / "results" / "figures" / "validation_b2"
EPS = 1e-8

def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(VAL / "bootstrap_summary_b1.csv")
    samples = pd.read_csv(VAL / "bootstrap_nai_samples_b1.csv")
    frozen = pd.read_csv(NORM / "nai_v10.csv")
    nai_col = "NAI_v10" if "NAI_v10" in frozen.columns else "NAI"
    frz = frozen.set_index("participant_id")[nai_col]

    df = summary.copy()
    if "NAI_frozen" not in df.columns or df["NAI_frozen"].isna().any():
        df["NAI_frozen"] = df["participant_id"].map(frz)

    df["bias_boot_minus_frozen"] = df["NAI_boot_mean"] - df["NAI_frozen"]
    df["bias_boot_minus_point"] = df["NAI_boot_mean"] - df["NAI_point"]

    df["CI_width_rel"] = df["NAI_CI_width"] / np.maximum(df["NAI_point"].abs(), EPS)

    df["frozen_in_CI"] = (
        (df["NAI_frozen"] >= df["NAI_q025"]) & (df["NAI_frozen"] <= df["NAI_q975"])
    )

    out_csv = VAL / "uncertainty_summary_b2.csv"
    df.to_csv(out_csv, index=False)

    td = df[df["group"] == "TD"]
    asd = df[df["group"] == "ASD"]

    meta = {
        "n_subjects": int(len(df)),
        "n_td": int(len(td)),
        "n_asd": int(len(asd)),
        "CI_width_median_all": float(df["NAI_CI_width"].median()),
        "CI_width_mean_all": float(df["NAI_CI_width"].mean()),
        "CI_width_median_TD": float(td["NAI_CI_width"].median()),
        "CI_width_median_ASD": float(asd["NAI_CI_width"].median()) if len(asd) else None,
        "CI_width_rel_median_all": float(df["CI_width_rel"].median()),
        "bias_median_abs": float(df["bias_boot_minus_frozen"].abs().median()),
        "bias_max_abs": float(df["bias_boot_minus_frozen"].abs().max()),
        "frac_frozen_in_CI": float(df["frozen_in_CI"].mean()),
        "note": (
            "CI = 2.5–97.5% quantiles of NAI under TD-reference bootstrap; "
            "subjects are scored with models fit on resampled TD only."
        ),
    }

    meta["asd_detail"] = (
        asd[
            [
                "participant_id",
                "NAI_frozen",
                "NAI_point",
                "NAI_boot_mean",
                "NAI_q025",
                "NAI_q975",
                "NAI_CI_width",
                "CI_width_rel",
                "bias_boot_minus_frozen",
                "frozen_in_CI",
            ]
        ]
        .assign(**{c: asd[c] for c in asd.columns if c in asd})
        .to_dict(orient="records")
    )

    meta["asd_detail"] = asd[
        [
            "participant_id",
            "NAI_frozen",
            "NAI_point",
            "NAI_boot_mean",
            "NAI_q025",
            "NAI_q975",
            "NAI_CI_width",
            "CI_width_rel",
            "bias_boot_minus_frozen",
            "frozen_in_CI",
        ]
    ].to_dict(orient="records")

    top_wide = df.nlargest(5, "NAI_CI_width")[
        ["participant_id", "group", "NAI_point", "NAI_CI_width", "CI_width_rel"]
    ]
    meta["top5_widest_CI"] = top_wide.to_dict(orient="records")

    with open(VAL / "uncertainty_summary_b2.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=float)

    print("=" * 72)
    print("B2 — Uncertainty summary")
    print("=" * 72)
    print(f"Median CI width (all): {meta['CI_width_median_all']:.4f}")
    print(f"Median CI width (TD):  {meta['CI_width_median_TD']:.4f}")
    if meta["CI_width_median_ASD"] is not None:
        print(f"Median CI width (ASD): {meta['CI_width_median_ASD']:.4f}")
    print(f"Median |bias| boot−frozen: {meta['bias_median_abs']:.4e}")
    print(f"Max |bias| boot−frozen:    {meta['bias_max_abs']:.4e}")
    print(f"Fraction frozen ∈ CI:      {meta['frac_frozen_in_CI']:.3f}")
    print("-" * 72)
    print("ASD detail:")
    print(
        asd[
            [
                "participant_id",
                "NAI_frozen",
                "NAI_q025",
                "NAI_q975",
                "NAI_CI_width",
                "CI_width_rel",
                "bias_boot_minus_frozen",
                "frozen_in_CI",
            ]
        ].to_string(index=False)
    )
    print("-" * 72)
    print("Top 5 widest CI:")
    print(top_wide.to_string(index=False))
    print(f"Saved → {out_csv}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["NAI_CI_width"], bins=20, edgecolor="black", alpha=0.85)
    ax.axvline(meta["CI_width_median_all"], color="C1", label="median")
    ax.set_xlabel("Bootstrap 95% CI width (NAI)")
    ax.set_ylabel("Subjects")
    ax.set_title("B2 — NAI uncertainty width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "ci_width_hist.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["C3" if g == "ASD" else "C0" for g in df["group"]]
    ax.scatter(df["NAI_CI_width"], df["bias_boot_minus_frozen"], c=colors, s=28)
    ax.axhline(0, color="0.5", lw=1)
    ax.set_xlabel("CI width")
    ax.set_ylabel("Bias (boot mean − frozen)")
    ax.set_title("B2 — bias vs interval width")
    fig.tight_layout()
    fig.savefig(FIG / "bias_vs_width.png", dpi=140)
    plt.close(fig)

    if len(asd):
        fig, ax = plt.subplots(figsize=(6, 3 + 0.4 * len(asd)))
        y = np.arange(len(asd))
        ax.hlines(y, asd["NAI_q025"], asd["NAI_q975"], color="0.5", lw=2)
        ax.scatter(asd["NAI_frozen"], y, c="C3", s=40, label="frozen", zorder=3)
        ax.scatter(asd["NAI_boot_mean"], y, c="black", s=20, marker="x", label="boot mean")
        ax.set_yticks(y)
        ax.set_yticklabels(asd["participant_id"])
        ax.set_xlabel("NAI")
        ax.set_title("B2 — ASD bootstrap intervals")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "asd_ci_detail.png", dpi=140)
        plt.close(fig)

    print(f"Figures → {FIG}")
    print("B2 finished (no refit; frozen core untouched).")
    print("=" * 72)

if __name__ == "__main__":
    main()