from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from nai.product.model_bundle import load_bundle
from nai.product.validate_input import validate_feature_row
from nai.product.score import score_row
from nai.product.report import (
    build_pass_report,
    build_reject_report,
    write_report,
)

def row_to_mapping(series: pd.Series) -> dict:
    return {k: series[k] for k in series.index}

def score_one(
    row: dict,
    bundle,
) -> dict:
    vr = validate_feature_row(row)
    model_ver = bundle.manifest.get("nai_version", "v1.0")

    if not vr.ok:
        return build_reject_report(
            reason=vr.reason or "REJECT",
            details=vr.details,
            warnings=list(vr.warnings),
            stage="feature_row",
            model_version=model_ver,
        )

    sc = score_row(row, bundle)
    return build_pass_report(
        scores=sc,
        bundle=bundle,
        validation=vr,
        participant_id=(
            str(row["participant_id"])
            if "participant_id" in row and pd.notna(row.get("participant_id"))
            else None
        ),
        age=float(row["age"]),
    )

def main() -> int:
    p = argparse.ArgumentParser(description="NAI v1 product score (feature-row path)")
    p.add_argument(
        "--features",
        type=Path,
        required=True,
        help="CSV with age + 54 feature columns",
    )
    p.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "models" / "nai_v1",
        help="Path to frozen model directory",
    )
    p.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row index in CSV (default 0)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON report to this path",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Score every row; --out is a directory of JSON files",
    )
    args = p.parse_args()

    if not args.features.exists():
        print(json.dumps({"status": "REJECT", "reason": "FILE_NOT_FOUND",
                          "path": str(args.features)}), flush=True)
        return 2

    bundle = load_bundle(args.bundle)
    df = pd.read_csv(args.features)

    if args.all:
        out_dir = args.out or (ROOT / "results" / "product" / "reports")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        n_pass = n_rej = 0
        for i, ser in df.iterrows():
            rep = score_one(row_to_mapping(ser), bundle)
            pid = rep.get("participant_id") or f"row{i}"
            path = out_dir / f"{pid}.json"
            write_report(rep, path)
            if rep["status"] == "PASS":
                n_pass += 1
            else:
                n_rej += 1
            print(f"{pid}: {rep['status']}", flush=True)
        print(f"done PASS={n_pass} REJECT={n_rej} → {out_dir}", flush=True)
        return 0 if n_rej == 0 else 1

    if args.row < 0 or args.row >= len(df):
        print(json.dumps({
            "status": "REJECT",
            "reason": "ROW_OUT_OF_RANGE",
            "row": args.row,
            "n_rows": len(df),
        }), flush=True)
        return 2

    rep = score_one(row_to_mapping(df.iloc[args.row]), bundle)
    text = json.dumps(rep, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        write_report(rep, args.out)
        print(f"Wrote {args.out}", file=sys.stderr)
    return 0 if rep["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())