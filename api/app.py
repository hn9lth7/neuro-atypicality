from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security import APIKeyHeader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nai.product.model_bundle import load_bundle
from nai.product.report import build_pass_report, build_reject_report
from nai.product.score import score_row
from nai.product.validate_input import AGE_REF_MAX, AGE_REF_MIN, validate_feature_row

MODEL_DIR = PROJECT_ROOT / "models" / "nai_v1"
BUNDLE = load_bundle(MODEL_DIR)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_KEYS = {"dev-key-change-me"}
DISCLAIMER = (
    "Research Use Only. Not a medical device. "
    "Not for diagnosis of ASD or any other condition."
)

app = FastAPI(
    title="NAI Scoring API",
    version="0.1.0",
    description="Pediatric normative atypicality scoring (reference ages 8–13). RUO.",
)

def require_key(key: str | None = Depends(API_KEY_HEADER)) -> str:
    if key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    return key

@app.get("/v1/health")
def health():
    return {
        "status": "ok",
        "model_version": BUNDLE.manifest.get("nai_version", "v1.0"),
        "lambda": BUNDLE.lambda_,
        "age_reference": [AGE_REF_MIN, AGE_REF_MAX],
        "blocks": BUNDLE.manifest.get("blocks"),
        "disclaimer": DISCLAIMER,
    }

@app.post("/v1/score", dependencies=[Depends(require_key)])
async def score(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot parse CSV: {e}") from e

    if df.empty:
        raise HTTPException(status_code=400, detail="empty CSV")

    results = []
    for i, row in df.iterrows():
        row_map = row.to_dict()
        vr = validate_feature_row(row_map)
        if not vr.ok:
            results.append(
                build_reject_report(
                    reason=vr.reason or "REJECT",
                    details={**vr.details, "row": int(i)},
                    warnings=vr.warnings,
                    stage="feature_row",
                    model_version=BUNDLE.manifest.get("nai_version"),
                )
            )
            continue
        scores = score_row(row, BUNDLE)
        results.append(
            build_pass_report(
                scores=scores,
                bundle=BUNDLE,
                validation=vr,
                participant_id=str(row_map.get("participant_id", i)),
                age=float(row_map["age"]),
                extra={"row": int(i)},
            )
        )

    return {
        "disclaimer": DISCLAIMER,
        "model_version": BUNDLE.manifest.get("nai_version", "v1.0"),
        "n_rows": len(results),
        "results": results,
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)