from __future__ import annotations

from typing import Any, Mapping
import json
from pathlib import Path

from nai.product.validate_input import ValidationResult
from nai.product.model_bundle import ModelBundle

def build_reject_report(
    *,
    reason: str,
    details: Mapping[str, Any] | None = None,
    warnings: list[str] | None = None,
    stage: str = "validation",
    model_version: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "REJECT",
        "stage": stage,
        "reason": reason,
        "details": dict(details or {}),
        "warnings": list(warnings or []),
        "model_version": model_version,
        "scores": None,
    }

def build_pass_report(
    *,
    scores: Mapping[str, float],
    bundle: ModelBundle,
    validation: ValidationResult | None = None,
    participant_id: str | None = None,
    age: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    warnings = list(validation.warnings) if validation else []
    return {
        "status": "PASS",
        "stage": "score",
        "reason": None,
        "details": dict(validation.details) if validation else {},
        "warnings": warnings,
        "model_version": bundle.manifest.get("nai_version", "v1.0"),
        "lambda": bundle.lambda_,
        "participant_id": participant_id,
        "age": age,
        "scores": {
            "D_SE": float(scores["D_SE"]),
            "D_C": float(scores["D_C"]),
            "D_G": float(scores["D_G"]),
            "D_D": float(scores["D_D"]),
            "NAI": float(scores["NAI"]),
        },
        "extra": dict(extra or {}),
    }

def report_from_validation(
    vr: ValidationResult,
    *,
    stage: str = "feature_row",
    model_version: str | None = None,
) -> dict[str, Any]:
    if not vr.ok:
        return build_reject_report(
            reason=vr.reason or "REJECT",
            details=vr.details,
            warnings=vr.warnings,
            stage=stage,
            model_version=model_version,
        )
    return {
        "status": "PASS",
        "stage": stage,
        "reason": None,
        "details": dict(vr.details),
        "warnings": list(vr.warnings),
        "model_version": model_version,
        "scores": None,
    }

def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path