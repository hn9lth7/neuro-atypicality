from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import math

from nai.features.blocks import ALL_BLOCKS

AGE_REF_MIN = 8.0
AGE_REF_MAX = 13.0

@dataclass(frozen=True)
class ValidationResult:
    status: str 
    reason: str | None
    details: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "PASS"

def _reject(reason: str, **details: Any) -> ValidationResult:
    return ValidationResult(
        status="REJECT",
        reason=reason,
        details=details,
        warnings=[],
    )

def validate_feature_row(row: Mapping[str, Any]) -> ValidationResult:
    warnings: list[str] = []

    if "age" not in row:
        return _reject("MISSING_AGE", field="age")

    try:
        age = float(row["age"])
    except (TypeError, ValueError):
        return _reject("INVALID_AGE", field="age", value=row["age"])

    if not math.isfinite(age):
        return _reject("INVALID_AGE", field="age", value=age)

    if not (AGE_REF_MIN <= age <= AGE_REF_MAX):
        warnings.append(
            f"AGE_OUT_OF_RANGE: age={age} "
            f"(reference [{AGE_REF_MIN}, {AGE_REF_MAX}])"
        )

    missing: list[str] = []
    invalid: list[str] = []

    for features in ALL_BLOCKS.values():
        for feature in features:
            if feature not in row:
                missing.append(feature)
                continue
            try:
                value = float(row[feature])
            except (TypeError, ValueError):
                invalid.append(feature)
                continue
            if not math.isfinite(value):
                invalid.append(feature)

    if missing:
        return _reject(
            "MISSING_FEATURES",
            missing_features=missing,
            n_missing=len(missing),
        )
    if invalid:
        return _reject(
            "INVALID_FEATURES",
            invalid_features=invalid,
            n_invalid=len(invalid),
        )

    return ValidationResult(
        status="PASS",
        reason=None,
        details={
            "age": age,
            "feature_dim": sum(len(v) for v in ALL_BLOCKS.values()),
            "blocks": {b: len(f) for b, f in ALL_BLOCKS.items()},
            "age_in_reference_range": AGE_REF_MIN <= age <= AGE_REF_MAX,
        },
        warnings=warnings,
    )