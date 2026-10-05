from __future__ import annotations

from typing import Any

def qc_to_row(qc: dict[str, Any], **extra: Any) -> dict[str, Any]:
    row = dict(qc)
    row.update(extra)
    return row