from __future__ import annotations

from dataclasses import dataclass

@dataclass
class BlockDistances:
    D_SE: float
    D_C: float
    D_G: float
    D_D: float

    def as_dict(self) -> dict[str, float]:
        return {
            "D_SE": float(self.D_SE),
            "D_C": float(self.D_C),
            "D_G": float(self.D_G),
            "D_D": float(self.D_D),
        }