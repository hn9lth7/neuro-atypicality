from __future__ import annotations

def add_log_ratios(powers: dict[str, float], eps: float = 1e-12) -> dict[str, float]:
    out = dict(powers)
    out["log_theta_alpha"] = float(
        __import__("numpy").log((powers["theta_abs"] + eps) / (powers["alpha_abs"] + eps))
    )
    out["log_theta_beta"] = float(
        __import__("numpy").log((powers["theta_abs"] + eps) / (powers["beta_abs"] + eps))
    )
    out["theta_alpha"] = powers["theta_abs"] / (powers["alpha_abs"] + eps)
    out["theta_beta"] = powers["theta_abs"] / (powers["beta_abs"] + eps)
    out["alpha_beta"] = powers["alpha_abs"] / (powers["beta_abs"] + eps)
    return out