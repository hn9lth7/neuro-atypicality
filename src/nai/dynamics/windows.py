from __future__ import annotations

def sliding_windows(
    n_times: int,
    sfreq: float,
    window_s: float = 10.0,
    step_s: float = 5.0,
) -> list[tuple[int, int]]:
    w = int(round(window_s * sfreq))
    step = int(round(step_s * sfreq))
    if w <= 0 or step <= 0 or w > n_times:
        return []
    starts = range(0, n_times - w + 1, step)
    return [(s, s + w) for s in starts]