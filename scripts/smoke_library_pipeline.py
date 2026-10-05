from __future__ import annotations

import numpy as np
import mne

from nai.preprocessing import preprocess_minimal
from nai.spectral import compute_band_powers, compute_spectral_entropy
from nai.connectivity.phase import compute_plv_band
from nai.graph.metrics import graph_metrics_dict
from nai.dynamics import dynamic_summary, transition_series

def main() -> None:
    sfreq = 128.0
    n_ch, duration = 16, 20.0
    n_times = int(sfreq * duration)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_ch, n_times)) * 1e-6
    info = mne.create_info(
        ch_names=[f"E{i}" for i in range(n_ch)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    raw = preprocess_minimal(raw, set_montage=False)

    powers = compute_band_powers(raw)
    entropy = compute_spectral_entropy(raw)
    print("SE sample:", {k: round(powers[k], 4) for k in ("alpha_rel", "theta_rel", "log_theta_alpha")})
    print("Entropy:", {k: round(v, 4) for k, v in entropy.items()})

    x = raw.get_data()
    W = compute_plv_band(x, sfreq, 8.0, 13.0)
    g = graph_metrics_dict(W)
    print("Graph alpha:", {k: round(v, 4) for k, v in g.items()})

    mats = [compute_plv_band(x[:, i : i + 256], sfreq, 8.0, 13.0) for i in range(0, 1024, 256)]
    deltas = transition_series(mats)
    print("Dynamics:", dynamic_summary(deltas))
    print("SMOKE OK")

if __name__ == "__main__":
    main()