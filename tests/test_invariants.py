from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nai.connectivity.matrices import clean_connectivity_matrix, n_edges, threshold_by_density
from nai.features.aggregation import assert_one_row_per_subject, subject_level_mean
from nai.nai.composite import compute_nai

def test_clean_matrix_symmetric_zero_diag():
    W = np.array([[1.0, 0.3, 0.1], [0.9, 1.0, 0.2], [0.1, 0.4, 1.0]])
    C = clean_connectivity_matrix(W)
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C), 0.0)
    assert (C >= 0).all() and (C <= 1).all()

def test_threshold_density():
    rng = np.random.default_rng(0)
    W = rng.random((10, 10))
    W = 0.5 * (W + W.T)
    T = threshold_by_density(W, 0.2)
    assert np.allclose(np.diag(T), 0.0)
    assert n_edges(T) <= n_edges(clean_connectivity_matrix(W))

def test_subject_aggregation():
    df = pd.DataFrame(
        {
            "participant_id": ["a", "a", "b"],
            "age": [10.0, 10.0, 11.0],
            "group": ["TD", "TD", "TD"],
            "f1": [1.0, 3.0, 5.0],
            "f2": [2.0, 4.0, 6.0],
        }
    )
    out = subject_level_mean(df, ["f1", "f2"])
    assert_one_row_per_subject(out)
    row_a = out.loc[out["participant_id"] == "a"].iloc[0]
    assert abs(row_a["f1"] - 2.0) < 1e-12
    assert abs(row_a["f2"] - 3.0) < 1e-12
    assert int(row_a["n_runs"]) == 2

def test_nai_equal_weight():
    assert abs(compute_nai({"SE": 1.0, "C": 2.0, "G": 3.0, "D": 4.0}) - 2.5) < 1e-12

def test_plv_clean_properties():
    from nai.connectivity.phase import compute_plv

    rng = np.random.default_rng(0)
    data = rng.standard_normal((8, 500))
    W = compute_plv(data, clean=True)
    assert W.shape == (8, 8)
    assert np.allclose(W, W.T)
    assert np.allclose(np.diag(W), 0.0)
    assert (W >= 0).all() and (W <= 1).all()

def test_graph_metrics_finite():
    from nai.connectivity.phase import compute_plv
    from nai.graph.metrics import graph_metrics_dict

    rng = np.random.default_rng(1)
    data = rng.standard_normal((12, 800))
    W = compute_plv(data, clean=True)
    m = graph_metrics_dict(W)
    for k, v in m.items():
        assert np.isfinite(v), k

def test_band_powers_relative_sum():
    import mne
    from nai.spectral.power import compute_band_powers, relative_power_sum

    sfreq = 128.0
    n_ch, n_times = 8, int(sfreq * 4)
    data = np.random.default_rng(0).standard_normal((n_ch, n_times)) * 1e-6
    info = mne.create_info(
        ch_names=[f"E{i}" for i in range(n_ch)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    powers = compute_band_powers(raw)
    s = relative_power_sum(powers)
    assert abs(s - 1.0) < 1e-6
    assert "log_theta_alpha" in powers
    assert "log_theta_beta" in powers

def test_spectral_entropy_range():
    import mne
    from nai.spectral.entropy import compute_spectral_entropy

    sfreq = 128.0
    n_ch, n_times = 8, int(sfreq * 4)
    data = np.random.default_rng(1).standard_normal((n_ch, n_times)) * 1e-6
    info = mne.create_info(
        ch_names=[f"E{i}" for i in range(n_ch)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    out = compute_spectral_entropy(raw)
    assert 0.0 <= out["spectral_entropy_mean"] <= 1.0 + 1e-6
    assert out["spectral_entropy_std"] >= 0.0

def test_preprocess_minimal_picks_eeg():
    import mne
    import numpy as np
    from nai.preprocessing.pipeline import n_eeg_channels, preprocess_minimal

    sfreq = 128.0
    n_times = int(sfreq * 2)
    data = np.random.default_rng(0).standard_normal((5, n_times)) * 1e-6
    ch_names = [f"E{i}" for i in range(4)] + ["STI"]
    ch_types = ["eeg"] * 4 + ["stim"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)

    clean = preprocess_minimal(raw, set_montage=False)
    assert n_eeg_channels(clean) == 4
    assert clean.preload
    assert clean.info["sfreq"] == sfreq

def test_frobenius_delta_zero():
    from nai.dynamics.transitions import frobenius_delta

    W = np.eye(4)
    assert frobenius_delta(W, W) < 1e-12

def test_dynamic_summary_keys():
    from nai.dynamics.transitions import dynamic_summary, transition_series

    mats = [np.eye(5) * (i + 1) for i in range(4)]
    deltas = transition_series(mats)
    assert len(deltas) == 3
    s = dynamic_summary(deltas)
    assert s["mean_delta"] >= 0
    assert s["cv_delta"] >= 0
    assert s["n_transitions"] == 3

def test_ledoit_wolf_pd_symmetric():
    from nai.normative.robustness import (
        is_positive_definite,
        is_symmetric,
        ledoit_wolf_covariance,
    )

    rng = np.random.default_rng(0)
    R = rng.standard_normal((40, 6))
    Sigma, shrinkage = ledoit_wolf_covariance(R, assume_centered=True)
    assert Sigma.shape == (6, 6)
    assert is_symmetric(Sigma)
    assert is_positive_definite(Sigma)
    assert 0.0 <= shrinkage <= 1.0

def test_ledoit_wolf_rank_high_dim():
    from nai.normative.robustness import is_positive_definite, ledoit_wolf_covariance

    rng = np.random.default_rng(1)
    R = rng.standard_normal((39, 24)) 
    Sigma, shrinkage = ledoit_wolf_covariance(R)
    assert Sigma.shape == (24, 24)
    assert is_positive_definite(Sigma)
    assert np.isfinite(Sigma).all()
    assert 0.0 <= shrinkage <= 1.0

def test_ledoit_wolf_does_not_change_v10_api():
    from nai.normative.covariance import regularize_covariance
    from nai.normative.robustness import ledoit_wolf_covariance

    rng = np.random.default_rng(2)
    R = rng.standard_normal((30, 8))
    S_emp = np.cov(R, rowvar=False)
    S_lam = regularize_covariance(S_emp, lam=0.10)
    S_lw, _ = ledoit_wolf_covariance(R)
    assert S_lam.shape == S_lw.shape
    assert not np.allclose(S_lam, S_lw)

def test_quadratic_age_models_predict_shape():
    from nai.normative.regression import (
        fit_age_models_quadratic,
        predict_age,
        compute_residuals,
    )

    rng = np.random.default_rng(10)
    age = np.linspace(8.0, 13.0, 39)
    X = rng.standard_normal((39, 6))

    models = fit_age_models_quadratic(X, age)
    X_hat = predict_age(models, age)
    R = compute_residuals(X, age, models)

    assert X_hat.shape == X.shape
    assert R.shape == X.shape
    assert np.isfinite(X_hat).all()
    assert np.isfinite(R).all()

def test_quadratic_age_models_are_nonlinear():
    from nai.normative.regression import fit_age_models_quadratic

    age = np.linspace(8.0, 13.0, 39)
    X = np.column_stack(
        [
            0.5 * age**2 - 2.0 * age + 3.0,
            np.sin(age),
        ]
    )

    models = fit_age_models_quadratic(X, age)
    for model in models:
        poly = model.named_steps["polynomialfeatures"]
        assert poly.n_output_features_ == 2

def test_score_nai_from_dataframe_smoke():
    import pandas as pd
    from nai.features.blocks import SE_FEATURES, C_FEATURES, G_FEATURES, D_FEATURES
    from nai.features.extractor import score_nai_from_dataframe

    n = 12
    rng = np.random.default_rng(0)
    data = {"participant_id": [f"s{i}" for i in range(n)], "age": rng.uniform(8, 13, n), "group": ["TD"] * 10 + ["ASD"] * 2}
    for cols in (SE_FEATURES, C_FEATURES, G_FEATURES, D_FEATURES):
        for c in cols:
            data[c] = rng.standard_normal(n)
    df = pd.DataFrame(data)
    out = score_nai_from_dataframe(df)
    assert "NAI" in out.columns
    assert len(out) == n
    assert np.isfinite(out["NAI"]).all()