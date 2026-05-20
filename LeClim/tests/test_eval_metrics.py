"""Golden tests for cos-weighted metrics (no torch)."""

from __future__ import annotations

import numpy as np

from climatebench_lewm.evaluate_climatebench import _nrmse_decomposition


def test_nrmse_decomposition_perfect():
    lat = np.array([-60.0, 0.0, 60.0])
    truth = [
        np.ones((3, 4), dtype=np.float32) * 280.0,
        np.ones((3, 4), dtype=np.float32) * 281.0,
    ]
    pred = [np.array(x, copy=True) for x in truth]
    out = _nrmse_decomposition(pred, truth, lat)
    assert out["rmse_total_k"] < 1e-6
    assert out["rmse_global_mean_k"] < 1e-6
    assert out["rmse_spatial_pattern_k"] < 1e-6
    assert out["nrmse_total"] < 1e-6


def test_nrmse_spatial_constant_bias():
    """Uniform +1 K bias: spatial pattern RMSE ~0, global RMSE > 0."""
    lat = np.array([-45.0, 0.0, 45.0])
    truth = [np.full((3, 2), 275.0, dtype=np.float32) for _ in range(5)]
    pred = [t + 1.0 for t in truth]
    out = _nrmse_decomposition(pred, truth, lat)
    assert out["rmse_spatial_pattern_k"] < 1e-5
    assert out["rmse_global_mean_k"] > 0.9
    assert abs(out["rmse_global_mean_k"] - 1.0) < 0.05
