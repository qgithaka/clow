"""Unit tests for strictly causal rolling scalers."""

import numpy as np
import pandas as pd

from training.data.scalers import RollingMinMaxScaler, RollingRobustScaler, RollingZScoreScaler


def test_rolling_zscore_batch_and_streaming_equivalence() -> None:
    """Verify batch transform matches streaming step-by-step updates exactly."""
    np.random.seed(42)
    raw_vals = np.random.normal(loc=1.0850, scale=0.0020, size=150)
    series = pd.Series(raw_vals)

    scaler = RollingZScoreScaler(window=30, min_periods=5, clip_val=4.0)
    batch_transformed = scaler.transform_series(series)

    # Streaming evaluation
    streaming_scaler = RollingZScoreScaler(window=30, min_periods=5, clip_val=4.0)
    stream_results = []
    for val in raw_vals:
        s_val = streaming_scaler.step("feat", float(val))
        stream_results.append(s_val)

    stream_series = pd.Series(stream_results)

    # After warmup window (>= 5 bars), values should match within floating point precision
    valid_mask = ~batch_transformed.iloc[10:].isna()
    diff = np.abs(batch_transformed.iloc[10:][valid_mask] - stream_series.iloc[10:][valid_mask])
    assert (diff < 1e-4).all()


def test_scaler_strict_causality() -> None:
    """Verify modifying future data does NOT alter past transformed values."""
    series_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    series_b = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

    scaler = RollingZScoreScaler(window=5, min_periods=2)
    res_a = scaler.transform_series(series_a)
    res_b = scaler.transform_series(series_b)

    # Indices 0 through 4 MUST be completely identical because index 5 (which differs) is in the future
    np.testing.assert_allclose(res_a.iloc[:5].values, res_b.iloc[:5].values, rtol=1e-5, atol=1e-5)


def test_rolling_minmax_and_robust_scalers() -> None:
    """Verify RobustScaler and MinMaxScaler properties and bounds."""
    vals = pd.Series(np.linspace(10.0, 20.0, 50))
    minmax = RollingMinMaxScaler(window=20, min_periods=5, feature_range=(-1.0, 1.0))
    mm_res = minmax.transform_series(vals)

    # After warmup, values must be in [-1.0, 1.0]
    valid_mm = mm_res.dropna()
    assert (valid_mm >= -1.0 - 1e-6).all()
    assert (valid_mm <= 1.0 + 1e-6).all()

    robust = RollingRobustScaler(window=20, min_periods=5, clip_val=3.0)
    rob_res = robust.transform_series(vals)
    valid_rob = rob_res.dropna()
    assert (valid_rob >= -3.0).all() and (valid_rob <= 3.0).all()


def test_empty_scaler_inputs() -> None:
    """Verify empty inputs return empty series safely."""
    assert RollingZScoreScaler().transform_series(pd.Series(dtype=float)).empty
    assert RollingRobustScaler().transform_series(pd.Series(dtype=float)).empty
    assert RollingMinMaxScaler().transform_series(pd.Series(dtype=float)).empty


def test_transform_df_and_reset() -> None:
    """Verify transform_df and reset across all scalers."""
    df = pd.DataFrame({
        "feat1": np.linspace(1.0, 10.0, 20),
        "feat2": np.linspace(100.0, 200.0, 20),
    })

    z_scaler = RollingZScoreScaler(window=10, min_periods=2)
    z_df = z_scaler.transform_df(df, columns=["feat1", "feat2"])
    assert "feat1_zscore" in z_df.columns
    assert "feat2_zscore" in z_df.columns
    z_scaler.reset()

    rob_scaler = RollingRobustScaler(window=10, min_periods=2)
    rob_df = rob_scaler.transform_df(df, columns=["feat1", "feat2"])
    assert "feat1_robust" in rob_df.columns
    assert "feat2_robust" in rob_df.columns
    rob_scaler.reset()

    minmax_scaler = RollingMinMaxScaler(window=10, min_periods=2)
    mm_df = minmax_scaler.transform_df(df, columns=["feat1", "feat2"])
    assert "feat1_minmax" in mm_df.columns
    assert "feat2_minmax" in mm_df.columns
    minmax_scaler.reset()
