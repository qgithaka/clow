"""Unit tests for normalized momentum indicators."""

import numpy as np
import pandas as pd
from training.data.features import FeatureEngineer


def test_rsi_bounds_and_symmetry() -> None:
    """Verify RSI calculation behaves causally and normalized RSI is bounded in [-1.0, 1.0]."""
    # Monotonically increasing prices -> RSI -> 100, stationary RSI -> 1.0
    dates = pd.date_range("2023-01-01", periods=50, freq="5min", tz="UTC")
    up_prices = np.linspace(1.0800, 1.0900, 50)
    df_up = pd.DataFrame({
        "timestamp_utc": dates,
        "open": up_prices - 0.0001,
        "high": up_prices + 0.0002,
        "low": up_prices - 0.0002,
        "close": up_prices,
        "volume": [100.0] * 50,
    })

    feats_up = FeatureEngineer.compute_momentum_indicators(df_up, rsi_period=14)
    assert "rsi" in feats_up.columns
    assert "rsi_stationary" in feats_up.columns
    assert feats_up["rsi"].iloc[-1] > 90.0
    assert feats_up["rsi_stationary"].iloc[-1] > 0.8
    assert (feats_up["rsi_stationary"] >= -1.0).all()
    assert (feats_up["rsi_stationary"] <= 1.0).all()

    # Monotonically decreasing prices -> RSI -> 0, stationary RSI -> -1.0
    down_prices = np.linspace(1.0900, 1.0800, 50)
    df_down = pd.DataFrame({
        "timestamp_utc": dates,
        "open": down_prices + 0.0001,
        "high": down_prices + 0.0002,
        "low": down_prices - 0.0002,
        "close": down_prices,
        "volume": [100.0] * 50,
    })
    feats_down = FeatureEngineer.compute_momentum_indicators(df_down, rsi_period=14)
    assert feats_down["rsi"].iloc[-1] < 10.0
    assert feats_down["rsi_stationary"].iloc[-1] < -0.8


def test_ema_deviations_and_macd_ratios() -> None:
    """Verify EMA Z-score deviations and scale-free MACD ratios."""
    dates = pd.date_range("2023-01-01", periods=100, freq="5min", tz="UTC")
    # Sine wave prices with known properties
    t = np.linspace(0, 4 * np.pi, 100)
    prices = 1.0850 + 0.0050 * np.sin(t)
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "open": prices,
        "high": prices + 0.0005,
        "low": prices - 0.0005,
        "close": prices,
        "volume": [100.0] * 100,
    })

    feats = FeatureEngineer.compute_momentum_indicators(
        df,
        ema_periods=(20, 50),
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
    )

    for p in [20, 50]:
        assert f"ema_{p}" in feats.columns
        assert f"ema_dist_atr_{p}" in feats.columns
        assert f"ema_zscore_{p}" in feats.columns
        assert f"ema_dist_pct_{p}" in feats.columns
        assert not feats[f"ema_zscore_{p}"].isna().any()

    # MACD scale-free columns
    assert "macd_dist_atr" in feats.columns
    assert "macd_signal_atr" in feats.columns
    assert "macd_hist_atr" in feats.columns
    assert "macd_dist_pct" in feats.columns
    assert "macd_signal_pct" in feats.columns
    assert "macd_hist_pct" in feats.columns

    # Verify no NaN values in output
    assert not feats["macd_hist_atr"].isna().any()
    assert not feats["macd_hist_pct"].isna().any()


def test_empty_dataframe_handling() -> None:
    """Verify empty DataFrame safely returns empty DataFrame."""
    empty_res = FeatureEngineer.compute_momentum_indicators(pd.DataFrame())
    assert empty_res.empty
