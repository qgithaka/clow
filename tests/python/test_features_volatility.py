"""Unit tests for scale-free volatility regime features."""

import numpy as np
import pandas as pd

from training.data.features import FeatureEngineer


def test_volatility_features_and_regimes() -> None:
    """Verify Bollinger Band width, volatility expansion ratios, and squeeze detection."""
    dates = pd.date_range("2023-01-01", periods=100, freq="5min", tz="UTC")
    # Generate 50 bars of low volatility followed by 50 bars of explosive volatility
    low_vol = 1.0850 + np.random.RandomState(42).normal(0, 0.0002, 50)
    high_vol = 1.0850 + np.random.RandomState(42).normal(0, 0.0020, 50)
    prices = np.concatenate([low_vol, high_vol])

    df = pd.DataFrame({
        "timestamp_utc": dates,
        "open": prices,
        "high": prices + 0.0003,
        "low": prices - 0.0003,
        "close": prices,
        "volume": [100.0] * 100,
    })

    feats = FeatureEngineer.compute_volatility_features(df, bb_period=20, atr_short=7, atr_medium=14, atr_long=50)

    assert "atr_7" in feats.columns
    assert "atr_14" in feats.columns
    assert "atr_50" in feats.columns
    assert "atr_pct" in feats.columns
    assert "volatility_expansion_ratio" in feats.columns
    assert "range_expansion_ratio" in feats.columns
    assert "bb_width_pct" in feats.columns
    assert "bb_width_atr" in feats.columns
    assert "bb_pct_b" in feats.columns
    assert "is_volatility_squeeze" in feats.columns

    # During high volatility phase, short ATR should exceed long ATR -> expansion ratio > 1.0
    assert feats["volatility_expansion_ratio"].iloc[-1] > 1.0
    # BB width during high vol should be much larger than during low vol
    assert feats["bb_width_pct"].iloc[-1] > feats["bb_width_pct"].iloc[30]
    # No NaNs in any computed series
    assert not feats["bb_width_pct"].isna().any()
    assert not feats["is_volatility_squeeze"].isna().any()


def test_empty_dataframe_volatility() -> None:
    """Verify empty DataFrame safely returns empty DataFrame."""
    empty_res = FeatureEngineer.compute_volatility_features(pd.DataFrame())
    assert empty_res.empty
