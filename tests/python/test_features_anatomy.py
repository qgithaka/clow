"""Unit tests for scale-free candle anatomy features."""

import numpy as np
import pandas as pd

from training.data.features import FeatureEngineer


def test_candle_anatomy_math() -> None:
    """Verify exact mathematical bounds of body and wick ratios."""
    # Construct a clean hammer pinbar: Open=1.0850, High=1.0860, Low=1.0800, Close=1.0855
    # Range = 60 pips (1.0800 to 1.0860)
    # Body = +5 pips (1.0850 to 1.0855) -> body_ratio = 5/60 ~ 0.0833
    # Upper Wick = 5 pips (1.0860 - 1.0855) -> upper_wick = 5/60 ~ 0.0833
    # Lower Wick = 50 pips (1.0850 - 1.0800) -> lower_wick = 50/60 ~ 0.8333
    df = pd.DataFrame({
        "timestamp_utc": pd.date_range("2023-01-01", periods=1, freq="5min", tz="UTC"),
        "open": [1.0850],
        "high": [1.0860],
        "low": [1.0800],
        "close": [1.0855],
        "volume": [100.0],
    })

    feats = FeatureEngineer.compute_candle_anatomy(df)

    assert np.isclose(feats["body_ratio"].iloc[0], 5.0 / 60.0, atol=1e-3)
    assert np.isclose(feats["upper_wick_ratio"].iloc[0], 5.0 / 60.0, atol=1e-3)
    assert np.isclose(feats["lower_wick_ratio"].iloc[0], 50.0 / 60.0, atol=1e-3)
    # Proportions sum to 1.0
    total_proportion = feats["abs_body_ratio"].iloc[0] + feats["upper_wick_ratio"].iloc[0] + feats["lower_wick_ratio"].iloc[0]
    assert np.isclose(total_proportion, 1.0, atol=1e-3)


def test_candle_anatomy_empty_and_ranges() -> None:
    """Verify empty handling and range to ATR calculation."""
    dates = pd.date_range("2023-01-01", periods=30, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "open": [1.0850] * 30,
        "high": [1.0870] * 30,
        "low": [1.0830] * 30,
        "close": [1.0860] * 30,
        "volume": [50.0] * 30,
    })

    feats = FeatureEngineer.compute_candle_anatomy(df, atr_period=14)
    assert "range_to_atr" in feats.columns
    assert "body_to_atr" in feats.columns
    assert feats["range_to_atr"].iloc[-1] > 0.0

    empty_feats = FeatureEngineer.compute_candle_anatomy(pd.DataFrame())
    assert empty_feats.empty
