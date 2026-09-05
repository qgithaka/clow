"""Unit tests for Multi-Timeframe Hierarchical Context Aligner."""

from datetime import timedelta
import numpy as np
import pandas as pd
from training.data.multi_timeframe import MultiTimeframeAligner
from training.data.features import FeatureEngineer


def test_resample_ohlcv() -> None:
    """Verify OHLCV resampling computes correct open, high, low, close, and volume aggregations."""
    dates = pd.date_range("2023-01-01 00:00:00", periods=12, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "open": [1.0800 + i * 0.0001 for i in range(12)],
        "high": [1.0820 + i * 0.0001 for i in range(12)],
        "low": [1.0790 + i * 0.0001 for i in range(12)],
        "close": [1.0805 + i * 0.0001 for i in range(12)],
        "volume": [10.0] * 12,
        "spread": [1.5] * 12,
    })

    # Resample 12 x 5min bars (1 hour) into 1h bar
    h1_df = MultiTimeframeAligner.resample_ohlcv(df, rule="1h")
    assert len(h1_df) == 1
    assert h1_df["open"].iloc[0] == df["open"].iloc[0]
    assert h1_df["high"].iloc[0] == df["high"].max()
    assert h1_df["low"].iloc[0] == df["low"].min()
    assert h1_df["close"].iloc[0] == df["close"].iloc[-1]
    assert h1_df["volume"].iloc[0] == 120.0
    assert np.isclose(h1_df["spread"].iloc[0], 1.5)


def test_multi_timeframe_causality_and_no_leakage() -> None:
    """Verify HTF features only become visible AFTER the HTF bar completely closes."""
    # 2 hours of 5-minute bars (24 bars: 08:00 to 09:55)
    ltf_dates = pd.date_range("2023-01-01 08:00:00", periods=24, freq="5min", tz="UTC")
    ltf_df = pd.DataFrame({
        "timestamp_utc": ltf_dates,
        "open": 1.0800,
        "high": 1.0820,
        "low": 1.0780,
        "close": 1.0810,
        "volume": 50.0,
    })

    # 2 H1 bars: Bar 0 is [08:00, 09:00), Bar 1 is [09:00, 10:00)
    htf_dates = pd.to_datetime(["2023-01-01 08:00:00", "2023-01-01 09:00:00"], utc=True)
    htf_df = pd.DataFrame({
        "timestamp_utc": htf_dates,
        "open": [1.0800, 1.0850],
        "high": [1.0830, 1.0890],
        "low": [1.0790, 1.0840],
        "close": [1.0825, 1.0880],
        "volume": [500.0, 600.0],
        "macro_trend_signal": [10.0, 20.0],  # Distinct signal
    })

    aligned = MultiTimeframeAligner.align_higher_timeframe(
        ltf_df=ltf_df,
        htf_df=htf_df,
        htf_name="H1",
        htf_bar_delta=timedelta(hours=1),
        feature_cols=["macro_trend_signal"],
    )

    assert "H1_macro_trend_signal" in aligned.columns

    # 1. During 08:00 to 08:55 (indices 0 to 11), H1 bar [08:00, 09:00) is NOT closed yet.
    # No completed H1 bars exist before 08:00, so aligned values MUST be NaN (zero future leak).
    for i in range(12):
        assert np.isnan(aligned["H1_macro_trend_signal"].iloc[i]), f"Leakage at index {i} ({aligned['timestamp_utc'].iloc[i]})"

    # 2. At 09:00:00 (index 12), the H1 bar [08:00, 09:00) with signal 10.0 has JUST closed.
    assert aligned["H1_macro_trend_signal"].iloc[12] == 10.0

    # 3. Throughout 09:05 to 09:55 (indices 13 to 23), the available signal remains 10.0
    # The H1 bar [09:00, 10:00) with signal 20.0 CANNOT be seen until 10:00:00.
    for i in range(12, 24):
        assert aligned["H1_macro_trend_signal"].iloc[i] == 10.0, f"Future leak at index {i}"


def test_multi_htf_dict_alignment() -> None:
    """Verify aligning multiple higher timeframes simultaneously."""
    dates = pd.date_range("2023-01-01 00:00:00", periods=50, freq="5min", tz="UTC")
    ltf_df = pd.DataFrame({
        "timestamp_utc": dates,
        "open": 1.0800,
        "high": 1.0810,
        "low": 1.0790,
        "close": 1.0805,
        "volume": 100.0,
    })

    h1_df = MultiTimeframeAligner.resample_ohlcv(ltf_df, rule="1h")
    h1_feats = FeatureEngineer.compute_candle_anatomy(h1_df)

    aligned = MultiTimeframeAligner.align_multi_timeframes(
        ltf_df=ltf_df,
        htf_dict={"H1": h1_feats},
    )

    assert "H1_body_ratio" in aligned.columns
    assert "H1_upper_wick_ratio" in aligned.columns


def test_empty_aligner_handling() -> None:
    """Verify empty inputs return empty outputs cleanly."""
    assert MultiTimeframeAligner.resample_ohlcv(pd.DataFrame(), "1h").empty
    assert MultiTimeframeAligner.align_higher_timeframe(pd.DataFrame(), pd.DataFrame(), "H1").empty
    assert MultiTimeframeAligner.align_multi_timeframes(pd.DataFrame(), {}).empty
