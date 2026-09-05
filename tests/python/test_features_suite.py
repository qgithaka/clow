"""Comprehensive verification suite for stationarity, scale-invariance, and causality."""

import numpy as np
import pandas as pd

from training.data.features import FeatureEngineer
from training.data.multi_timeframe import MultiTimeframeAligner


def generate_synthetic_series(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic Forex OHLCV bars with random walk price drift."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="5min", tz="UTC")

    # Drift + Random Walk
    returns = rng.normal(loc=0.00001, scale=0.0008, size=n_bars)
    close_prices = 1.0800 * np.exp(np.cumsum(returns))

    spread_bars = rng.uniform(0.0001, 0.0005, size=n_bars)
    high_prices = close_prices + spread_bars * rng.uniform(0.5, 1.5, size=n_bars)
    low_prices = close_prices - spread_bars * rng.uniform(0.5, 1.5, size=n_bars)
    open_prices = low_prices + (high_prices - low_prices) * rng.uniform(0.2, 0.8, size=n_bars)
    volumes = rng.uniform(10.0, 500.0, size=n_bars)

    return pd.DataFrame({
        "timestamp_utc": dates,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    })


def test_strict_scale_invariance() -> None:
    """Verify scale-free features are strictly invariant when raw prices are scaled 100x (EURUSD vs USDJPY)."""
    df_base = generate_synthetic_series(n_bars=200, seed=101)
    df_scaled = df_base.copy()

    # Scale raw prices by 100x
    for col in ["open", "high", "low", "close"]:
        df_scaled[col] = df_base[col] * 100.0

    feats_base = FeatureEngineer.compute_all_features(df_base)
    feats_scaled = FeatureEngineer.compute_all_features(df_scaled)

    scale_invariant_cols = [
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "range_pct",
        "range_to_atr",
        "body_to_atr",
        "rsi_stationary",
        "macd_dist_pct",
        "bb_pct_b",
        "volatility_expansion_ratio",
    ]

    for col in scale_invariant_cols:
        assert col in feats_base.columns
        assert col in feats_scaled.columns
        np.testing.assert_allclose(
            feats_base[col].values[20:],
            feats_scaled[col].values[20:],
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"Feature {col} is not scale-invariant!",
        )


def test_strict_causality_pipeline_invariance() -> None:
    """Verify perturbing bar t+1..T has ZERO effect on features at bars 0..t."""
    df_clean = generate_synthetic_series(n_bars=200, seed=202)
    split_idx = 100

    # Pipeline run on full series
    full_feats = FeatureEngineer.compute_all_features(df_clean)

    # Pipeline run only up to split_idx
    truncated_df = df_clean.iloc[:split_idx].copy()
    truncated_feats = FeatureEngineer.compute_all_features(truncated_df)

    # Corrupted future series (insane price shock at split_idx onwards)
    corrupted_df = df_clean.copy()
    corrupted_df.loc[split_idx:, ["open", "high", "low", "close"]] *= 1000.0
    corrupted_feats = FeatureEngineer.compute_all_features(corrupted_df)

    # Verify features at 0..split_idx-1 in full, truncated, and corrupted are 100% IDENTICAL
    check_cols = [
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
        "range_to_atr", "rsi_stationary", "macd_dist_atr",
        "bb_width_pct", "volatility_expansion_ratio", "session_london",
    ]

    for col in check_cols:
        np.testing.assert_allclose(
            full_feats[col].iloc[:split_idx].values,
            truncated_feats[col].values,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Causality leak detected in {col} between full and truncated series!",
        )
        np.testing.assert_allclose(
            full_feats[col].iloc[:split_idx].values,
            corrupted_feats[col].iloc[:split_idx].values,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Future corruption leaked into past in {col}!",
        )


def test_end_to_end_pipeline_with_htf_alignment() -> None:
    """Verify compute_all_features with multi-timeframe HTF alignment."""
    df_m5 = generate_synthetic_series(n_bars=300, seed=303)
    df_h1 = MultiTimeframeAligner.resample_ohlcv(df_m5, rule="1h")
    df_h1_feats = FeatureEngineer.compute_all_features(df_h1)

    all_feats = FeatureEngineer.compute_all_features(
        df_m5,
        htf_dict={"H1": df_h1_feats},
    )

    assert "H1_body_ratio" in all_feats.columns
    assert "H1_rsi_stationary" in all_feats.columns
    assert "H1_volatility_expansion_ratio" in all_feats.columns
    assert "session_london" in all_feats.columns
    assert not all_feats.empty
