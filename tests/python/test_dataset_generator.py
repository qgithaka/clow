"""Unit tests for time-series sliding window dataset generator."""

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from training.data.features import FeatureEngineer
from training.models.dataset import TimeSeriesSlidingWindowDataset


def generate_sample_df(n_bars: int = 200) -> pd.DataFrame:
    """Generates clean synthetic feature DataFrame for dataset generator tests."""
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="5min", tz="UTC")
    prices = 1.0850 + np.sin(np.linspace(0, 10, n_bars)) * 0.0050
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "open": prices,
        "high": prices + 0.0005,
        "low": prices - 0.0005,
        "close": prices + 0.0002,
        "volume": [100.0] * n_bars,
    })
    return FeatureEngineer.compute_all_features(df)


def test_sliding_window_shapes_and_dataloader() -> None:
    """Verify sliding window tensor shapes and PyTorch DataLoader batching."""
    df = generate_sample_df(n_bars=200)
    context_len = 64
    pred_horizon = 1

    ds = TimeSeriesSlidingWindowDataset(
        df=df,
        context_length=context_len,
        prediction_horizon=pred_horizon,
    )

    expected_len = 200 - context_len - pred_horizon + 1
    assert len(ds) == expected_len

    # Single item checks
    sample = ds[0]
    assert sample["context"].shape == (context_len, len(ds.feature_cols))
    assert sample["target_candle"].shape == (pred_horizon, len(ds.target_cols))
    assert sample["target_direction"].shape == (pred_horizon,)
    assert sample["target_quantiles"].shape == (pred_horizon, 2)
    assert sample["target_return_atr"].shape == (pred_horizon,)

    # PyTorch DataLoader batching
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    batch = next(iter(loader))
    assert batch["context"].shape == (16, context_len, len(ds.feature_cols))
    assert batch["target_candle"].shape == (16, pred_horizon, len(ds.target_cols))
    assert batch["target_direction"].shape == (16, pred_horizon)
    assert batch["target_quantiles"].shape == (16, pred_horizon, 2)


def test_temporal_train_val_test_split() -> None:
    """Verify chronological split respects purging boundaries without leakage."""
    df = generate_sample_df(n_bars=300)
    purge_len = 20

    train_df, val_df, test_df = TimeSeriesSlidingWindowDataset.temporal_train_val_test_split(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_window=purge_len
    )

    assert not train_df.empty
    assert not val_df.empty
    assert not test_df.empty

    # Verify chronological order
    assert train_df["timestamp_utc"].max() < val_df["timestamp_utc"].min()
    assert val_df["timestamp_utc"].max() < test_df["timestamp_utc"].min()

    # Verify purging gap
    train_end_idx = train_df.index[-1]
    val_start_idx = val_df.index[0]
    val_end_idx = val_df.index[-1]
    test_start_idx = test_df.index[0]

    assert val_start_idx - train_end_idx >= purge_len
    assert test_start_idx - val_end_idx >= purge_len


def test_error_handling_and_bounds() -> None:
    """Verify validation and error raising on invalid inputs."""
    with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
        TimeSeriesSlidingWindowDataset(pd.DataFrame())

    tiny_df = generate_sample_df(n_bars=10)
    with pytest.raises(ValueError, match="is insufficient for context_length"):
        TimeSeriesSlidingWindowDataset(tiny_df, context_length=64)

    valid_ds = TimeSeriesSlidingWindowDataset(generate_sample_df(n_bars=100), context_length=32)
    with pytest.raises(IndexError):
        _ = valid_ds[9999]
    with pytest.raises(IndexError):
        _ = valid_ds[-1]
