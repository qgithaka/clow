"""Unit tests for TacticalForecasterPipeline end-to-end integration."""

import numpy as np
import pandas as pd
import pytest
import torch
from training.models.dataset import TimeSeriesSlidingWindowDataset
from training.models.forecaster import ClowForecaster
from training.models.inference import ForecasterPredictor
from training.models.order_policy import TacticalOrderPolicy, OrderType
from training.models.pipeline import TacticalForecasterPipeline


def generate_test_df(n_bars: int = 100) -> pd.DataFrame:
    """Generates synthetic OHLCV bars."""
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="5min", tz="UTC")
    prices = 1.0850 + np.sin(np.linspace(0, 10, n_bars)) * 0.0050
    return pd.DataFrame({
        "timestamp_utc": dates,
        "open": prices,
        "high": prices + 0.0005,
        "low": prices - 0.0005,
        "close": prices + 0.0002,
        "volume": [100.0] * n_bars,
        "spread": [0.00015] * n_bars,
    })


def test_end_to_end_pipeline() -> None:
    """Verify end-to-end pipeline execution from raw bars to tactical order proposal."""
    df = generate_test_df(n_bars=80)
    context_len = 32

    # Instantiate Model 1
    num_feats = len(TimeSeriesSlidingWindowDataset.DEFAULT_FEATURE_COLS)
    model = ClowForecaster(
        input_dim=num_feats,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
    )
    predictor = ForecasterPredictor(model=model, device="cpu")
    policy = TacticalOrderPolicy(min_directional_confidence=0.50)

    pipeline = TacticalForecasterPipeline(
        predictor=predictor,
        policy=policy,
        context_length=context_len,
    )

    proposal = pipeline.process_bars(df, symbol="EURUSD")

    assert proposal.symbol == "EURUSD"
    assert proposal.current_price > 0.0
    assert proposal.entry_price > 0.0
    assert proposal.order_type in [OrderType.BUY_LIMIT, OrderType.SELL_LIMIT, OrderType.HOLD_NO_ACTION]


def test_pipeline_insufficient_bars() -> None:
    """Verify pipeline raises ValueError on insufficient bars."""
    model = ClowForecaster(input_dim=16, d_model=16, nhead=2, num_layers=1)
    predictor = ForecasterPredictor(model=model)
    pipeline = TacticalForecasterPipeline(predictor=predictor, context_length=64)

    short_df = generate_test_df(n_bars=20)
    with pytest.raises(ValueError, match="Insufficient bars"):
        pipeline.process_bars(short_df)
