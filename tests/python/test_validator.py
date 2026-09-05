"""Unit tests for strict data health validation."""

import pandas as pd

from training.data.chunked_extractor import CanonicalTimeframe
from training.data.validator import DataValidator


def test_validator_clean_dataset_passes() -> None:
    """Verify clean OHLCV dataset passes validation with 0 errors."""
    dates = pd.date_range("2023-01-02 00:00:00", periods=50, freq="5min", tz="UTC")
    clean_df = pd.DataFrame({
        "timestamp_utc": dates,
        "open": [1.0850] * 50,
        "high": [1.0860] * 50,
        "low": [1.0840] * 50,
        "close": [1.0855] * 50,
        "volume": [10.0] * 50,
        "spread": [1.2] * 50,
    })

    res = DataValidator.validate(clean_df, timeframe=CanonicalTimeframe.M5)
    assert res.is_valid is True
    assert res.error_count == 0
    assert res.total_rows == 50


def test_validator_detects_bad_geometry_and_disorder() -> None:
    """Verify detection of negative prices, high < low, and disordered timestamps."""
    bad_df = pd.DataFrame({
        "timestamp_utc": [
            "2023-01-02 10:10:00+00:00",
            "2023-01-02 10:05:00+00:00",  # Disordered
            "2023-01-02 10:15:00+00:00",
        ],
        "open": [1.0850, 1.0850, -1.0850],  # Negative price
        "high": [1.0840, 1.0860, 1.0860],   # High < Open
        "low": [1.0830, 1.0840, 1.0840],
        "close": [1.0835, 1.0850, 1.0850],
        "spread": [1.0, -2.0, 1.0],          # Negative spread
    })

    res = DataValidator.validate(bad_df, timeframe=CanonicalTimeframe.M5)
    assert res.is_valid is False
    assert res.error_count >= 3
    categories = [iss["category"] for iss in res.issues]
    assert "DISORDERED_TIMESTAMPS" in categories
    assert "INVALID_GEOMETRY" in categories
    assert "NON_POSITIVE_PRICE" in categories
    assert "NEGATIVE_SPREAD" in categories


def test_validator_empty_dataset() -> None:
    """Verify empty dataframe is rejected."""
    empty_df = pd.DataFrame()
    res = DataValidator.validate(empty_df)
    assert res.is_valid is False
    assert res.error_count == 1
