"""Unit tests for historical data cleaning and weekend handling."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
from training.data.cleaner import DataCleaner


def test_forex_weekend_detection() -> None:
    """Verify standard Forex weekend closure intervals."""
    # Friday 21:00 UTC -> Open
    fri_open = datetime(2023, 10, 6, 21, 0, tzinfo=timezone.utc)
    assert not DataCleaner.is_forex_weekend(fri_open)

    # Friday 23:00 UTC -> Closed
    fri_closed = datetime(2023, 10, 6, 23, 0, tzinfo=timezone.utc)
    assert DataCleaner.is_forex_weekend(fri_closed)

    # Saturday 12:00 UTC -> Closed
    sat_closed = datetime(2023, 10, 7, 12, 0, tzinfo=timezone.utc)
    assert DataCleaner.is_forex_weekend(sat_closed)

    # Sunday 20:00 UTC -> Closed
    sun_closed = datetime(2023, 10, 8, 20, 0, tzinfo=timezone.utc)
    assert DataCleaner.is_forex_weekend(sun_closed)

    # Sunday 22:00 UTC -> Open
    sun_open = datetime(2023, 10, 8, 22, 0, tzinfo=timezone.utc)
    assert not DataCleaner.is_forex_weekend(sun_open)


def test_clean_dataset_deduplication_and_sorting() -> None:
    """Verify duplicate removal and chronological reordering."""
    dirty_df = pd.DataFrame({
        "timestamp_utc": [
            "2023-01-01 10:15:00+00:00",
            "2023-01-01 10:05:00+00:00",  # Disordered
            "2023-01-01 10:05:00+00:00",  # Duplicate
            "2023-01-01 10:10:00+00:00",
        ],
        "open": [1.0850, 1.0840, 1.0840, 1.0845],
        "high": [1.0860, 1.0850, 1.0850, 1.0855],
        "low": [1.0845, 1.0835, 1.0835, 1.0840],
        "close": [1.0855, 1.0848, 1.0848, 1.0850],
        "volume": [10.0, 20.0, 20.0, 15.0],
        "tick_volume": [20.0, 40.0, 40.0, 30.0],
        "spread": [1.0, 1.0, 1.0, 1.0],
    })

    cleaned, stats = DataCleaner.clean_dataset(dirty_df)
    assert stats["duplicates_removed"] == 1
    assert stats["disordered_rows"] == 1
    assert stats["final_rows"] == 3
    assert cleaned["timestamp_utc"].is_monotonic_increasing


def test_clean_dataset_geometry_filtering() -> None:
    """Verify invalid high/low and negative prices are rejected."""
    corrupt_df = pd.DataFrame({
        "timestamp_utc": [
            "2023-01-01 10:00:00+00:00",
            "2023-01-01 10:05:00+00:00",  # High < Low (Corrupt)
            "2023-01-01 10:10:00+00:00",  # Negative Price (Corrupt)
            "2023-01-01 10:15:00+00:00",  # NaN Close (Corrupt)
            "2023-01-01 10:20:00+00:00",  # Valid
        ],
        "open": [1.0850, 1.0850, -1.0850, 1.0850, 1.0850],
        "high": [1.0860, 1.0830, 1.0860, 1.0860, 1.0860],
        "low": [1.0840, 1.0860, 1.0840, 1.0840, 1.0840],
        "close": [1.0855, 1.0840, 1.0850, np.nan, 1.0855],
        "volume": [10.0, 10.0, 10.0, 10.0, 10.0],
        "tick_volume": [20.0, 20.0, 20.0, 20.0, 20.0],
        "spread": [1.0, 1.0, 1.0, 1.0, 1.0],
    })

    cleaned, stats = DataCleaner.clean_dataset(corrupt_df)
    assert stats["invalid_geometry_rows"] == 3
    assert stats["final_rows"] == 2
