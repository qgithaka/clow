"""Comprehensive synthetic dirty/clean dataset stress test suite."""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest
from training.data.chunked_extractor import CanonicalTimeframe
from training.data.cleaner import DataCleaner
from training.data.health_report import HealthReportGenerator
from training.data.storage import ParquetStorageManager
from training.data.validator import DataValidator


def generate_synthetic_dirty_dataset() -> pd.DataFrame:
    """Generates a dataset with 8 distinct classes of deliberate corruption."""
    base_time = datetime(2023, 5, 1, 0, 0, tzinfo=timezone.utc)
    rows = []

    for i in range(100):
        t = base_time + timedelta(minutes=i * 5)
        # 1. Normal row
        o, h, l, c = 1.0850, 1.0860, 1.0840, 1.0855
        spread = 1.2
        vol = 100.0

        # Inject Corruptions:
        if i == 10:
            # Corruption 1: Negative price
            o, h, l, c = -1.0850, 1.0860, 1.0840, 1.0855
        elif i == 20:
            # Corruption 2: High < Low
            o, h, l, c = 1.0850, 1.0830, 1.0870, 1.0855
        elif i == 30:
            # Corruption 3: Negative Spread
            spread = -2.5
        elif i == 40:
            # Corruption 4: NaN Close
            c = np.nan
        elif i == 50:
            # Corruption 5: Duplicate timestamp
            t = base_time + timedelta(minutes=49 * 5)
        elif i == 60:
            # Corruption 6: Disordered timestamp (Step back 1 hour)
            t = base_time + timedelta(minutes=10 * 5)
        elif i == 70:
            # Corruption 7: Extreme Spread (> 100 pips)
            spread = 125.0

        rows.append({
            "timestamp_utc": t,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": vol,
            "tick_volume": vol * 2,
            "spread": spread,
        })

    return pd.DataFrame(rows)


def test_validator_catches_all_corruptions() -> None:
    """Verify DataValidator flags all injected problem classes."""
    dirty_df = generate_synthetic_dirty_dataset()
    val = DataValidator.validate(dirty_df, timeframe=CanonicalTimeframe.M5)

    assert val.is_valid is False
    assert val.error_count >= 5
    categories = {iss["category"] for iss in val.issues}
    assert "NON_POSITIVE_PRICE" in categories
    assert "INVALID_GEOMETRY" in categories
    assert "NEGATIVE_SPREAD" in categories
    assert "NAN_VALUES" in categories
    assert "DUPLICATE_TIMESTAMPS" in categories
    assert "DISORDERED_TIMESTAMPS" in categories
    assert "HIGH_SPREAD" in categories


def test_cleaner_recovers_pristine_subset() -> None:
    """Verify DataCleaner purges all invalid rows and produces 100% valid dataset."""
    dirty_df = generate_synthetic_dirty_dataset()
    cleaned_df, stats = DataCleaner.clean_dataset(dirty_df, timeframe=CanonicalTimeframe.M5)

    # Validate cleaned dataset
    val = DataValidator.validate(cleaned_df, timeframe=CanonicalTimeframe.M5)
    assert val.is_valid is True
    assert val.error_count == 0
    assert stats["duplicates_removed"] >= 1
    assert stats["invalid_geometry_rows"] >= 3
    assert len(cleaned_df) < len(dirty_df)


def test_end_to_end_data_engine_pipeline(tmp_path) -> None:
    """Verify complete extract -> clean -> validate -> store -> health report pipeline."""
    dirty_df = generate_synthetic_dirty_dataset()
    cleaned_df, _ = DataCleaner.clean_dataset(dirty_df, timeframe=CanonicalTimeframe.M5)

    storage = ParquetStorageManager(base_dir=tmp_path)
    meta = storage.save_dataset(cleaned_df, symbol="EURUSD", timeframe="M5")

    # Verify reload
    reloaded = storage.load_dataset(meta.file_path)
    assert len(reloaded) == len(cleaned_df)

    # Generate health report
    report = HealthReportGenerator.generate_report(reloaded, symbol="EURUSD", timeframe=CanonicalTimeframe.M5)
    assert report.status in ("PASSED", "WARNING")
    assert report.total_rows == len(cleaned_df)
    assert report.validation_summary.is_valid is True
