"""Unit tests for institutional Forex session timing features."""

import numpy as np
import pandas as pd
from training.data.features import FeatureEngineer


def test_session_features_utc_masks() -> None:
    """Verify session masks activate correctly at precise UTC hours."""
    # Test specific known hours
    # 2023-01-02 is a Monday (dow = 0)
    # Asian: 03:00 UTC (hour 3)
    # London Open: 08:30 UTC (hour 8)
    # London/NY Overlap: 14:00 UTC (hour 14)
    # London Close: 16:00 UTC (hour 16)
    # Friday close risk: 2023-01-06 (Friday, dow = 4) 21:00 UTC
    timestamps = pd.to_datetime([
        "2023-01-02 03:00:00",
        "2023-01-02 08:30:00",
        "2023-01-02 14:00:00",
        "2023-01-02 16:00:00",
        "2023-01-06 21:00:00",
    ], utc=True)

    df = pd.DataFrame({
        "timestamp_utc": timestamps,
        "open": [1.0800] * 5,
        "high": [1.0850] * 5,
        "low": [1.0750] * 5,
        "close": [1.0820] * 5,
        "volume": [100.0] * 5,
    })

    feats = FeatureEngineer.compute_session_features(df)

    # 1. Row 0 (03:00 UTC Asian)
    assert feats["session_asian"].iloc[0] == 1.0
    assert feats["session_london"].iloc[0] == 0.0
    assert feats["session_ny"].iloc[0] == 0.0

    # 2. Row 1 (08:30 UTC London Open)
    assert feats["session_london"].iloc[1] == 1.0
    assert feats["session_london_open"].iloc[1] == 1.0
    assert feats["session_ny"].iloc[1] == 0.0

    # 3. Row 2 (14:00 UTC NY / London Overlap)
    assert feats["session_london"].iloc[2] == 1.0
    assert feats["session_ny"].iloc[2] == 1.0
    assert feats["session_ny_london_overlap"].iloc[2] == 1.0

    # 4. Row 3 (16:00 UTC London Close window)
    assert feats["session_london_close"].iloc[3] == 1.0

    # 5. Row 4 (Friday 21:00 UTC Weekend Risk)
    assert feats["is_weekend_close_risk"].iloc[4] == 1.0

    # Continuous Cyclical Embeddings
    assert "sin_time_of_day" in feats.columns
    assert "cos_time_of_day" in feats.columns
    assert "sin_day_of_week" in feats.columns
    assert "cos_day_of_week" in feats.columns
    assert (feats["sin_time_of_day"] >= -1.0).all() and (feats["sin_time_of_day"] <= 1.0).all()


def test_empty_dataframe_sessions() -> None:
    """Verify empty DataFrame safely returns empty DataFrame."""
    empty_res = FeatureEngineer.compute_session_features(pd.DataFrame())
    assert empty_res.empty
