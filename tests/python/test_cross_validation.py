"""Unit tests for PurgedWalkForwardCV with embargo."""

import numpy as np
import pandas as pd
import pytest

from training.validation.cross_validation import PurgedWalkForwardCV


def test_purged_walk_forward_cv_causality_and_purging() -> None:
    """Verify strictly chronological splits and enforced purging gaps across all folds."""
    n_bars = 500
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "close": np.linspace(1.0800, 1.0900, n_bars),
    })

    purge_len = 30
    cv = PurgedWalkForwardCV(
        n_splits=4,
        min_train_ratio=0.40,
        purge_window=purge_len,
        embargo_pct=0.01,
        expanding=True,
    )

    splits = list(cv.split(df))
    assert len(splits) >= 3

    for split in splits:
        # 1. Train strictly precedes test
        assert split.train_end_idx <= split.test_start_idx
        assert split.train_df["timestamp_utc"].max() < split.test_df["timestamp_utc"].min()

        # 2. Purging gap is strictly enforced
        gap = split.test_start_idx - split.train_end_idx
        assert gap >= purge_len, f"Purge violation in fold {split.fold_idx}: gap={gap}"

        # 3. Expanding window starts at index 0
        assert split.train_start_idx == 0


def test_rolling_window_mode() -> None:
    """Verify rolling (non-expanding) window behavior."""
    n_bars = 400
    df = pd.DataFrame({
        "timestamp_utc": pd.date_range("2023-01-01", periods=n_bars, freq="5min", tz="UTC"),
        "close": np.random.randn(n_bars),
    })

    cv = PurgedWalkForwardCV(n_splits=3, min_train_ratio=0.30, purge_window=20, expanding=False)
    splits = list(cv.split(df))
    assert len(splits) > 0


def test_invalid_parameters_and_empty_df() -> None:
    """Verify validation error handling."""
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        PurgedWalkForwardCV(n_splits=1)

    cv = PurgedWalkForwardCV(n_splits=3)
    assert list(cv.split(pd.DataFrame())) == []
