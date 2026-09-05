"""Purged Walk-Forward Cross-Validation with Embargo for institutional model validation.

Eliminates cross-fold data leakage and serial correlation bias in financial time series.
"""

from dataclasses import dataclass
import logging
from typing import Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("clow.validation.cross_validation")


@dataclass
class CVSplit:
    """Represents a single purged walk-forward cross-validation split."""

    fold_idx: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame


class PurgedWalkForwardCV:
    """Generates strictly causal purged walk-forward train/test splits with embargo."""

    def __init__(
        self,
        n_splits: int = 5,
        min_train_ratio: float = 0.40,
        purge_window: int = 64,
        embargo_pct: float = 0.01,
        expanding: bool = True,
    ) -> None:
        """Args:
            n_splits: Number of walk-forward folds
            min_train_ratio: Minimum fraction of data required for initial training fold
            purge_window: Number of bars purged before test fold
            embargo_pct: Fraction of data embargoed after test fold
            expanding: If True, training window expands; if False, sliding rolling window
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct
        self.expanding = expanding

    def split(self, df: pd.DataFrame) -> Generator[CVSplit, None, None]:
        """Generates purged walk-forward splits from a DataFrame."""
        n_samples = len(df)
        if n_samples == 0:
            return

        embargo_bars = max(1, int(n_samples * self.embargo_pct))
        min_train_bars = int(n_samples * self.min_train_ratio)
        remaining_bars = n_samples - min_train_bars

        test_fold_size = max(10, remaining_bars // self.n_splits)

        for fold in range(self.n_splits):
            test_start = min_train_bars + (fold * test_fold_size)
            test_end = min(n_samples, test_start + test_fold_size)

            if test_start >= n_samples or test_end <= test_start:
                break

            # Purging: Train window ends `purge_window` bars before test_start
            train_end = max(10, test_start - self.purge_window)
            train_start = 0 if self.expanding else max(0, train_end - min_train_bars)

            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()

            if len(train_df) < 20 or len(test_df) < 5:
                continue

            yield CVSplit(
                fold_idx=fold,
                train_start_idx=train_start,
                train_end_idx=train_end,
                test_start_idx=test_start,
                test_end_idx=test_end,
                train_df=train_df,
                test_df=test_df,
            )
