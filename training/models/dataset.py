"""Time-series sliding window dataset generator for Clow-Forecaster foundation model.

Constructs strictly causal sliding context windows with multi-target candle anatomy,
directional classification labels, and quantile excursion targets.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesSlidingWindowDataset(Dataset):
    """PyTorch Dataset generating sliding context windows for next-candle forecasting."""

    DEFAULT_FEATURE_COLS = [
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "range_pct",
        "range_to_atr",
        "body_to_atr",
        "rsi_stationary",
        "ema_dist_atr_20",
        "ema_dist_atr_50",
        "macd_dist_atr",
        "bb_width_pct",
        "volatility_expansion_ratio",
        "sin_time_of_day",
        "cos_time_of_day",
        "session_london",
        "session_ny",
    ]

    DEFAULT_TARGET_COLS = [
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "range_to_atr",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
        context_length: int = 64,
        prediction_horizon: int = 1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty.")

        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.eps = eps

        # Validate feature columns
        available_cols = set(df.columns)
        if feature_cols is None:
            self.feature_cols = [c for c in self.DEFAULT_FEATURE_COLS if c in available_cols]
            if not self.feature_cols:
                # Fallback to numeric columns
                exclude = {"timestamp_utc", "symbol"}
                self.feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        else:
            self.feature_cols = list(feature_cols)

        if not self.feature_cols:
            raise ValueError("No valid feature columns found for TimeSeriesSlidingWindowDataset.")

        # Target columns
        if target_cols is None:
            self.target_cols = [c for c in self.DEFAULT_TARGET_COLS if c in available_cols]
            if not self.target_cols:
                self.target_cols = ["body_ratio"] if "body_ratio" in available_cols else [self.feature_cols[0]]
        else:
            self.target_cols = list(target_cols)

        # Precompute arrays for speed
        n_samples = len(df)
        self.num_windows = n_samples - self.context_length - self.prediction_horizon + 1
        if self.num_windows <= 0:
            raise ValueError(
                f"DataFrame length ({n_samples}) is insufficient for context_length "
                f"({self.context_length}) + prediction_horizon ({self.prediction_horizon})."
            )

        # Feature matrix [N, D]
        self.features_arr = np.nan_to_num(df[self.feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Target anatomy matrix [N, T]
        self.targets_arr = np.nan_to_num(df[self.target_cols].values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Directional labels: 1.0 if close >= open else 0.0
        open_vals = df["open"].values if "open" in df.columns else np.zeros(n_samples)
        close_vals = df["close"].values if "close" in df.columns else np.zeros(n_samples)
        high_vals = df["high"].values if "high" in df.columns else np.zeros(n_samples)
        low_vals = df["low"].values if "low" in df.columns else np.zeros(n_samples)
        atr_vals = df["atr"].values if "atr" in df.columns else (np.abs(close_vals - open_vals) + eps)

        self.direction_arr = (close_vals >= open_vals).astype(np.float32)

        # Quantile excursions relative to open price normalized by ATR:
        # High excursion = (High - Open) / (ATR + eps)
        # Low excursion = (Open - Low) / (ATR + eps)
        high_excursion = (high_vals - open_vals) / (atr_vals + eps)
        low_excursion = (open_vals - low_vals) / (atr_vals + eps)
        self.quantiles_arr = np.nan_to_num(np.stack([high_excursion, low_excursion], axis=-1), nan=0.0).astype(np.float32)

        # Normalized return relative to ATR: (Close - Open) / (ATR + eps)
        self.return_atr_arr = np.nan_to_num((close_vals - open_vals) / (atr_vals + eps), nan=0.0).astype(np.float32)

        # Timestamps if present
        self.timestamps = df["timestamp_utc"].values if "timestamp_utc" in df.columns else np.arange(n_samples)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.num_windows}.")

        ctx_start = idx
        ctx_end = idx + self.context_length
        tgt_start = ctx_end
        tgt_end = ctx_end + self.prediction_horizon

        context = torch.from_numpy(self.features_arr[ctx_start:ctx_end])  # [L, D]
        target_candle = torch.from_numpy(self.targets_arr[tgt_start:tgt_end])  # [H, T]
        direction = torch.from_numpy(self.direction_arr[tgt_start:tgt_end])  # [H]
        quantiles = torch.from_numpy(self.quantiles_arr[tgt_start:tgt_end])  # [H, 2]
        return_atr = torch.from_numpy(self.return_atr_arr[tgt_start:tgt_end])  # [H]

        return {
            "context": context,
            "target_candle": target_candle,
            "target_direction": direction,
            "target_quantiles": quantiles,
            "target_return_atr": return_atr,
            "target_idx": torch.tensor(tgt_start, dtype=torch.long),
        }

    @staticmethod
    def temporal_train_val_test_split(
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_window: int = 64,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs strictly chronological dataset split with purging gaps between folds.
        
        Purging removes `purge_window` bars at boundaries so context windows never
        span across train/val or val/test boundaries.
        """
        n = len(df)
        if n == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total

        train_end = int(n * train_ratio)
        val_start = train_end + purge_window
        val_end = int(n * (train_ratio + val_ratio))
        test_start = val_end + purge_window

        if val_start >= val_end or test_start >= n:
            # If dataset is too small for purging gaps, fall back to unpurged slices
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()
        else:
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[val_start:val_end].copy()
            test_df = df.iloc[test_start:].copy()

        return train_df, val_df, test_df
