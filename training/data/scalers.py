"""Strictly causal rolling scalers for Clow feature normalization.

Eliminates look-ahead bias and distribution drift by scaling features solely
using historical sliding windows, with zero future information leakage.
"""

from collections import deque
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("clow.data.scalers")


class RollingZScoreScaler:
    r"""Strictly causal rolling Z-Score scaler ($z = (x - \mu) / (\sigma + \epsilon)$)."""

    def __init__(
        self,
        window: int = 100,
        min_periods: int = 10,
        clip_val: Optional[float] = 5.0,
        eps: float = 1e-8,
    ) -> None:
        self.window = window
        self.min_periods = min_periods
        self.clip_val = clip_val
        self.eps = eps
        self._buffers: Dict[str, deque] = {}

    def transform_series(self, series: pd.Series) -> pd.Series:
        """Transforms a full historical series using causal pandas rolling window."""
        if series.empty:
            return pd.Series(dtype=float)

        rolling = series.rolling(window=self.window, min_periods=self.min_periods)
        mean = rolling.mean()
        std = rolling.std().fillna(0.0)

        z = (series - mean) / (std + self.eps)
        if self.clip_val is not None:
            z = z.clip(lower=-self.clip_val, upper=self.clip_val)

        return z

    def transform_df(self, df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """Transforms specified columns in a DataFrame in-place or on a copy."""
        res = df.copy()
        for col in columns:
            if col in res.columns:
                res[f"{col}_zscore"] = self.transform_series(res[col])
        return res

    def step(self, feature_name: str, value: float) -> float:
        """Single-step causal streaming update for live inference."""
        if feature_name not in self._buffers:
            self._buffers[feature_name] = deque(maxlen=self.window)

        buf = self._buffers[feature_name]
        buf.append(value)

        if len(buf) < self.min_periods:
            return 0.0

        arr = np.array(buf, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(buf) > 1 else 0.0

        z = (value - mean) / (std + self.eps)
        if self.clip_val is not None:
            z = max(-self.clip_val, min(self.clip_val, z))

        return z

    def reset(self) -> None:
        """Resets streaming buffers."""
        self._buffers.clear()


class RollingRobustScaler:
    """Strictly causal rolling Robust Scaler using median and IQR."""

    def __init__(
        self,
        window: int = 100,
        min_periods: int = 10,
        clip_val: Optional[float] = 5.0,
        eps: float = 1e-8,
    ) -> None:
        self.window = window
        self.min_periods = min_periods
        self.clip_val = clip_val
        self.eps = eps
        self._buffers: Dict[str, deque] = {}

    def transform_series(self, series: pd.Series) -> pd.Series:
        """Transforms a series using rolling median and IQR."""
        if series.empty:
            return pd.Series(dtype=float)

        rolling = series.rolling(window=self.window, min_periods=self.min_periods)
        q25 = rolling.quantile(0.25)
        q50 = rolling.quantile(0.50)
        q75 = rolling.quantile(0.75)
        iqr = q75 - q25

        scaled = (series - q50) / (iqr + self.eps)
        if self.clip_val is not None:
            scaled = scaled.clip(lower=-self.clip_val, upper=self.clip_val)

        return scaled

    def transform_df(self, df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """Transforms specified columns in a DataFrame."""
        res = df.copy()
        for col in columns:
            if col in res.columns:
                res[f"{col}_robust"] = self.transform_series(res[col])
        return res

    def step(self, feature_name: str, value: float) -> float:
        """Single-step causal streaming update."""
        if feature_name not in self._buffers:
            self._buffers[feature_name] = deque(maxlen=self.window)

        buf = self._buffers[feature_name]
        buf.append(value)

        if len(buf) < self.min_periods:
            return 0.0

        arr = np.array(buf, dtype=float)
        q25 = float(np.percentile(arr, 25))
        q50 = float(np.percentile(arr, 50))
        q75 = float(np.percentile(arr, 75))
        iqr = q75 - q25

        scaled = (value - q50) / (iqr + self.eps)
        if self.clip_val is not None:
            scaled = max(-self.clip_val, min(self.clip_val, scaled))

        return scaled

    def reset(self) -> None:
        """Resets streaming buffers."""
        self._buffers.clear()


class RollingMinMaxScaler:
    """Strictly causal rolling MinMax Scaler mapping values to [feature_range]."""

    def __init__(
        self,
        window: int = 100,
        min_periods: int = 10,
        feature_range: Tuple[float, float] = (-1.0, 1.0),
        eps: float = 1e-8,
    ) -> None:
        self.window = window
        self.min_periods = min_periods
        self.feature_range = feature_range
        self.eps = eps
        self._buffers: Dict[str, deque] = {}

    def transform_series(self, series: pd.Series) -> pd.Series:
        """Transforms a series using rolling min and max."""
        if series.empty:
            return pd.Series(dtype=float)

        rolling = series.rolling(window=self.window, min_periods=self.min_periods)
        r_min = rolling.min()
        r_max = rolling.max()

        unit_scaled = (series - r_min) / (r_max - r_min + self.eps)
        low, high = self.feature_range
        scaled = low + unit_scaled * (high - low)
        return scaled

    def transform_df(self, df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """Transforms specified columns in a DataFrame."""
        res = df.copy()
        for col in columns:
            if col in res.columns:
                res[f"{col}_minmax"] = self.transform_series(res[col])
        return res

    def step(self, feature_name: str, value: float) -> float:
        """Single-step causal streaming update."""
        if feature_name not in self._buffers:
            self._buffers[feature_name] = deque(maxlen=self.window)

        buf = self._buffers[feature_name]
        buf.append(value)

        if len(buf) < self.min_periods:
            return self.feature_range[0]

        arr = np.array(buf, dtype=float)
        r_min = float(np.min(arr))
        r_max = float(np.max(arr))

        unit_scaled = (value - r_min) / (r_max - r_min + self.eps)
        low, high = self.feature_range
        scaled = low + unit_scaled * (high - low)
        return scaled

    def reset(self) -> None:
        """Resets streaming buffers."""
        self._buffers.clear()
