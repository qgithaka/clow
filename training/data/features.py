"""Scale-free stationary feature engineering pipeline for Clow."""

from datetime import datetime, timezone
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("clow.data.features")


class FeatureEngineer:
    """Calculates strictly causal, scale-free, asset-agnostic financial features."""

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculates causal Average True Range (ATR) using Wilder's EMA smoothing."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        n = len(df)

        if n == 0:
            return pd.Series(dtype=float)

        tr = np.zeros(n, dtype=float)
        tr[0] = high[0] - low[0]

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        atr_series = pd.Series(tr, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        return atr_series

    @classmethod
    def compute_candle_anatomy(cls, df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
        """Computes scale-free candle anatomy and range-to-ATR ratios."""
        if df.empty:
            return pd.DataFrame()

        res = df.copy()
        eps = 1e-8

        open_p = res["open"].values
        high_p = res["high"].values
        low_p = res["low"].values
        close_p = res["close"].values
        total_range = high_p - low_p + eps

        # 1. Body Ratios
        body = close_p - open_p
        abs_body = np.abs(body)
        res["body_ratio"] = body / total_range
        res["abs_body_ratio"] = abs_body / total_range

        # 2. Wick Ratios
        max_oc = np.maximum(open_p, close_p)
        min_oc = np.minimum(open_p, close_p)
        res["upper_wick_ratio"] = (high_p - max_oc) / total_range
        res["lower_wick_ratio"] = (min_oc - low_p) / total_range

        # 3. Normalized Range Percentage
        res["range_pct"] = (high_p - low_p) / (close_p + eps)

        # 4. ATR & Range-to-ATR (Volatility Expansion)
        atr = cls.calculate_atr(res, period=atr_period)
        res["atr"] = atr
        res["range_to_atr"] = (high_p - low_p) / (atr.values + eps)
        res["body_to_atr"] = abs_body / (atr.values + eps)

        return res
