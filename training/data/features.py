"""Scale-free stationary feature engineering pipeline for Clow."""

from datetime import datetime, timezone
import logging
from typing import Optional, Sequence, Tuple
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

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculates causal Relative Strength Index (RSI) via Wilder's exponential smoothing."""
        if df.empty or "close" not in df.columns:
            return pd.Series(dtype=float)

        close = df["close"]
        delta = close.diff()
        eps = 1e-8

        gain = delta.clip(lower=0.0).fillna(0.0)
        loss = (-delta.clip(upper=0.0)).fillna(0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

        rs = avg_gain / (avg_loss + eps)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

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

    @classmethod
    def compute_momentum_indicators(
        cls,
        df: pd.DataFrame,
        ema_periods: Sequence[int] = (20, 50, 200),
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
    ) -> pd.DataFrame:
        """Computes stationary normalized momentum indicators (EMA deviations, stationary RSI, scale-free MACD)."""
        if df.empty:
            return pd.DataFrame()

        res = df.copy()
        eps = 1e-8
        close = res["close"]

        # Ensure ATR is available
        if "atr" not in res.columns:
            res["atr"] = cls.calculate_atr(res, period=atr_period)
        atr = res["atr"]

        # 1. Stationary RSI centered around 0 in [-1.0, 1.0]
        rsi_raw = cls.calculate_rsi(res, period=rsi_period)
        res["rsi"] = rsi_raw
        res["rsi_stationary"] = (rsi_raw - 50.0) / 50.0

        # 2. Rolling Z-Score Deviations from EMAs
        for p in ema_periods:
            ema = close.ewm(span=p, adjust=False).mean()
            rolling_std = close.rolling(window=p, min_periods=1).std().fillna(0.0)
            res[f"ema_{p}"] = ema
            res[f"ema_dist_atr_{p}"] = (close - ema) / (atr + eps)
            res[f"ema_zscore_{p}"] = (close - ema) / (rolling_std + eps)
            res[f"ema_dist_pct_{p}"] = (close - ema) / (ema + eps)

        # 3. Scale-Free MACD Normalized Ratios
        ema_fast_s = close.ewm(span=macd_fast, adjust=False).mean()
        ema_slow_s = close.ewm(span=macd_slow, adjust=False).mean()
        macd_diff = ema_fast_s - ema_slow_s

        res["macd_dist_atr"] = macd_diff / (atr + eps)
        res["macd_signal_atr"] = res["macd_dist_atr"].ewm(span=macd_signal, adjust=False).mean()
        res["macd_hist_atr"] = res["macd_dist_atr"] - res["macd_signal_atr"]

        res["macd_dist_pct"] = macd_diff / (close + eps)
        res["macd_signal_pct"] = res["macd_dist_pct"].ewm(span=macd_signal, adjust=False).mean()
        res["macd_hist_pct"] = res["macd_dist_pct"] - res["macd_signal_pct"]

        return res
