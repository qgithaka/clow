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

    @classmethod
    def compute_volatility_features(
        cls,
        df: pd.DataFrame,
        bb_period: int = 20,
        bb_num_std: float = 2.0,
        atr_short: int = 7,
        atr_medium: int = 14,
        atr_long: int = 50,
    ) -> pd.DataFrame:
        """Computes scale-free volatility regime features (Normalized ATR, Bollinger Band width, expansion ratios)."""
        if df.empty:
            return pd.DataFrame()

        res = df.copy()
        eps = 1e-8
        close = res["close"]
        high = res["high"]
        low = res["low"]

        # 1. Multi-Period ATRs
        atr_s = cls.calculate_atr(res, period=atr_short)
        atr_m = cls.calculate_atr(res, period=atr_medium)
        atr_l = cls.calculate_atr(res, period=atr_long)

        res[f"atr_{atr_short}"] = atr_s
        res[f"atr_{atr_medium}"] = atr_m
        res[f"atr_{atr_long}"] = atr_l

        # 2. Normalized ATR Percentages and Ratios
        res["atr_pct"] = atr_m / (close + eps)
        res["volatility_expansion_ratio"] = atr_s / (atr_l + eps)
        res["range_expansion_ratio"] = (high - low) / (atr_m + eps)

        # 3. Bollinger Bands & Normalized Bandwidth
        bb_mid = close.rolling(window=bb_period, min_periods=1).mean()
        bb_std = close.rolling(window=bb_period, min_periods=1).std().fillna(0.0)
        bb_upper = bb_mid + bb_num_std * bb_std
        bb_lower = bb_mid - bb_num_std * bb_std

        res["bb_upper"] = bb_upper
        res["bb_lower"] = bb_lower
        res["bb_mid"] = bb_mid
        res["bb_width_pct"] = (bb_upper - bb_lower) / (bb_mid + eps)
        res["bb_width_atr"] = (bb_upper - bb_lower) / (atr_m + eps)
        res["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower + eps)

        # 4. Keltner Channels & Volatility Squeeze State
        keltner_mid = close.ewm(span=bb_period, adjust=False).mean()
        keltner_upper = keltner_mid + 1.5 * atr_m
        keltner_lower = keltner_mid - 1.5 * atr_m
        res["is_volatility_squeeze"] = ((bb_upper < keltner_upper) & (bb_lower > keltner_lower)).astype(float)

        return res

    @classmethod
    def compute_session_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Computes institutional Forex session masks and cyclical continuous time embeddings."""
        if df.empty or "timestamp_utc" not in df.columns:
            return pd.DataFrame()

        res = df.copy()
        ts = pd.to_datetime(res["timestamp_utc"], utc=True)

        hour = ts.dt.hour
        minute = ts.dt.minute
        dow = ts.dt.dayofweek
        minute_of_day = hour * 60 + minute

        # 1. Cyclical Time Continuous Embeddings (stationary [-1.0, 1.0])
        res["sin_time_of_day"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
        res["cos_time_of_day"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
        res["sin_day_of_week"] = np.sin(2.0 * np.pi * dow / 7.0)
        res["cos_day_of_week"] = np.cos(2.0 * np.pi * dow / 7.0)

        # 2. Institutional Session Binary Masks (in UTC)
        # Asian Session: 00:00 - 08:00 UTC
        res["session_asian"] = ((hour >= 0) & (hour < 8)).astype(float)

        # London Session: 07:00 - 16:00 UTC
        res["session_london"] = ((hour >= 7) & (hour < 16)).astype(float)

        # London Open Breakout Window: 07:00 - 10:00 UTC
        res["session_london_open"] = ((hour >= 7) & (hour < 10)).astype(float)

        # New York Session: 12:00 - 21:00 UTC
        res["session_ny"] = ((hour >= 12) & (hour < 21)).astype(float)

        # London / NY Peak Liquidity Overlap: 12:00 - 16:00 UTC
        res["session_ny_london_overlap"] = ((hour >= 12) & (hour < 16)).astype(float)

        # London Fixing / Close Window: 15:00 - 17:00 UTC
        res["session_london_close"] = ((hour >= 15) & (hour < 17)).astype(float)

        # Weekend Approaching (Friday >= 20:00 UTC)
        res["is_weekend_close_risk"] = ((dow == 4) & (hour >= 20)).astype(float)

        return res
