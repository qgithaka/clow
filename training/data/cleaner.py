"""Historical market data cleaning and chronological continuity pipeline."""

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from training.data.chunked_extractor import CanonicalTimeframe

logger = logging.getLogger("clow.data.cleaner")


class DataCleaner:
    """Performs deterministic cleaning, deduplication, and weekend gap handling."""

    @staticmethod
    def is_forex_weekend(dt: datetime) -> bool:
        """Returns True if the timestamp falls within standard Forex weekend closure (Fri 22:00 UTC to Sun 21:00 UTC)."""
        weekday = dt.weekday()  # Monday is 0, Sunday is 6
        # Friday after 22:00 UTC
        if weekday == 4 and dt.hour >= 22:
            return True
        # Saturday all day
        if weekday == 5:
            return True
        # Sunday before 21:00 UTC
        if weekday == 6 and dt.hour < 21:
            return True
        return False

    @classmethod
    def clean_dataset(
        cls,
        df: pd.DataFrame,
        timeframe: CanonicalTimeframe = CanonicalTimeframe.M5,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Cleans and sanitizes historical OHLCV data strictly without look-ahead bias."""
        stats = {
            "initial_rows": len(df),
            "duplicates_removed": 0,
            "disordered_rows": 0,
            "invalid_geometry_rows": 0,
            "final_rows": 0,
        }

        if df.empty:
            stats["final_rows"] = 0
            return df.copy(), stats

        clean_df = df.copy()

        # 1. Ensure UTC timezone
        if not pd.api.types.is_datetime64_any_dtype(clean_df["timestamp_utc"]):
            clean_df["timestamp_utc"] = pd.to_datetime(clean_df["timestamp_utc"], utc=True)
        elif clean_df["timestamp_utc"].dt.tz is None:
            clean_df["timestamp_utc"] = clean_df["timestamp_utc"].dt.tz_localize(UTC)
        else:
            clean_df["timestamp_utc"] = clean_df["timestamp_utc"].dt.tz_convert(UTC)

        # 2. Check chronological ordering
        if not clean_df["timestamp_utc"].is_monotonic_increasing:
            stats["disordered_rows"] = 1
            clean_df = clean_df.sort_values("timestamp_utc")

        # 3. Deduplicate on timestamp_utc
        initial_len = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["timestamp_utc"], keep="first")
        stats["duplicates_removed"] = initial_len - len(clean_df)

        # 4. Filter invalid price geometry (High must be >= Low, Open/Close within [Low, High], Prices > 0)
        valid_mask = (
            (clean_df["open"] > 0)
            & (clean_df["high"] > 0)
            & (clean_df["low"] > 0)
            & (clean_df["close"] > 0)
            & (clean_df["high"] >= clean_df["low"])
            & (clean_df["high"] >= clean_df["open"])
            & (clean_df["high"] >= clean_df["close"])
            & (clean_df["low"] <= clean_df["open"])
            & (clean_df["low"] <= clean_df["close"])
            & np.isfinite(clean_df["open"])
            & np.isfinite(clean_df["high"])
            & np.isfinite(clean_df["low"])
            & np.isfinite(clean_df["close"])
        )

        if "spread" in clean_df.columns:
            valid_mask = valid_mask & (clean_df["spread"] >= 0) & np.isfinite(clean_df["spread"])

        invalid_count = len(clean_df) - int(valid_mask.sum())
        stats["invalid_geometry_rows"] = invalid_count
        clean_df = clean_df[valid_mask].reset_index(drop=True)

        stats["final_rows"] = len(clean_df)
        logger.info("Cleaned dataset: %d -> %d rows (Duplicates: %d, Invalid: %d)",
                    stats["initial_rows"], stats["final_rows"], stats["duplicates_removed"], stats["invalid_geometry_rows"])
        return clean_df, stats
