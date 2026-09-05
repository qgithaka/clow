"""Chunked historical time-window data extraction engine."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger("clow.data.extractor")

try:
    import MetaTrader5 as mt5  # type: ignore
    HAS_MT5 = True
except ImportError:
    mt5 = None
    HAS_MT5 = False


class CanonicalTimeframe(str, Enum):
    """Supported canonical timeframe frequencies."""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"

    @property
    def seconds(self) -> int:
        mapping = {
            "M1": 60,
            "M5": 300,
            "M15": 900,
            "H1": 3600,
            "H4": 14400,
            "D1": 86400,
        }
        return mapping[self.value]

    @property
    def mt5_timeframe(self) -> int:
        if not HAS_MT5 or mt5 is None:
            return 0
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        return mapping.get(self.value, mt5.TIMEFRAME_M5)


class OHLCVBar(BaseModel):
    """Canonical OHLCV bar representation."""
    timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    tick_volume: float = 0.0
    spread: float = 0.0


@dataclass
class DateRangeChunk:
    """Defines a discrete time slice for chunked querying."""
    chunk_index: int
    start_utc: datetime
    end_utc: datetime


class ChunkedHistoricalExtractor:
    """Paginates historical requests into time-window chunks to bypass row limits."""

    def __init__(self, chunk_years: int = 1) -> None:
        self.chunk_years = max(1, chunk_years)

    def generate_chunks(
        self,
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[DateRangeChunk]:
        """Splits a multi-year date range into chronological time slices."""
        if start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=UTC)
        if end_utc.tzinfo is None:
            end_utc = end_utc.replace(tzinfo=UTC)

        if start_utc >= end_utc:
            raise ValueError(f"start_utc ({start_utc}) must be strictly earlier than end_utc ({end_utc})")

        chunks: list[DateRangeChunk] = []
        current_start = start_utc
        idx = 0

        while current_start < end_utc:
            try:
                next_year = current_start.year + self.chunk_years
                current_end = current_start.replace(year=next_year)
            except ValueError:
                # Handle leap years (e.g. Feb 29)
                current_end = current_start.replace(year=next_year, day=28)

            if current_end > end_utc:
                current_end = end_utc

            chunks.append(DateRangeChunk(chunk_index=idx, start_utc=current_start, end_utc=current_end))
            current_start = current_end
            idx += 1

        return chunks

    def fetch_chunk_from_mt5(
        self,
        symbol: str,
        timeframe: CanonicalTimeframe,
        chunk: DateRangeChunk,
    ) -> pd.DataFrame:
        """Fetches a single chunk of OHLCV data from MT5 terminal."""
        if not HAS_MT5 or mt5 is None:
            logger.warning("MT5 library unavailable. Returning empty chunk DataFrame.")
            return pd.DataFrame()

        rates = mt5.copy_rates_range(
            symbol,
            timeframe.mt5_timeframe,
            chunk.start_utc,
            chunk.end_utc,
        )

        if rates is None or len(rates) == 0:
            logger.debug("No rates returned for %s in chunk %s", symbol, chunk)
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["timestamp_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"real_volume": "volume"})
        return df[["timestamp_utc", "open", "high", "low", "close", "volume", "tick_volume", "spread"]]

    def extract_full_history(
        self,
        symbol: str,
        timeframe: CanonicalTimeframe,
        start_utc: datetime,
        end_utc: datetime,
        fetcher_fn: Callable[[DateRangeChunk], pd.DataFrame] | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> pd.DataFrame:
        """Extracts and stitches all chunks across the full date range."""
        chunks = self.generate_chunks(start_utc, end_utc)
        total_chunks = len(chunks)
        collected_frames: list[pd.DataFrame] = []
        total_rows_collected = 0

        logger.info("Extracting %s (%s) across %d chronological chunks...", symbol, timeframe.value, total_chunks)

        for chunk in chunks:
            if fetcher_fn is not None:
                chunk_df = fetcher_fn(chunk)
            else:
                chunk_df = self.fetch_chunk_from_mt5(symbol, timeframe, chunk)

            if not chunk_df.empty:
                collected_frames.append(chunk_df)
                total_rows_collected += len(chunk_df)

            if progress_callback is not None:
                progress_callback(chunk.chunk_index + 1, total_chunks, total_rows_collected)

        if not collected_frames:
            return pd.DataFrame(columns=["timestamp_utc", "open", "high", "low", "close", "volume", "tick_volume", "spread"])

        stitched_df = pd.concat(collected_frames, ignore_index=True)
        # Deduplicate and sort chronologically
        stitched_df = stitched_df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
        return stitched_df
