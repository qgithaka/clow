"""Unit tests for chunked historical data extraction."""

from datetime import datetime, timezone
import pandas as pd
import pytest
from training.data.chunked_extractor import (
    CanonicalTimeframe,
    ChunkedHistoricalExtractor,
    DateRangeChunk,
    OHLCVBar,
)


def test_canonical_timeframe_properties() -> None:
    """Verify CanonicalTimeframe properties."""
    assert CanonicalTimeframe.M1.seconds == 60
    assert CanonicalTimeframe.M5.seconds == 300
    assert CanonicalTimeframe.H1.seconds == 3600
    assert CanonicalTimeframe.D1.seconds == 86400


def test_ohlcv_bar_schema() -> None:
    """Verify OHLCVBar validation."""
    bar = OHLCVBar(
        timestamp_utc=datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc),
        open=1.0850,
        high=1.0860,
        low=1.0840,
        close=1.0855,
        volume=100.0,
        spread=1.2,
    )
    assert bar.open == 1.0850
    assert bar.close == 1.0855


def test_generate_chunks_partitioning() -> None:
    """Verify slicing multi-year ranges into 1-year discrete intervals."""
    extractor = ChunkedHistoricalExtractor(chunk_years=1)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2023, 6, 1, tzinfo=timezone.utc)

    chunks = extractor.generate_chunks(start, end)
    assert len(chunks) == 4
    assert chunks[0].start_utc == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert chunks[0].end_utc == datetime(2021, 1, 1, tzinfo=timezone.utc)
    assert chunks[-1].end_utc == datetime(2023, 6, 1, tzinfo=timezone.utc)


def test_invalid_chunk_dates_raise() -> None:
    """Verify start >= end raises ValueError."""
    extractor = ChunkedHistoricalExtractor()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2023, 1, 1, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="must be strictly earlier than"):
        extractor.generate_chunks(start, end)


def test_extract_full_history_with_mock_fetcher() -> None:
    """Verify stitching multiple chunks with deduplication and progress callback."""
    extractor = ChunkedHistoricalExtractor(chunk_years=1)
    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    end = datetime(2023, 1, 1, tzinfo=timezone.utc)

    def mock_fetcher(chunk: DateRangeChunk) -> pd.DataFrame:
        dates = pd.date_range(chunk.start_utc, chunk.end_utc, freq="12h", tz=timezone.utc)
        return pd.DataFrame({
            "timestamp_utc": dates,
            "open": [1.08] * len(dates),
            "high": [1.09] * len(dates),
            "low": [1.07] * len(dates),
            "close": [1.085] * len(dates),
            "volume": [50.0] * len(dates),
            "tick_volume": [100.0] * len(dates),
            "spread": [1.0] * len(dates),
        })

    progress_events: list[tuple[int, int, int]] = []

    def progress_cb(current_chunk: int, total_chunks: int, rows: int) -> None:
        progress_events.append((current_chunk, total_chunks, rows))

    df = extractor.extract_full_history(
        symbol="EURUSD",
        timeframe=CanonicalTimeframe.H1,
        start_utc=start,
        end_utc=end,
        fetcher_fn=mock_fetcher,
        progress_callback=progress_cb,
    )

    assert not df.empty
    assert len(progress_events) == 2
    # Ensure boundary deduplication
    assert df["timestamp_utc"].is_monotonic_increasing
    assert len(df) == len(df.drop_duplicates(subset=["timestamp_utc"]))
