"""Unit tests for dataset health report generation."""

import pandas as pd

from training.data.chunked_extractor import CanonicalTimeframe
from training.data.health_report import HealthReportGenerator


def test_health_report_clean_data() -> None:
    """Verify health report generation on clean dataset."""
    dates = pd.date_range("2023-01-02 00:00:00", periods=100, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "open": [1.0850] * 100,
        "high": [1.0865] * 100,
        "low": [1.0845] * 100,
        "close": [1.0855] * 100,
        "volume": [10.0] * 100,
        "spread": [1.2] * 100,
    })

    report = HealthReportGenerator.generate_report(df, symbol="EURUSD", timeframe=CanonicalTimeframe.M5)
    assert report.symbol == "EURUSD"
    assert report.total_rows == 100
    assert report.status == "PASSED"
    assert report.median_spread_pips == 1.2
    assert report.min_price == 1.0845
    assert report.max_price == 1.0865

    md = HealthReportGenerator.format_markdown_report(report)
    assert "Dataset Health Diagnostic Report" in md
    assert "PASSED" in md


def test_health_report_empty_data() -> None:
    """Verify health report handles empty data."""
    empty_df = pd.DataFrame()
    report = HealthReportGenerator.generate_report(empty_df, symbol="GBPUSD", timeframe=CanonicalTimeframe.H1)
    assert report.status == "FAILED"
    assert report.total_rows == 0
