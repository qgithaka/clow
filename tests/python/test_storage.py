"""Unit tests for Parquet storage and DuckDB queries."""

from pathlib import Path

import pandas as pd

from training.data.storage import ParquetStorageManager


def test_save_and_load_parquet_dataset(tmp_path: Path) -> None:
    """Verify storing and loading Parquet datasets with metadata."""
    manager = ParquetStorageManager(base_dir=tmp_path)
    dates = pd.date_range("2023-01-01 00:00:00", periods=20, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "open": [1.08] * 20,
        "high": [1.09] * 20,
        "low": [1.07] * 20,
        "close": [1.085] * 20,
        "volume": [100.0] * 20,
        "spread": [1.2] * 20,
    })

    meta = manager.save_dataset(df, symbol="EURUSD", timeframe="H1")
    assert meta.symbol == "EURUSD"
    assert meta.row_count == 20
    assert len(meta.checksum_sha256) == 64
    assert Path(meta.file_path).exists()

    loaded = manager.load_dataset(meta.file_path)
    assert len(loaded) == 20
    assert "close" in loaded.columns


def test_duckdb_sql_query(tmp_path: Path) -> None:
    """Verify DuckDB SQL analytical queries on Parquet files."""
    manager = ParquetStorageManager(base_dir=tmp_path)
    dates = pd.date_range("2023-01-01 00:00:00", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "open": [1.0800 + (i * 0.001) for i in range(10)],
        "high": [1.0900] * 10,
        "low": [1.0700] * 10,
        "close": [1.0850] * 10,
        "volume": [50.0] * 10,
        "spread": [1.0] * 10,
    })

    meta = manager.save_dataset(df, symbol="GBPUSD", timeframe="H1")

    # Query using DuckDB SQL
    sql = "SELECT AVG(open) as avg_open, COUNT(*) as total_bars FROM {dataset}"
    res = manager.query_sql(sql, meta.file_path)

    assert len(res) == 1
    assert res["total_bars"].iloc[0] == 10
    assert res["avg_open"].iloc[0] > 1.0800
