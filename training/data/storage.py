"""Immutable Parquet dataset storage and DuckDB analytical query layer."""

import hashlib
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

logger = logging.getLogger("clow.data.storage")


class DatasetMetadata(BaseModel):
    """Metadata descriptor for versioned immutable datasets."""
    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timeframe: str
    start_time_utc: datetime
    end_time_utc: datetime
    row_count: int
    checksum_sha256: str
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    file_path: str
    feature_version: int = 1


class ParquetStorageManager:
    """Manages immutable dataset saving, versioning, checksums, and DuckDB access."""

    def __init__(self, base_dir: Path | str = "data/parquet") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_checksum(file_path: Path) -> str:
        """Calculates SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def save_dataset(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        custom_name: str | None = None,
    ) -> DatasetMetadata:
        """Saves a cleaned DataFrame to compressed immutable Parquet format."""
        if df.empty:
            raise ValueError("Cannot store empty DataFrame in Parquet dataset.")

        symbol_clean = symbol.upper().replace("/", "_")
        tf_clean = timeframe.upper()

        if custom_name:
            filename = f"{custom_name}.parquet"
        else:
            filename = f"{symbol_clean}_{tf_clean}.parquet"

        target_file = self.base_dir / filename

        # Convert to Arrow Table
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            target_file,
            compression="snappy",
            use_dictionary=True,
        )

        checksum = self.calculate_checksum(target_file)
        start_ts = pd.to_datetime(df["timestamp_utc"].iloc[0]).to_pydatetime()
        end_ts = pd.to_datetime(df["timestamp_utc"].iloc[-1]).to_pydatetime()

        metadata = DatasetMetadata(
            symbol=symbol_clean,
            timeframe=tf_clean,
            start_time_utc=start_ts,
            end_time_utc=end_ts,
            row_count=len(df),
            checksum_sha256=checksum,
            file_path=str(target_file),
        )

        # Save metadata companion JSON
        meta_file = self.base_dir / f"{filename}.meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))

        logger.info("Saved dataset %s (%d rows, SHA256: %s)", filename, len(df), checksum[:8])
        return metadata

    def load_dataset(self, file_path: Path | str) -> pd.DataFrame:
        """Loads a Parquet dataset into a pandas DataFrame."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Parquet dataset not found: {p}")
        return pd.read_parquet(p)

    def query_sql(self, sql: str, dataset_path: Path | str) -> pd.DataFrame:
        """Executes analytical SQL query directly over Parquet file via DuckDB."""
        p = Path(dataset_path)
        if not p.exists():
            raise FileNotFoundError(f"Parquet dataset not found: {p}")

        conn = duckdb.connect(database=":memory:")
        # Replace template placeholder {dataset} with parquet path
        clean_path = str(p).replace("\\", "/")
        formatted_sql = sql.replace("{dataset}", f"'{clean_path}'")
        res_df = conn.execute(formatted_sql).df()
        conn.close()
        return res_df
