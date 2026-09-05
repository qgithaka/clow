"""Central configuration management for Clow AI Research & Training Engine."""

from pathlib import Path
from typing import Any, Literal
import os
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MT5Settings(BaseModel):
    """MetaTrader 5 connection configuration."""
    server: str = "MetaQuotes-Demo"
    login: int = 0
    password: str = ""
    path: str = ""
    timeout_seconds: int = 30
    portable: bool = False


class DataSettings(BaseModel):
    """Historical data storage and chunking settings."""
    storage_dir: Path = Field(default_factory=lambda: Path("data/storage"))
    parquet_dir: Path = Field(default_factory=lambda: Path("data/parquet"))
    chunk_years: int = 1
    canonical_timeframes: list[str] = Field(default_factory=lambda: ["M1", "M5", "M15", "H1", "H4", "D1"])


class ModelSettings(BaseModel):
    """Time-series foundation and tactical model settings."""
    foundation_model: str = "amazon/chronos-t5-small"
    context_length: int = 64
    prediction_length: int = 1
    quantile_levels: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 5
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"


class RiskSettings(BaseModel):
    """Sovereign risk parameters and execution gates."""
    max_account_risk_pct: float = 1.0
    max_daily_drawdown_pct: float = 4.0
    max_open_trades: int = 3
    max_spread_pips: float = 2.5
    min_confidence_threshold: float = 0.65
    default_expiration_bars: int = 3


class LoggingSettings(BaseModel):
    """Logging and telemetry settings."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_format: bool = False
    log_dir: Path = Field(default_factory=lambda: Path("logs"))


class ClowSettings(BaseSettings):
    """Root configuration object for Clow."""
    model_config = SettingsConfigDict(
        env_prefix="CLOW_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Literal["development", "staging", "production"] = "development"
    mt5: MT5Settings = Field(default_factory=MT5Settings)
    data: DataSettings = Field(default_factory=DataSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def load_config(config_path: Path | str | None = None) -> ClowSettings:
    """Load configuration from YAML file and apply environment variable overrides."""
    if config_path is None:
        default_file = Path("config/default_config.yaml")
        if default_file.exists():
            config_path = default_file

    config_dict: dict[str, Any] = {}
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    config_dict = loaded

    return ClowSettings(**config_dict)
