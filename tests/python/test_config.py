"""Unit tests for Clow configuration management."""

import os
from pathlib import Path
import pytest
from training.utils.config import ClowSettings, load_config


def test_default_config_instantiation() -> None:
    """Verify default configuration values."""
    cfg = ClowSettings()
    assert cfg.environment == "development"
    assert cfg.mt5.server == "MetaQuotes-Demo"
    assert cfg.models.foundation_model == "amazon/chronos-t5-small"
    assert cfg.risk.max_account_risk_pct == 1.0
    assert cfg.risk.min_confidence_threshold == 0.65


def test_load_config_from_yaml(tmp_path: Path) -> None:
    """Verify loading from a YAML file."""
    yaml_file = tmp_path / "custom_config.yaml"
    yaml_file.write_text(
        """
environment: production
risk:
  max_account_risk_pct: 0.5
  max_open_trades: 5
models:
  batch_size: 64
""",
        encoding="utf-8",
    )

    cfg = load_config(yaml_file)
    assert cfg.environment == "production"
    assert cfg.risk.max_account_risk_pct == 0.5
    assert cfg.risk.max_open_trades == 5
    assert cfg.models.batch_size == 64


def test_env_var_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify environment variable overrides."""
    monkeypatch.setenv("CLOW_ENVIRONMENT", "staging")
    monkeypatch.setenv("CLOW_RISK__MAX_ACCOUNT_RISK_PCT", "2.0")

    cfg = ClowSettings()
    assert cfg.environment == "staging"
    assert cfg.risk.max_account_risk_pct == 2.0
