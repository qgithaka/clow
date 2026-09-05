"""Unit tests for Clow structured logging."""

import json
import logging
from pathlib import Path
from training.utils.logging import JSONFormatter, setup_logger


def test_standard_logger_setup(tmp_path: Path) -> None:
    """Verify setup_logger creates stream and file handlers."""
    logger = setup_logger(name="test_logger_std", level="DEBUG", log_dir=tmp_path)
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 2

    logger.info("Standard test info message")
    log_file = tmp_path / "test_logger_std.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Standard test info message" in content


def test_json_formatter() -> None:
    """Verify JSONFormatter outputs valid JSON strings."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_json",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Structured JSON event occurred",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_json"
    assert parsed["message"] == "Structured JSON event occurred"
    assert "timestamp" in parsed
