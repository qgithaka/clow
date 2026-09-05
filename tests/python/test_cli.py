"""Unit tests for Clow CLI."""

from typer.testing import CliRunner

from training.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """Verify clow --help prints usage."""
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "Clow: Predictive AI Quantitative Trading Terminal" in res.stdout


def test_cli_version() -> None:
    """Verify clow --version outputs current version."""
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert "0.1.0" in res.stdout


def test_cli_info() -> None:
    """Verify clow info outputs environment summary."""
    res = runner.invoke(app, ["info"])
    assert res.exit_code == 0
    assert "Clow Runtime Environment" in res.stdout
    assert "MetaQuotes-Demo" in res.stdout
