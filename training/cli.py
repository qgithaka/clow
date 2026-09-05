"""Clow Command-Line Interface."""

import sys

import typer
from rich.console import Console
from rich.table import Table

from training.utils.config import load_config
from training.utils.logging import setup_logger

app = typer.Typer(name="clow", help="Clow: Predictive AI Quantitative Trading Terminal & Research Engine")
console = Console()
logger = setup_logger("clow.cli")

__version__ = "0.1.0"


def version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold cyan]Clow Engine[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show Clow version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Clow platform entry point."""
    pass


@app.command()
def info() -> None:
    """Displays platform configuration and runtime state."""
    cfg = load_config()
    table = Table(title="Clow Runtime Environment")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("Environment", cfg.environment)
    table.add_row("Foundation Model", cfg.models.foundation_model)
    table.add_row("Risk Max Drawdown", f"{cfg.risk.max_daily_drawdown_pct}%")
    table.add_row("Max Account Risk", f"{cfg.risk.max_account_risk_pct}%")
    table.add_row("MT5 Server", cfg.mt5.server)

    console.print(table)


@app.command()
def test() -> None:
    """Executes the Python test suite."""
    import pytest
    console.print("[yellow]Running Python test suite...[/yellow]")
    ret = pytest.main(["tests/python", "-v"])
    sys.exit(ret)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
