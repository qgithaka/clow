"""Dataset health report generation engine."""

import logging
from datetime import UTC, datetime

import pandas as pd
from pydantic import BaseModel, Field

from training.data.chunked_extractor import CanonicalTimeframe
from training.data.validator import DataValidator, ValidationSummary

logger = logging.getLogger("clow.data.health")


class DatasetHealthReport(BaseModel):
    """Complete diagnostic health profile for a dataset."""
    symbol: str
    timeframe: str
    generated_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_rows: int
    start_time_utc: datetime | None = None
    end_time_utc: datetime | None = None
    completeness_pct: float = 100.0
    median_spread_pips: float = 0.0
    max_spread_pips: float = 0.0
    min_price: float = 0.0
    max_price: float = 0.0
    status: str = "PASSED"  # "PASSED", "WARNING", "FAILED"
    validation_summary: ValidationSummary


class HealthReportGenerator:
    """Generates diagnostic reports and markdown summaries for datasets."""

    @classmethod
    def generate_report(
        cls,
        df: pd.DataFrame,
        symbol: str,
        timeframe: CanonicalTimeframe = CanonicalTimeframe.M5,
    ) -> DatasetHealthReport:
        """Analyzes dataset and constructs comprehensive diagnostic report."""
        val_summary = DataValidator.validate(df, timeframe=timeframe)

        if df.empty:
            return DatasetHealthReport(
                symbol=symbol,
                timeframe=timeframe.value,
                total_rows=0,
                completeness_pct=0.0,
                status="FAILED",
                validation_summary=val_summary,
            )

        start_ts = pd.to_datetime(df["timestamp_utc"].iloc[0]).to_pydatetime()
        end_ts = pd.to_datetime(df["timestamp_utc"].iloc[-1]).to_pydatetime()

        median_spread = float(df["spread"].median()) if "spread" in df.columns else 0.0
        max_spread = float(df["spread"].max()) if "spread" in df.columns else 0.0
        min_p = float(df["low"].min())
        max_p = float(df["high"].max())

        # Determine overall status
        if not val_summary.is_valid:
            status = "FAILED"
        elif val_summary.warning_count > 0:
            status = "WARNING"
        else:
            status = "PASSED"

        # Calculate theoretical completeness vs actual row count
        total_seconds = (end_ts - start_ts).total_seconds()
        theoretical_bars = max(1, int(total_seconds / timeframe.seconds))
        # Account for ~28% weekend closure in Forex
        adjusted_theoretical = max(1, int(theoretical_bars * 0.72))
        completeness = min(100.0, round((len(df) / adjusted_theoretical) * 100.0, 2))

        return DatasetHealthReport(
            symbol=symbol,
            timeframe=timeframe.value,
            total_rows=len(df),
            start_time_utc=start_ts,
            end_time_utc=end_ts,
            completeness_pct=completeness,
            median_spread_pips=round(median_spread, 2),
            max_spread_pips=round(max_spread, 2),
            min_price=round(min_p, 5),
            max_price=round(max_p, 5),
            status=status,
            validation_summary=val_summary,
        )

    @classmethod
    def format_markdown_report(cls, report: DatasetHealthReport) -> str:
        """Formats the health report into human-readable markdown table."""
        status_badge = "🟢 PASSED" if report.status == "PASSED" else ("🟡 WARNING" if report.status == "WARNING" else "🔴 FAILED")
        md = f"""# 📊 Dataset Health Diagnostic Report

**Symbol**: `{report.symbol}` | **Timeframe**: `{report.timeframe}` | **Status**: {status_badge}  
**Total Rows**: `{report.total_rows:,}` | **Completeness**: `{report.completeness_pct}%`  
**Time Span**: `{report.start_time_utc}` ──► `{report.end_time_utc}`

---

### 🔍 Metric Breakdown
* **Price Range**: `{report.min_price:.5f}` to `{report.max_price:.5f}`
* **Median Spread**: `{report.median_spread_pips} pips` (Max Spread: `{report.max_spread_pips} pips`)
* **Validation Errors**: `{report.validation_summary.error_count}`
* **Validation Warnings**: `{report.validation_summary.warning_count}`

---

### ⚠️ Detected Issues & Observations
"""
        if not report.validation_summary.issues:
            md += "✅ No data anomalies or geometry corruptions detected.\n"
        else:
            for iss in report.validation_summary.issues:
                md += f"- **[{iss.get('severity', 'INFO')}]** `{iss.get('category', 'GENERAL')}`: {iss.get('message', '')} ({iss.get('row_count', 0)} rows)\n"

        return md
