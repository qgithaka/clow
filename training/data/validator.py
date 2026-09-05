"""Strict data health validation engine for market datasets."""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from training.data.chunked_extractor import CanonicalTimeframe
from training.data.cleaner import DataCleaner

logger = logging.getLogger("clow.data.validator")


@dataclass
class ValidationIssue:
    """Represents a specific detected data anomaly."""
    severity: str  # "ERROR" or "WARNING"
    category: str
    message: str
    row_count: int = 1


class ValidationSummary(BaseModel):
    """Structured summary of dataset validation results."""
    is_valid: bool
    total_rows: int
    error_count: int
    warning_count: int
    issues: list[dict[str, Any]] = Field(default_factory=list)


class DataValidator:
    """Detects invalid geometry, bad spreads, disordered timestamps, and non-weekend gaps."""

    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        timeframe: CanonicalTimeframe = CanonicalTimeframe.M5,
        max_allowed_spread_pips: float = 50.0,
    ) -> ValidationSummary:
        """Runs strict health validation on historical OHLCV DataFrame."""
        if df.empty:
            return ValidationSummary(
                is_valid=False,
                total_rows=0,
                error_count=1,
                warning_count=0,
                issues=[{"severity": "ERROR", "category": "EMPTY_DATASET", "message": "Dataset contains 0 rows.", "row_count": 0}],
            )

        issues: list[dict[str, Any]] = []
        error_count = 0
        warning_count = 0

        # 1. Required Columns Check
        required_cols = {"timestamp_utc", "open", "high", "low", "close"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            error_count += len(missing_cols)
            issues.append({
                "severity": "ERROR",
                "category": "MISSING_COLUMNS",
                "message": f"Missing required columns: {list(missing_cols)}",
                "row_count": len(df),
            })
            return ValidationSummary(
                is_valid=False,
                total_rows=len(df),
                error_count=error_count,
                warning_count=warning_count,
                issues=issues,
            )

        # 2. NaN / Inf Check
        nan_mask = df[["open", "high", "low", "close"]].isna().any(axis=1)
        nan_count = int(nan_mask.sum())
        if nan_count > 0:
            error_count += nan_count
            issues.append({
                "severity": "ERROR",
                "category": "NAN_VALUES",
                "message": f"Found {nan_count} rows containing NaN/Inf values.",
                "row_count": nan_count,
            })

        # 3. Negative / Zero Prices
        neg_price_mask = (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
        neg_count = int(neg_price_mask.sum())
        if neg_count > 0:
            error_count += neg_count
            issues.append({
                "severity": "ERROR",
                "category": "NON_POSITIVE_PRICE",
                "message": f"Found {neg_count} rows with non-positive price values.",
                "row_count": neg_count,
            })

        # 4. Impossible OHLC Geometry (High < Low or Open/Close outside [Low, High])
        geom_mask = (
            (df["high"] < df["low"])
            | (df["open"] > df["high"])
            | (df["close"] > df["high"])
            | (df["open"] < df["low"])
            | (df["close"] < df["low"])
        )
        geom_count = int(geom_mask.sum())
        if geom_count > 0:
            error_count += geom_count
            issues.append({
                "severity": "ERROR",
                "category": "INVALID_GEOMETRY",
                "message": f"Found {geom_count} rows violating High >= max(Open, Close) or Low <= min(Open, Close).",
                "row_count": geom_count,
            })

        # 5. Disordered Timestamps
        if not df["timestamp_utc"].is_monotonic_increasing:
            error_count += 1
            issues.append({
                "severity": "ERROR",
                "category": "DISORDERED_TIMESTAMPS",
                "message": "Timestamps are not strictly in ascending chronological order.",
                "row_count": 1,
            })

        # 6. Duplicate Timestamps
        dup_count = int(df["timestamp_utc"].duplicated().sum())
        if dup_count > 0:
            error_count += dup_count
            issues.append({
                "severity": "ERROR",
                "category": "DUPLICATE_TIMESTAMPS",
                "message": f"Found {dup_count} duplicate timestamp rows.",
                "row_count": dup_count,
            })

        # 7. Spread Anomaly Check
        if "spread" in df.columns:
            neg_spread_count = int((df["spread"] < 0).sum())
            if neg_spread_count > 0:
                error_count += neg_spread_count
                issues.append({
                    "severity": "ERROR",
                    "category": "NEGATIVE_SPREAD",
                    "message": f"Found {neg_spread_count} rows with negative spread.",
                    "row_count": neg_spread_count,
                })

            high_spread_count = int((df["spread"] > max_allowed_spread_pips).sum())
            if high_spread_count > 0:
                warning_count += high_spread_count
                issues.append({
                    "severity": "WARNING",
                    "category": "HIGH_SPREAD",
                    "message": f"Found {high_spread_count} rows with extreme spread (> {max_allowed_spread_pips} pips).",
                    "row_count": high_spread_count,
                })

        # 8. Mid-Week Large Gap Detection
        if len(df) > 1 and df["timestamp_utc"].is_monotonic_increasing:
            ts_series = pd.to_datetime(df["timestamp_utc"])
            time_diffs = ts_series.diff()
            expected_step = timedelta(seconds=timeframe.seconds)
            gap_threshold = expected_step * 5  # 5 missing bars in a row

            for i in range(1, len(df)):
                delta = time_diffs.iloc[i]
                if delta > gap_threshold:
                    prev_ts = ts_series.iloc[i - 1].to_pydatetime()
                    curr_ts = ts_series.iloc[i].to_pydatetime()
                    # If the gap is not within standard weekend, flag it
                    if not (DataCleaner.is_forex_weekend(prev_ts) and DataCleaner.is_forex_weekend(curr_ts - timedelta(hours=2))):
                        warning_count += 1
                        issues.append({
                            "severity": "WARNING",
                            "category": "MIDWEEK_DATA_GAP",
                            "message": f"Mid-week gap of {delta} detected between {prev_ts} and {curr_ts}.",
                            "row_count": 1,
                        })
                        if warning_count >= 10:  # Cap gap issue reporting
                            break

        is_valid = error_count == 0
        return ValidationSummary(
            is_valid=is_valid,
            total_rows=len(df),
            error_count=error_count,
            warning_count=warning_count,
            issues=issues,
        )
