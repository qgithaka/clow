"""Multi-Timeframe Hierarchical Context Aligner for Clow.

Aligns macro timeframe features (e.g. H1, H4, D1) to micro execution bars (e.g. M1, M5)
with mathematical guarantee of ZERO look-ahead bias and zero data leakage.
"""

import logging
from datetime import timedelta

import pandas as pd

logger = logging.getLogger("clow.data.multi_timeframe")

# Common pandas resampling frequencies mapped to duration
TIMEFRAME_DELTAS = {
    "M1": timedelta(minutes=1),
    "1min": timedelta(minutes=1),
    "M5": timedelta(minutes=5),
    "5min": timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "15min": timedelta(minutes=15),
    "M30": timedelta(minutes=30),
    "30min": timedelta(minutes=30),
    "H1": timedelta(hours=1),
    "1h": timedelta(hours=1),
    "H4": timedelta(hours=4),
    "4h": timedelta(hours=4),
    "D1": timedelta(days=1),
    "1d": timedelta(days=1),
}


class MultiTimeframeAligner:
    """Strictly causal hierarchical multi-timeframe feature aligner."""

    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resamples lower timeframe OHLCV data into higher timeframe bars.
        
        Preserves canonical OHLCV columns:
        - open: first
        - high: max
        - low: min
        - close: last
        - volume: sum
        - spread: mean (optional)
        """
        if df.empty or "timestamp_utc" not in df.columns:
            return pd.DataFrame()

        work_df = df.copy()
        work_df["timestamp_utc"] = pd.to_datetime(work_df["timestamp_utc"], utc=True)
        work_df = work_df.sort_values("timestamp_utc").set_index("timestamp_utc")

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        if "spread" in work_df.columns:
            agg_dict["spread"] = "mean"

        resampled = work_df.resample(rule, closed="left", label="left").agg(agg_dict).dropna(subset=["open", "close"]).reset_index()
        return resampled

    @classmethod
    def align_higher_timeframe(
        cls,
        ltf_df: pd.DataFrame,
        htf_df: pd.DataFrame,
        htf_name: str,
        htf_bar_delta: timedelta | None = None,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Aligns HTF features to LTF bars using backward asof merge on bar close timestamps.
        
        Zero Look-Ahead Guarantee:
        An HTF bar starting at T with duration D closes at T + D.
        An LTF bar at timestamp t only has access to HTF bars where T + D <= t.
        """
        if ltf_df.empty:
            return pd.DataFrame()
        if htf_df.empty:
            return ltf_df.copy()

        ltf_sorted = ltf_df.copy()
        ltf_sorted["timestamp_utc"] = pd.to_datetime(ltf_sorted["timestamp_utc"], utc=True)
        ltf_sorted = ltf_sorted.sort_values("timestamp_utc")

        htf_work = htf_df.copy()
        htf_work["timestamp_utc"] = pd.to_datetime(htf_work["timestamp_utc"], utc=True)
        htf_work = htf_work.sort_values("timestamp_utc")

        # Determine HTF bar duration if not explicitly passed
        if htf_bar_delta is None:
            if htf_name in TIMEFRAME_DELTAS:
                htf_bar_delta = TIMEFRAME_DELTAS[htf_name]
            elif len(htf_work) > 1:
                # Infer from median delta
                diffs = htf_work["timestamp_utc"].diff().dropna()
                htf_bar_delta = diffs.median().to_pytimedelta()
            else:
                htf_bar_delta = timedelta(hours=1)

        # HTF features are completed and valid only at open_time + htf_bar_delta
        htf_work["htf_close_time"] = htf_work["timestamp_utc"] + htf_bar_delta

        # Select columns to include
        if feature_cols is None:
            # Exclude raw OHLC and internal timestamps, take engineered features
            exclude = {"open", "high", "low", "close", "volume", "spread", "symbol", "timestamp_utc"}
            feature_cols = [c for c in htf_work.columns if c not in exclude and c != "htf_close_time"]

        # Prefix feature columns with HTF name
        rename_map = {c: f"{htf_name}_{c}" for c in feature_cols}
        htf_merge_subset = htf_work[["htf_close_time"] + feature_cols].rename(columns=rename_map)

        # Causal backward merge_asof: match ltf timestamp >= htf_close_time
        merged = pd.merge_asof(
            ltf_sorted,
            htf_merge_subset,
            left_on="timestamp_utc",
            right_on="htf_close_time",
            direction="backward",
        )

        merged = merged.drop(columns=["htf_close_time"], errors="ignore")
        return merged

    @classmethod
    def align_multi_timeframes(
        cls,
        ltf_df: pd.DataFrame,
        htf_dict: dict[str, pd.DataFrame],
        feature_cols_dict: dict[str, list[str]] | None = None,
    ) -> pd.DataFrame:
        """Aligns multiple higher timeframe feature sets (e.g. H1, H4, D1) into an LTF dataframe."""
        if ltf_df.empty:
            return pd.DataFrame()

        result = ltf_df.copy()
        for htf_name, htf_df in htf_dict.items():
            f_cols = feature_cols_dict.get(htf_name) if feature_cols_dict else None
            result = cls.align_higher_timeframe(
                ltf_df=result,
                htf_df=htf_df,
                htf_name=htf_name,
                feature_cols=f_cols,
            )
        return result
