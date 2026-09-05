"""End-to-End Pipeline connecting Model 1 Forecaster to Model 2 Tactical Order Policy."""

import logging

import numpy as np
import pandas as pd

from training.data.features import FeatureEngineer
from training.models.inference import ForecasterPredictor
from training.models.order_policy import TacticalOrderPolicy, TacticalOrderProposal

logger = logging.getLogger("clow.models.pipeline")


class TacticalForecasterPipeline:
    """End-to-end decision and execution pipeline uniting Model 1 and Model 2."""

    def __init__(
        self,
        predictor: ForecasterPredictor,
        policy: TacticalOrderPolicy | None = None,
        context_length: int = 64,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.predictor = predictor
        self.policy = policy or TacticalOrderPolicy()
        self.context_length = context_length
        self.feature_cols = feature_cols

    def process_bars(
        self,
        df: pd.DataFrame,
        symbol: str = "EURUSD",
    ) -> TacticalOrderProposal:
        """Processes historical bars through feature engineering, Model 1 forecast, and Model 2 order generation."""
        if len(df) < self.context_length:
            raise ValueError(
                f"Insufficient bars ({len(df)}) for pipeline context_length ({self.context_length})."
            )

        # 1. Feature Engineering
        feat_df = FeatureEngineer.compute_all_features(df)
        available_cols = set(feat_df.columns)

        if self.feature_cols:
            f_cols = [c for c in self.feature_cols if c in available_cols]
        else:
            # Fallback to standard features
            from training.models.dataset import TimeSeriesSlidingWindowDataset
            f_cols = [c for c in TimeSeriesSlidingWindowDataset.DEFAULT_FEATURE_COLS if c in available_cols]

        # Extract sliding context window (last context_length bars)
        context_window = feat_df[f_cols].iloc[-self.context_length :].values.astype(np.float32)

        # 2. Model 1 Prediction
        forecast = self.predictor.predict_next_candle(context_window)

        # 3. Model 2 Order Construction
        current_close = float(feat_df["close"].iloc[-1])
        current_atr = float(feat_df["atr"].iloc[-1]) if "atr" in feat_df.columns else (current_close * 0.001)
        current_spread = float(feat_df["spread"].iloc[-1]) if "spread" in feat_df.columns else None
        vol_expansion = float(feat_df["volatility_expansion_ratio"].iloc[-1]) if "volatility_expansion_ratio" in feat_df.columns else 1.0

        proposal = self.policy.generate_order_proposal(
            forecast=forecast,
            current_price=current_close,
            atr=current_atr,
            symbol=symbol,
            spread=current_spread,
            volatility_expansion=vol_expansion,
        )

        return proposal
