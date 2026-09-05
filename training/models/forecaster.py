"""Clow-Forecaster deep time-series foundation architecture.

Implements multi-task transformer backbone predicting candle anatomy,
directional probability, and probabilistic quantile excursions for upcoming candles.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence context."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class ClowForecaster(nn.Module):
    """Multi-task neural foundation model for next-candle quantitative forecasting."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_anatomy_targets: int = 4,
        quantiles: Sequence[float] = (0.10, 0.50, 0.90),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.quantiles = list(quantiles)
        self.num_quantiles = len(self.quantiles)
        self.num_anatomy_targets = num_anatomy_targets

        # Input projection & Positional Encoding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)

        # Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Temporal Pooling / Context Aggregation
        self.context_pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Head 1: Scale-free Candle Anatomy [body_ratio, upper_wick, lower_wick, range_to_atr]
        self.anatomy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_anatomy_targets),
        )

        # Head 2: Directional Probability (P_Bull vs P_Bear)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Head 3: Quantile Excursions (High and Low excursions at each quantile)
        # Outputs [B, 2 * num_quantiles] -> (q_high_10, q_high_50, q_high_90, q_low_10, q_low_50, q_low_90)
        self.quantile_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2 * self.num_quantiles),
            nn.Softplus(),  # Excursions are strictly non-negative
        )

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            context: [B, L, input_dim] historical stationary feature sequence.
            
        Returns:
            Dict containing:
            - 'anatomy_pred': [B, num_anatomy_targets]
            - 'direction_logit': [B, 1]
            - 'direction_prob': [B, 1] (in [0, 1])
            - 'quantiles_high': [B, num_quantiles]
            - 'quantiles_low': [B, num_quantiles]
        """
        # 1. Project and encode
        x = self.input_proj(context)
        x = self.pos_encoder(x)
        x = self.layer_norm_in(x)

        # 2. Transformer representations
        encoded = self.transformer_encoder(x)  # [B, L, d_model]

        # 3. Context aggregation (take representation at last time step t)
        last_step = encoded[:, -1, :]  # [B, d_model]
        context_vec = self.context_pooler(last_step)  # [B, d_model]

        # 4. Heads
        anatomy_pred = self.anatomy_head(context_vec)
        direction_logit = self.direction_head(context_vec)
        direction_prob = torch.sigmoid(direction_logit)

        quantiles_raw = self.quantile_head(context_vec)  # [B, 2 * num_quantiles]
        quantiles_high = quantiles_raw[:, : self.num_quantiles]
        quantiles_low = quantiles_raw[:, self.num_quantiles :]

        return {
            "anatomy_pred": anatomy_pred,
            "direction_logit": direction_logit,
            "direction_prob": direction_prob,
            "quantiles_high": quantiles_high,
            "quantiles_low": quantiles_low,
        }
