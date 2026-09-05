"""Inference engine and CPU latency benchmark for Clow-Forecaster."""

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
from training.models.forecaster import ClowForecaster

logger = logging.getLogger("clow.models.inference")


@dataclass
class NextCandleForecast:
    """Probabilistic and geometric next-candle forecast results."""

    direction_prob: float
    is_bullish: bool
    body_ratio: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    range_to_atr: float
    quantiles_high: List[float]
    quantiles_low: List[float]
    latency_ms: float


class ForecasterPredictor:
    """Real-time inference wrapper for next-candle foundation model."""

    def __init__(self, model: ClowForecaster, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "ForecasterPredictor":
        """Initializes predictor from a saved model checkpoint file."""
        state = torch.load(checkpoint_path, map_location=device)
        input_dim = state["input_dim"]
        d_model = state.get("d_model", 64)
        nhead = state.get("nhead", 4)
        num_layers = state.get("num_layers", 2)
        dim_feedforward = state.get("dim_feedforward", 128)
        quantiles = state.get("quantiles", (0.10, 0.50, 0.90))
        num_anatomy = state.get("num_anatomy_targets", 4)

        model = ClowForecaster(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            quantiles=quantiles,
            num_anatomy_targets=num_anatomy,
        )
        model.load_state_dict(state["model_state_dict"])
        return cls(model=model, device=device)

    def predict_next_candle(self, context: Union[np.ndarray, torch.Tensor]) -> NextCandleForecast:
        """Executes single-step real-time inference on a sliding context window.
        
        Args:
            context: [L, D] or [1, L, D] array/tensor of stationary historical context features.
            
        Returns:
            NextCandleForecast object.
        """
        start_time = time.perf_counter()

        if isinstance(context, np.ndarray):
            ctx_tensor = torch.from_numpy(context).float()
        else:
            ctx_tensor = context.float()

        if ctx_tensor.dim() == 2:
            ctx_tensor = ctx_tensor.unsqueeze(0)  # [1, L, D]

        ctx_tensor = ctx_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(ctx_tensor)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        dir_prob = float(outputs["direction_prob"][0, 0].item())
        is_bullish = dir_prob >= 0.50

        anat = outputs["anatomy_pred"][0].cpu().numpy()
        body_ratio = float(anat[0]) if len(anat) > 0 else 0.0
        upper_wick = float(anat[1]) if len(anat) > 1 else 0.0
        lower_wick = float(anat[2]) if len(anat) > 2 else 0.0
        range_atr = float(anat[3]) if len(anat) > 3 else 1.0

        q_high = outputs["quantiles_high"][0].cpu().numpy().tolist()
        q_low = outputs["quantiles_low"][0].cpu().numpy().tolist()

        return NextCandleForecast(
            direction_prob=dir_prob,
            is_bullish=is_bullish,
            body_ratio=body_ratio,
            upper_wick_ratio=upper_wick,
            lower_wick_ratio=lower_wick,
            range_to_atr=range_atr,
            quantiles_high=q_high,
            quantiles_low=q_low,
            latency_ms=elapsed_ms,
        )


class ForecasterInferenceBenchmark:
    """Measures single-step CPU inference latency for real-time live trading."""

    @staticmethod
    def benchmark_latency(
        predictor: ForecasterPredictor,
        context_length: int = 64,
        input_dim: int = 16,
        warmup_iters: int = 20,
        benchmark_iters: int = 100,
    ) -> Dict[str, float]:
        """Runs single-bar sliding window CPU latency benchmark."""
        sample_context = np.random.randn(context_length, input_dim).astype(np.float32)

        # 1. Warmup
        for _ in range(warmup_iters):
            _ = predictor.predict_next_candle(sample_context)

        # 2. Timing loops
        latencies_ms: List[float] = []
        for _ in range(benchmark_iters):
            forecast = predictor.predict_next_candle(sample_context)
            latencies_ms.append(forecast.latency_ms)

        arr = np.array(latencies_ms)
        summary = {
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "iterations": benchmark_iters,
        }

        logger.info(
            f"CPU Benchmark ({benchmark_iters} iters) | "
            f"Median: {summary['median_ms']:.2f}ms | "
            f"p95: {summary['p95_ms']:.2f}ms | "
            f"p99: {summary['p99_ms']:.2f}ms"
        )
        return summary
