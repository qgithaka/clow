"""Unit tests for Forecaster real-time inference and CPU latency benchmark."""

import os
import shutil
import numpy as np
import pytest
import torch
from training.models.forecaster import ClowForecaster
from training.models.inference import ForecasterPredictor, ForecasterInferenceBenchmark
from training.models.trainer import ForecasterTrainer, TrainingConfig


@pytest.fixture
def clean_ckpt_dir(tmp_path: str) -> str:
    path = str(tmp_path / "inference_ckpts")
    yield path
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def test_predictor_and_benchmark(clean_ckpt_dir: str) -> None:
    """Verify single-bar real-time predictor and CPU latency benchmarking."""
    input_dim = 16
    context_len = 32

    model = ClowForecaster(
        input_dim=input_dim,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        quantiles=(0.10, 0.50, 0.90),
    )

    trainer = ForecasterTrainer(model=model, config=TrainingConfig(checkpoint_dir=clean_ckpt_dir))
    ckpt_file = os.path.join(clean_ckpt_dir, "test_model.pt")
    trainer.save_checkpoint(ckpt_file)

    # 1. Initialize predictor from checkpoint
    predictor = ForecasterPredictor.from_checkpoint(ckpt_file, device="cpu")

    # 2. Single-step prediction from numpy array
    context_arr = np.random.randn(context_len, input_dim).astype(np.float32)
    forecast = predictor.predict_next_candle(context_arr)

    assert 0.0 <= forecast.direction_prob <= 1.0
    assert isinstance(forecast.is_bullish, bool)
    assert len(forecast.quantiles_high) == 3
    assert len(forecast.quantiles_low) == 3
    assert all(q >= 0.0 for q in forecast.quantiles_high)
    assert all(q >= 0.0 for q in forecast.quantiles_low)
    assert forecast.latency_ms >= 0.0

    # 3. CPU Latency Benchmark
    bench_results = ForecasterInferenceBenchmark.benchmark_latency(
        predictor=predictor,
        context_length=context_len,
        input_dim=input_dim,
        warmup_iters=5,
        benchmark_iters=20,
    )

    assert "median_ms" in bench_results
    assert "p95_ms" in bench_results
    assert "p99_ms" in bench_results
    assert bench_results["iterations"] == 20
    assert bench_results["median_ms"] > 0.0
    # Single-step CPU inference should be fast (< 100ms)
    assert bench_results["median_ms"] < 100.0


def test_predictor_from_tensor() -> None:
    """Verify prediction directly from PyTorch Tensor."""
    model = ClowForecaster(input_dim=8, d_model=16, nhead=2, num_layers=1)
    predictor = ForecasterPredictor(model=model, device="cpu")

    ctx_tensor = torch.randn(32, 8)
    forecast = predictor.predict_next_candle(ctx_tensor)
    assert 0.0 <= forecast.direction_prob <= 1.0
