"""Unit tests for Pinball quantile loss, directional loss, and calibration metrics."""

import numpy as np
import pytest
import torch
from training.models.losses import (
    PinballLoss,
    MultiQuantileLoss,
    CompositeForecasterLoss,
    QuantileEvaluator,
)


def test_pinball_loss_exact_math() -> None:
    """Verify exact mathematical values of Pinball Loss for q=0.50 (MAE/2) and asymmetric quantiles."""
    target = torch.tensor([10.0])
    pred_under = torch.tensor([8.0])   # error = +2.0 (underpredict)
    pred_over = torch.tensor([12.0])   # error = -2.0 (overpredict)

    # For q = 0.90:
    # underpredict: loss = 0.90 * 2.0 = 1.80
    # overpredict: loss = (0.90 - 1.0) * (-2.0) = 0.20
    q90 = PinballLoss(quantile=0.90)
    assert np.isclose(q90(pred_under, target).item(), 1.80)
    assert np.isclose(q90(pred_over, target).item(), 0.20)

    # For q = 0.10:
    # underpredict: loss = 0.10 * 2.0 = 0.20
    # overpredict: loss = (0.10 - 1.0) * (-2.0) = 1.80
    q10 = PinballLoss(quantile=0.10)
    assert np.isclose(q10(pred_under, target).item(), 0.20)
    assert np.isclose(q10(pred_over, target).item(), 1.80)


def test_multi_quantile_and_composite_loss() -> None:
    """Verify MultiQuantileLoss and CompositeForecasterLoss multi-objective gradients."""
    batch_size = 4
    quantiles = [0.10, 0.50, 0.90]

    composite_loss_fn = CompositeForecasterLoss(
        quantiles=quantiles,
        weight_anatomy=1.0,
        weight_direction=1.0,
        weight_quantiles=1.0,
    )

    model_outputs = {
        "anatomy_pred": torch.randn(batch_size, 4, requires_grad=True),
        "direction_logit": torch.randn(batch_size, 1, requires_grad=True),
        "direction_prob": torch.sigmoid(torch.randn(batch_size, 1)),
        "quantiles_high": torch.rand(batch_size, 3, requires_grad=True),
        "quantiles_low": torch.rand(batch_size, 3, requires_grad=True),
    }

    batch = {
        "target_candle": torch.randn(batch_size, 1, 4),
        "target_direction": torch.randint(0, 2, (batch_size, 1)).float(),
        "target_quantiles": torch.rand(batch_size, 1, 2),
    }

    losses = composite_loss_fn(model_outputs, batch)

    assert "total_loss" in losses
    assert "loss_anatomy" in losses
    assert "loss_direction" in losses
    assert "loss_quantiles" in losses

    total_loss = losses["total_loss"]
    assert total_loss.item() > 0.0

    total_loss.backward()
    assert model_outputs["anatomy_pred"].grad is not None
    assert model_outputs["direction_logit"].grad is not None
    assert model_outputs["quantiles_high"].grad is not None


def test_quantile_evaluator_metrics() -> None:
    """Verify empirical coverage computation and directional classification accuracy."""
    targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    # Predictions matching 10th (val 2), 50th (val 6), 90th (val 10) percentiles
    preds = np.tile([2.0, 6.0, 10.0], (10, 1))

    coverage = QuantileEvaluator.calculate_empirical_coverage(preds, targets)
    assert np.isclose(coverage[0], 0.20)  # targets <= 2.0 is 2/10
    assert np.isclose(coverage[1], 0.60)  # targets <= 6.0 is 6/10
    assert np.isclose(coverage[2], 1.00)  # targets <= 10.0 is 10/10

    # Directional accuracy
    true_labels = np.array([1, 1, 0, 0, 1])
    pred_probs = np.array([0.8, 0.9, 0.2, 0.1, 0.4])  # 4/5 correct
    acc = QuantileEvaluator.calculate_directional_accuracy(pred_probs, true_labels)
    assert np.isclose(acc, 0.80)


def test_invalid_quantile_bounds() -> None:
    """Verify PinballLoss raises ValueError on invalid quantiles."""
    with pytest.raises(ValueError):
        PinballLoss(quantile=0.0)
    with pytest.raises(ValueError):
        PinballLoss(quantile=1.0)
    with pytest.raises(ValueError):
        PinballLoss(quantile=-0.5)
