"""Unit tests for ClowForecaster neural foundation model architecture."""

import torch

from training.models.forecaster import ClowForecaster


def test_forecaster_forward_shapes_and_bounds() -> None:
    """Verify ClowForecaster output shapes, gradient flow, and probability bounds."""
    batch_size = 8
    seq_len = 32
    input_dim = 16
    quantiles = [0.10, 0.50, 0.90]

    model = ClowForecaster(
        input_dim=input_dim,
        d_model=32,
        nhead=2,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        num_anatomy_targets=4,
        quantiles=quantiles,
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    outputs = model(x)

    # 1. Output keys
    assert "anatomy_pred" in outputs
    assert "direction_logit" in outputs
    assert "direction_prob" in outputs
    assert "quantiles_high" in outputs
    assert "quantiles_low" in outputs

    # 2. Tensor shapes
    assert outputs["anatomy_pred"].shape == (batch_size, 4)
    assert outputs["direction_logit"].shape == (batch_size, 1)
    assert outputs["direction_prob"].shape == (batch_size, 1)
    assert outputs["quantiles_high"].shape == (batch_size, 3)
    assert outputs["quantiles_low"].shape == (batch_size, 3)

    # 3. Probability and non-negative quantile bounds
    assert (outputs["direction_prob"] >= 0.0).all()
    assert (outputs["direction_prob"] <= 1.0).all()
    assert (outputs["quantiles_high"] >= 0.0).all()
    assert (outputs["quantiles_low"] >= 0.0).all()

    # 4. Backward gradient flow
    loss = (
        outputs["anatomy_pred"].sum()
        + outputs["direction_logit"].sum()
        + outputs["quantiles_high"].sum()
        + outputs["quantiles_low"].sum()
    )
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
