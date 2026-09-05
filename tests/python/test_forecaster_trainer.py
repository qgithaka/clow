"""Unit tests for ForecasterTrainer training, validation, and checkpointing."""

import os
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from training.data.features import FeatureEngineer
from training.models.dataset import TimeSeriesSlidingWindowDataset
from training.models.forecaster import ClowForecaster
from training.models.trainer import ForecasterTrainer, TrainingConfig


@pytest.fixture
def clean_checkpoint_dir(tmp_path: str) -> str:
    ckpt_dir = str(tmp_path / "checkpoints")
    yield ckpt_dir
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_forecaster_trainer_fit_and_checkpoint(clean_checkpoint_dir: str) -> None:
    """Verify complete training loop, evaluation metrics, and checkpoint save/load."""
    # Generate synthetic training dataset
    dates = pd.date_range("2023-01-01", periods=150, freq="5min", tz="UTC")
    prices = 1.0850 + np.sin(np.linspace(0, 10, 150)) * 0.0050
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "open": prices,
        "high": prices + 0.0005,
        "low": prices - 0.0005,
        "close": prices + 0.0002,
        "volume": [100.0] * 150,
    })
    feat_df = FeatureEngineer.compute_all_features(df)

    train_df, val_df, _ = TimeSeriesSlidingWindowDataset.temporal_train_val_test_split(
        feat_df, train_ratio=0.70, val_ratio=0.30, test_ratio=0.0, purge_window=10
    )

    train_ds = TimeSeriesSlidingWindowDataset(train_df, context_length=16)
    val_ds = TimeSeriesSlidingWindowDataset(val_df, context_length=16)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = ClowForecaster(
        input_dim=len(train_ds.feature_cols),
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )

    config = TrainingConfig(
        learning_rate=1e-3,
        max_epochs=4,
        patience=3,
        checkpoint_dir=clean_checkpoint_dir,
    )

    trainer = ForecasterTrainer(model=model, config=config, feature_cols=train_ds.feature_cols)
    history = trainer.fit(train_loader, val_loader)

    # 1. Verify history keys and length
    assert len(history["train_loss"]) == 4
    assert len(history["val_loss"]) == 4
    assert len(history["val_dir_acc"]) == 4

    # 2. Verify checkpoint existence and reloading
    ckpt_file = os.path.join(clean_checkpoint_dir, "best_forecaster.pt")
    assert os.path.exists(ckpt_file)

    new_model = ClowForecaster(
        input_dim=len(train_ds.feature_cols),
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
    )
    new_trainer = ForecasterTrainer(model=new_model, config=config)
    new_trainer.load_checkpoint(ckpt_file)

    # Verify model outputs match
    sample_ctx = torch.randn(2, 16, len(train_ds.feature_cols))
    model.eval()
    new_model.eval()
    with torch.no_grad():
        out1 = model(sample_ctx)
        out2 = new_model(sample_ctx)
        np.testing.assert_allclose(out1["direction_prob"].numpy(), out2["direction_prob"].numpy(), atol=1e-5)


def test_empty_loader_handling() -> None:
    """Verify empty loaders do not crash evaluation."""
    model = ClowForecaster(input_dim=8, d_model=16, nhead=2, num_layers=1)
    trainer = ForecasterTrainer(model=model)
    empty_loader = DataLoader([])
    loss = trainer.train_epoch(empty_loader)
    assert loss == 0.0
    val_loss, val_acc, _ = trainer.evaluate(empty_loader)
    assert val_loss == 0.0 and val_acc == 0.0
