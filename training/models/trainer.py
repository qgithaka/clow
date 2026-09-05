"""Training, validation, early stopping, and checkpointing pipeline for Clow-Forecaster."""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.models.forecaster import ClowForecaster
from training.models.losses import CompositeForecasterLoss, QuantileEvaluator

logger = logging.getLogger("clow.models.trainer")


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for Forecaster model training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    patience: int = 5
    clip_grad_norm: float = 1.0
    device: str = "cpu"
    checkpoint_dir: str = "models/checkpoints"
    quantiles: Sequence[float] = (0.10, 0.50, 0.90)
    weight_anatomy: float = 1.0
    weight_direction: float = 1.0
    weight_quantiles: float = 1.0


class ForecasterTrainer:
    """Manages training, validation, early stopping, and checkpointing for ClowForecaster."""

    def __init__(
        self,
        model: ClowForecaster,
        config: TrainingConfig | None = None,
        feature_cols: Sequence[str] | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.feature_cols = list(feature_cols) if feature_cols else []

        self.loss_fn = CompositeForecasterLoss(
            quantiles=self.config.quantiles,
            weight_anatomy=self.config.weight_anatomy,
            weight_direction=self.config.weight_direction,
            weight_quantiles=self.config.weight_quantiles,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_dir_acc": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Trains model for one epoch and returns mean loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        if num_batches == 0:
            return 0.0

        for batch in train_loader:
            # Move tensors to target device
            batch_dev = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            self.optimizer.zero_grad()
            outputs = self.model(batch_dev["context"])
            losses = self.loss_fn(outputs, batch_dev)
            loss = losses["total_loss"]

            loss.backward()
            if self.config.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / num_batches

    def evaluate(self, val_loader: DataLoader) -> tuple[float, float, dict[str, float]]:
        """Evaluates model performance on validation / test set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        if num_batches == 0:
            return 0.0, 0.0, {}

        all_dir_probs: list[np.ndarray] = []
        all_dir_targets: list[np.ndarray] = []

        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(batch_dev["context"])
                losses = self.loss_fn(outputs, batch_dev)
                total_loss += losses["total_loss"].item()

                dir_probs = outputs["direction_prob"].cpu().numpy()
                dir_targets = batch_dev["target_direction"].cpu().numpy()
                all_dir_probs.append(dir_probs)
                all_dir_targets.append(dir_targets)

        mean_loss = total_loss / num_batches
        cat_probs = np.concatenate(all_dir_probs, axis=0)
        cat_targets = np.concatenate(all_dir_targets, axis=0)
        dir_acc = QuantileEvaluator.calculate_directional_accuracy(cat_probs, cat_targets)

        metrics = {
            "val_loss": mean_loss,
            "directional_accuracy": dir_acc,
        }
        return mean_loss, dir_acc, metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, list[float]]:
        """Runs complete training loop with validation, learning rate scheduling, and early stopping."""
        patience_counter = 0
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = Path(self.config.checkpoint_dir) / "best_forecaster.pt"

        logger.info(f"Starting Forecaster training for max {self.config.max_epochs} epochs...")

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc, _ = self.evaluate(val_loader)

            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dir_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch:02d}/{self.config.max_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dir Acc: {val_acc * 100:.2f}%"
            )

            # Checkpoint on best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(str(checkpoint_path))
                logger.info(f"Saved new best model checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (patience={self.config.patience}).")
                    break

        # Reload best weights
        if checkpoint_path.exists():
            self.load_checkpoint(str(checkpoint_path))

        return self.history

    def save_checkpoint(self, path: str) -> None:
        """Saves complete model checkpoint with state dict and metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "feature_cols": self.feature_cols,
            "input_dim": self.model.input_dim,
            "d_model": self.model.d_model,
            "nhead": self.model.nhead,
            "num_layers": self.model.num_layers,
            "dim_feedforward": self.model.dim_feedforward,
            "quantiles": self.model.quantiles,
            "num_anatomy_targets": self.model.num_anatomy_targets,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        """Loads model state from saved checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.best_val_loss = state.get("best_val_loss", self.best_val_loss)
        self.best_epoch = state.get("best_epoch", self.best_epoch)
        if "feature_cols" in state:
            self.feature_cols = state["feature_cols"]
