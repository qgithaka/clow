"""Loss functions and probabilistic evaluation metrics for Clow-Forecaster.

Implements multi-quantile pinball loss, directional binary cross-entropy,
anatomy geometry loss, and quantile calibration evaluation metrics.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PinballLoss(nn.Module):
    """Pinball / Quantile Loss for a specific quantile q in (0, 1).
    
    Formula: L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """

    def __init__(self, quantile: float, reduction: str = "mean") -> None:
        super().__init__()
        if not (0.0 < quantile < 1.0):
            raise ValueError(f"Quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - pred
        loss = torch.max((self.quantile - 1.0) * error, self.quantile * error)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiQuantileLoss(nn.Module):
    """Multi-quantile pinball loss across multiple quantiles (e.g. 10th, 50th, 90th)."""

    def __init__(
        self,
        quantiles: Sequence[float] = (0.10, 0.50, 0.90),
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.quantiles = list(quantiles)
        self.reduction = reduction
        self.pinball_losses = nn.ModuleList([PinballLoss(q, reduction=reduction) for q in self.quantiles])

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            preds: [B, num_quantiles]
            targets: [B] or [B, 1] (ground truth values to compare against each quantile prediction)
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        total_loss = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
        for i, pinball in enumerate(self.pinball_losses):
            pred_q = preds[:, i : i + 1]
            total_loss = total_loss + pinball(pred_q, targets)

        return total_loss / len(self.quantiles)


class CompositeForecasterLoss(nn.Module):
    """Multi-task composite loss combining Anatomy MSE, Directional BCE, and Quantile Pinball."""

    def __init__(
        self,
        quantiles: Sequence[float] = (0.10, 0.50, 0.90),
        weight_anatomy: float = 1.0,
        weight_direction: float = 1.0,
        weight_quantiles: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight_anatomy = weight_anatomy
        self.weight_direction = weight_direction
        self.weight_quantiles = weight_quantiles

        self.anatomy_loss = nn.HuberLoss(delta=1.0)
        self.direction_loss = nn.BCEWithLogitsLoss()
        self.quantile_loss = MultiQuantileLoss(quantiles=quantiles)

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Calculates multi-objective losses.
        
        Args:
            model_outputs: Dict from ClowForecaster
            batch: Dict from TimeSeriesSlidingWindowDataset
            
        Returns:
            Dict containing individual loss components and 'total_loss'.
        """
        # 1. Anatomy Loss
        target_anatomy = batch["target_candle"].squeeze(1) if batch["target_candle"].dim() == 3 else batch["target_candle"]
        pred_anatomy = model_outputs["anatomy_pred"]
        loss_anat = self.anatomy_loss(pred_anatomy, target_anatomy)

        # 2. Directional Loss
        target_dir = batch["target_direction"].squeeze(1) if batch["target_direction"].dim() == 2 else batch["target_direction"]
        target_dir = target_dir.unsqueeze(1) if target_dir.dim() == 1 else target_dir
        pred_dir_logits = model_outputs["direction_logit"]
        loss_dir = self.direction_loss(pred_dir_logits, target_dir)

        # 3. Quantiles Loss (High and Low excursions)
        # target_quantiles: [B, 2] -> col 0: high excursion, col 1: low excursion
        target_quantiles = batch["target_quantiles"].squeeze(1) if batch["target_quantiles"].dim() == 3 else batch["target_quantiles"]
        target_high = target_quantiles[:, 0]
        target_low = target_quantiles[:, 1]

        loss_q_high = self.quantile_loss(model_outputs["quantiles_high"], target_high)
        loss_q_low = self.quantile_loss(model_outputs["quantiles_low"], target_low)
        loss_quant = (loss_q_high + loss_q_low) / 2.0

        # Total Weighted Loss
        total_loss = (
            self.weight_anatomy * loss_anat
            + self.weight_direction * loss_dir
            + self.weight_quantiles * loss_quant
        )

        return {
            "total_loss": total_loss,
            "loss_anatomy": loss_anat,
            "loss_direction": loss_dir,
            "loss_quantiles": loss_quant,
        }


class QuantileEvaluator:
    """Evaluates empirical quantile coverage and calibration."""

    @staticmethod
    def calculate_empirical_coverage(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Calculates fraction of targets <= predicted quantile values.
        
        Args:
            preds: [N, num_quantiles]
            targets: [N]
            
        Returns:
            Array of shape [num_quantiles] with empirical coverage fraction in [0, 1].
        """
        if targets.ndim == 1:
            targets = targets[:, np.newaxis]
        # targets <= preds
        coverage = np.mean(targets <= preds, axis=0)
        return coverage

    @staticmethod
    def calculate_directional_accuracy(pred_probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
        """Calculates directional binary classification accuracy."""
        pred_labels = (pred_probs >= threshold).astype(float)
        targets_flat = targets.flatten()
        pred_flat = pred_labels.flatten()
        return float(np.mean(pred_flat == targets_flat))
