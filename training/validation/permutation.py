"""Synthetic Noise Permutation Test for Proof Lab / Clow.

Proves whether predictive model performance is statistically significant
or if it collapses to 50% random chance under randomized noise permutations.
"""

from dataclasses import dataclass
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.models.forecaster import ClowForecaster
from training.models.losses import QuantileEvaluator

logger = logging.getLogger("clow.validation.permutation")


@dataclass
class PermutationTestResult:
    """Statistical summary of permutation test."""

    baseline_accuracy: float
    permuted_mean_accuracy: float
    permuted_std_accuracy: float
    p_value: float
    is_statistically_significant: bool
    num_permutations: int


class PermutationTester:
    """Executes target and feature permutation tests against predictive foundation models."""

    @staticmethod
    def run_directional_permutation_test(
        model: ClowForecaster,
        test_loader: DataLoader,
        num_permutations: int = 50,
        significance_level: float = 0.01,
        seed: int = 42,
    ) -> PermutationTestResult:
        """Runs target label permutation test to calculate empirical p-value.
        
        Under the null hypothesis (H0), the model predictions have no true relationship
        with targets. Permuting the targets randomizes any temporal dependency.
        """
        rng = np.random.RandomState(seed)
        model.eval()

        # 1. Collect all predictions and true targets
        all_pred_probs: List[float] = []
        all_true_targets: List[float] = []

        with torch.no_grad():
            for batch in test_loader:
                preds = model(batch["context"])
                probs = preds["direction_prob"].squeeze().cpu().numpy()
                targets = batch["target_direction"].squeeze().cpu().numpy()

                if np.ndim(probs) == 0:
                    all_pred_probs.append(float(probs))
                    all_true_targets.append(float(targets))
                else:
                    all_pred_probs.extend(probs.tolist())
                    all_true_targets.extend(targets.tolist())

        pred_arr = np.array(all_pred_probs)
        target_arr = np.array(all_true_targets)

        # Baseline accuracy on unpermuted data
        baseline_acc = QuantileEvaluator.calculate_directional_accuracy(pred_arr, target_arr)

        # 2. Permutation Trials
        permuted_accs: List[float] = []
        for _ in range(num_permutations):
            shuffled_targets = rng.permutation(target_arr)
            perm_acc = QuantileEvaluator.calculate_directional_accuracy(pred_arr, shuffled_targets)
            permuted_accs.append(perm_acc)

        perm_arr = np.array(permuted_accs)
        mean_perm_acc = float(np.mean(perm_arr))
        std_perm_acc = float(np.std(perm_arr))

        # Empirical p-value: fraction of permuted trials that scored >= baseline
        p_val = float(np.mean(perm_arr >= baseline_acc))
        is_significant = p_val < significance_level and baseline_acc > 0.50

        logger.info(
            f"Permutation Test ({num_permutations} trials) | "
            f"Baseline Acc: {baseline_acc * 100:.2f}% | "
            f"Null Acc: {mean_perm_acc * 100:.2f}% +- {std_perm_acc * 100:.2f}% | "
            f"p-value: {p_val:.4f} | Significant: {is_significant}"
        )

        return PermutationTestResult(
            baseline_accuracy=baseline_acc,
            permuted_mean_accuracy=mean_perm_acc,
            permuted_std_accuracy=std_perm_acc,
            p_value=p_val,
            is_statistically_significant=is_significant,
            num_permutations=num_permutations,
        )
