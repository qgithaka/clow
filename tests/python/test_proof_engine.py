"""Unit tests for Statistical Proof Engine, DSR, Monte Carlo, and stress testing."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.models.forecaster import ClowForecaster
from training.validation.monte_carlo import MonteCarloSimulator
from training.validation.permutation import PermutationTester
from training.validation.proof_engine import StatisticalProofEngine
from training.validation.stress_test import SpreadShockSimulator


def test_permutation_tester_on_synthetic_data() -> None:
    """Verify PermutationTester computes valid baseline and empirical p-values."""
    model = ClowForecaster(input_dim=8, d_model=16, nhead=2, num_layers=1)
    x = torch.randn(20, 16, 8)
    y = torch.randint(0, 2, (20, 1)).float()

    dataset = [{"context": x[i], "target_direction": y[i]} for i in range(20)]
    loader = DataLoader(dataset, batch_size=5)

    result = PermutationTester.run_directional_permutation_test(
        model=model,
        test_loader=loader,
        num_permutations=10,
    )

    assert 0.0 <= result.baseline_accuracy <= 1.0
    assert 0.0 <= result.permuted_mean_accuracy <= 1.0
    assert 0.0 <= result.p_value <= 1.0
    assert result.num_permutations == 10


def test_spread_shock_simulator() -> None:
    """Verify spread shock multiplier and slippage deductions."""
    # 10 winning trades of +20 pips, 5 losing trades of -10 pips
    gross_returns = [0.0020] * 10 + [-0.0010] * 5
    trade_sides = [1] * 15

    sim = SpreadShockSimulator(base_spread_pips=0.00015, shock_multipliers=(1.0, 3.0, 5.0))
    results = sim.stress_test_trades(gross_returns, trade_sides)

    assert "1.0x_spread" in results
    assert "3.0x_spread" in results
    assert "5.0x_spread" in results

    assert results["1.0x_spread"].is_profitable is True
    # Profit factor must decrease as spread multiplier increases
    assert results["1.0x_spread"].profit_factor > results["5.0x_spread"].profit_factor


def test_monte_carlo_drawdown_simulation() -> None:
    """Verify Monte Carlo 1,000-run simulation calculates realistic percentiles."""
    np.random.seed(42)
    # Positive expectancy trades: +2% with 60% probability, -1% with 40% probability
    returns = np.random.choice([0.02, -0.01], size=100, p=[0.60, 0.40])

    summary = MonteCarloSimulator.run_simulation(returns, num_simulations=200)

    assert summary.num_simulations == 200
    assert summary.median_final_equity > 10000.0  # Grew initial capital
    assert summary.p95_max_drawdown_pct >= summary.median_max_drawdown_pct
    assert summary.p99_max_drawdown_pct >= summary.p95_max_drawdown_pct
    assert summary.risk_of_ruin_pct <= 5.0


def test_statistical_proof_engine_and_dsr_gates() -> None:
    """Verify StatisticalProofEngine calculates DSR and approves/rejects strategies."""
    # 1. Clean profitable trade series
    np.random.seed(101)
    prof_returns = list(np.random.normal(0.0050, 0.0020, 100))  # Consistent positive returns

    proof_good = StatisticalProofEngine.evaluate_strategy_proof(
        trade_returns=prof_returns,
        symbol="EURUSD",
        min_dsr=0.50,
        min_profit_factor=1.20,
    )

    assert proof_good.symbol == "EURUSD"
    assert proof_good.profit_factor > 1.20
    assert proof_good.deflated_sharpe_ratio > 0.50
    assert proof_good.gates.all_gates_passed is True
    assert "APPROVED FOR DEPLOYMENT" in proof_good.report_markdown

    # 2. Losing / Overfitting trade series
    bad_returns = [-0.0050] * 50
    proof_bad = StatisticalProofEngine.evaluate_strategy_proof(
        trade_returns=bad_returns,
        symbol="GBPUSD",
    )
    assert proof_bad.gates.all_gates_passed is False
    assert len(proof_bad.gates.rejection_reasons) > 0
    assert "REJECTED BY GATES" in proof_bad.report_markdown

    # 3. Empty returns handling
    empty_proof = StatisticalProofEngine.evaluate_strategy_proof([])
    assert empty_proof.gates.all_gates_passed is False
