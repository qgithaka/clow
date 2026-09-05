"""Monte Carlo 1,000-run trade sequence simulation for Clow.

Performs bootstrap trade resamplings to calculate worst-case Max Drawdown,
Value-at-Risk (VaR), Conditional VaR (Expected Shortfall), and Risk of Ruin.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

logger = logging.getLogger("clow.validation.monte_carlo")


@dataclass
class MonteCarloSummary:
    """Statistical summary of 1,000 Monte Carlo bootstrap simulation paths."""

    num_simulations: int
    trades_per_simulation: int
    median_final_equity: float
    p05_final_equity: float
    p95_final_equity: float
    median_max_drawdown_pct: float
    p95_max_drawdown_pct: float
    p99_max_drawdown_pct: float
    var_95_pct: float
    cvar_95_pct: float
    risk_of_ruin_pct: float


class MonteCarloSimulator:
    """Bootstrap trade path simulator."""

    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """Calculates percentage maximum peak-to-trough drawdown."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / (running_max + 1e-8)
        return float(np.max(drawdowns))

    @classmethod
    def run_simulation(
        cls,
        trade_returns: Sequence[float],
        initial_capital: float = 10000.0,
        num_simulations: int = 1000,
        ruin_threshold_pct: float = 0.50,  # 50% drawdown considered ruin
        seed: int = 42,
    ) -> MonteCarloSummary:
        """Runs 1,000 bootstrap simulation paths.
        
        Args:
            trade_returns: Array of percentage/fractional returns per trade
            initial_capital: Starting equity
            num_simulations: Number of simulation iterations
            ruin_threshold_pct: Fraction loss threshold triggering ruin flag
            seed: Random seed for reproducibility
            
        Returns:
            MonteCarloSummary dataclass
        """
        n_trades = len(trade_returns)
        if n_trades == 0:
            return MonteCarloSummary(
                num_simulations=0,
                trades_per_simulation=0,
                median_final_equity=initial_capital,
                p05_final_equity=initial_capital,
                p95_final_equity=initial_capital,
                median_max_drawdown_pct=0.0,
                p95_max_drawdown_pct=0.0,
                p99_max_drawdown_pct=0.0,
                var_95_pct=0.0,
                cvar_95_pct=0.0,
                risk_of_ruin_pct=0.0,
            )

        rng = np.random.RandomState(seed)
        ret_arr = np.array(trade_returns, dtype=float)

        final_equities: List[float] = []
        max_drawdowns: List[float] = []
        ruin_count = 0

        for _ in range(num_simulations):
            # Bootstrap sample with replacement
            sampled_rets = rng.choice(ret_arr, size=n_trades, replace=True)
            # Compounding equity curve: Eq_t = Eq_0 * cumprod(1 + r_t)
            # Or additive for fixed fraction / pip sizing:
            equity_curve = initial_capital * np.cumprod(1.0 + np.clip(sampled_rets, -0.99, 10.0))
            equity_path = np.insert(equity_curve, 0, initial_capital)

            max_dd = cls.calculate_max_drawdown(equity_path)
            max_drawdowns.append(max_dd)
            final_equities.append(float(equity_path[-1]))

            if max_dd >= ruin_threshold_pct:
                ruin_count += 1

        eq_arr = np.array(final_equities)
        dd_arr = np.array(max_drawdowns)

        p95_dd = float(np.percentile(dd_arr, 95))
        p99_dd = float(np.percentile(dd_arr, 99))
        var_95 = p95_dd
        cvar_95 = float(np.mean(dd_arr[dd_arr >= p95_dd]))

        summary = MonteCarloSummary(
            num_simulations=num_simulations,
            trades_per_simulation=n_trades,
            median_final_equity=float(np.median(eq_arr)),
            p05_final_equity=float(np.percentile(eq_arr, 5)),
            p95_final_equity=float(np.percentile(eq_arr, 95)),
            median_max_drawdown_pct=float(np.median(dd_arr)) * 100.0,
            p95_max_drawdown_pct=p95_dd * 100.0,
            p99_max_drawdown_pct=p99_dd * 100.0,
            var_95_pct=var_95 * 100.0,
            cvar_95_pct=cvar_95 * 100.0,
            risk_of_ruin_pct=(ruin_count / num_simulations) * 100.0,
        )

        logger.info(
            f"Monte Carlo ({num_simulations} paths) | "
            f"Median Equity: ${summary.median_final_equity:.2f} | "
            f"p95 Max DD: {summary.p95_max_drawdown_pct:.2f}% | "
            f"p99 Max DD: {summary.p99_max_drawdown_pct:.2f}% | "
            f"Ruin Risk: {summary.risk_of_ruin_pct:.2f}%"
        )
        return summary
