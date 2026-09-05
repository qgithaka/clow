"""Broker Friction and Spread Shock Simulator for Clow.

Tests strategy robustness and positive expectancy under 2x, 3x, and 5x spread widening
and execution latency slippage shocks.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("clow.validation.stress_test")


@dataclass
class SpreadShockScenario:
    """Stress testing scenario results."""

    multiplier: float
    effective_spread_pips: float
    slippage_pips: float
    win_rate: float
    profit_factor: float
    net_profit_pips: float
    expectancy_per_trade: float
    is_profitable: bool


class SpreadShockSimulator:
    """Simulates severe broker spread widening and latency execution slippage."""

    def __init__(
        self,
        base_spread_pips: float = 0.00015,  # 1.5 pips default
        shock_multipliers: Sequence[float] = (1.0, 2.0, 3.0, 5.0),
        slippage_ms: float = 150.0,
        pip_slippage_per_100ms: float = 0.00005,  # 0.5 pip per 100ms latency
    ) -> None:
        self.base_spread_pips = base_spread_pips
        self.shock_multipliers = list(shock_multipliers)
        self.slippage_pips = (slippage_ms / 100.0) * pip_slippage_per_100ms

    def stress_test_trades(
        self,
        gross_returns_pips: Sequence[float],
        trade_sides: Sequence[int],
    ) -> dict[str, SpreadShockScenario]:
        """Evaluates trade returns under multiple friction shock scenarios.
        
        Args:
            gross_returns_pips: Array of gross price moves (Exit - Entry) * side
            trade_sides: +1 or -1
            
        Returns:
            Dict mapping scenario name to SpreadShockScenario.
        """
        results: dict[str, SpreadShockScenario] = {}
        n_trades = len(gross_returns_pips)
        if n_trades == 0:
            return results

        gross_arr = np.array(gross_returns_pips, dtype=float)

        for mult in self.shock_multipliers:
            eff_spread = self.base_spread_pips * mult
            total_friction = eff_spread + self.slippage_pips

            net_rets = gross_arr - total_friction
            wins = net_rets[net_rets > 0]
            losses = net_rets[net_rets < 0]

            win_rate = float(len(wins) / n_trades)
            gross_win_sum = float(np.sum(wins))
            gross_loss_sum = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-8

            profit_factor = gross_win_sum / gross_loss_sum if gross_loss_sum > 0 else float("inf")
            net_profit = float(np.sum(net_rets))
            expectancy = float(np.mean(net_rets))
            is_profitable = net_profit > 0.0 and expectancy > 0.0

            scenario_name = f"{mult:.1f}x_spread"
            results[scenario_name] = SpreadShockScenario(
                multiplier=mult,
                effective_spread_pips=eff_spread,
                slippage_pips=self.slippage_pips,
                win_rate=win_rate,
                profit_factor=profit_factor,
                net_profit_pips=net_profit,
                expectancy_per_trade=expectancy,
                is_profitable=is_profitable,
            )

            logger.info(
                f"Spread Shock {scenario_name} | Friction: {total_friction * 10000:.1f} pips | "
                f"Win Rate: {win_rate * 100:.1f}% | PF: {profit_factor:.2f} | "
                f"EV: {expectancy * 10000:.2f} pips | Profitable: {is_profitable}"
            )

        return results
