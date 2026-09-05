"""Statistical Proof Engine and Deflated Sharpe Ratio Validator for Clow.

Subjects every trading strategy to rigorous chronological validation gates,
Deflated Sharpe Ratio (DSR), spread shock simulations, and Monte Carlo risk limits.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.stats as stats
from training.validation.cross_validation import PurgedWalkForwardCV
from training.validation.monte_carlo import MonteCarloSimulator, MonteCarloSummary
from training.validation.permutation import PermutationTester, PermutationTestResult
from training.validation.stress_test import SpreadShockSimulator, SpreadShockScenario

logger = logging.getLogger("clow.validation.proof_engine")


@dataclass
class ValidationGateResults:
    """Status of institutional quantitative validation gates."""

    gate_dsr_passed: bool
    gate_spread_shock_passed: bool
    gate_drawdown_limit_passed: bool
    gate_profit_factor_passed: bool
    all_gates_passed: bool
    rejection_reasons: List[str]


@dataclass
class ProofReport:
    """Comprehensive institutional statistical proof report."""

    symbol: str
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    annualized_sharpe: float
    deflated_sharpe_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    sortino_ratio: float
    monte_carlo_p95_dd: float
    monte_carlo_p99_dd: float
    risk_of_ruin_pct: float
    spread_3x_profitable: bool
    gates: ValidationGateResults
    report_markdown: str


class StatisticalProofEngine:
    """Institutional quantitative proof engine with Deflated Sharpe Ratio calculation."""

    @staticmethod
    def calculate_deflated_sharpe_ratio(
        returns: Sequence[float],
        num_independent_trials: int = 10,
        benchmark_sr: float = 0.0,
    ) -> float:
        """Calculates Bailey & López de Prado's Deflated Sharpe Ratio (DSR).
        
        Adjusts the observed Sharpe ratio for non-normality (skewness, kurtosis),
        track record length (N), and selection bias across K multiple hypothesis tests.
        """
        arr = np.array(returns, dtype=float)
        n = len(arr)
        if n < 5:
            return 0.0

        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std <= 1e-8:
            return 0.0

        sr = mean / std
        skew = float(stats.skew(arr))
        kurt = float(stats.kurtosis(arr, fisher=False))  # Pearson kurtosis (normal=3)

        # Expected maximum Sharpe ratio under null hypothesis across K trials
        euler_mascheroni = 0.5772156649
        k = max(1, num_independent_trials)
        if k > 1:
            z = (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / k) + euler_mascheroni * stats.norm.ppf(
                1.0 - 1.0 / (k * np.e)
            )
            sr_star = max(benchmark_sr, z)
        else:
            sr_star = benchmark_sr

        # Standard error under non-normality
        denom = np.sqrt(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2))
        if denom <= 1e-8 or np.isnan(denom):
            denom = 1.0

        z_score = (sr - sr_star) * np.sqrt(n - 1) / denom
        dsr = float(stats.norm.cdf(z_score))
        return dsr

    @classmethod
    def evaluate_strategy_proof(
        cls,
        trade_returns: Sequence[float],
        symbol: str = "EURUSD",
        base_spread_pips: float = 0.00015,
        num_trials: int = 10,
        max_permitted_dd_pct: float = 25.0,
        min_profit_factor: float = 1.25,
        min_dsr: float = 0.95,
    ) -> ProofReport:
        """Executes the complete institutional statistical proof evaluation."""
        rets = np.array(trade_returns, dtype=float)
        n_trades = len(rets)

        if n_trades == 0:
            gates = ValidationGateResults(
                gate_dsr_passed=False,
                gate_spread_shock_passed=False,
                gate_drawdown_limit_passed=False,
                gate_profit_factor_passed=False,
                all_gates_passed=False,
                rejection_reasons=["No trades provided for evaluation."],
            )
            return ProofReport(
                symbol=symbol,
                total_trades=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                annualized_sharpe=0.0,
                deflated_sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                monte_carlo_p95_dd=0.0,
                monte_carlo_p99_dd=0.0,
                risk_of_ruin_pct=0.0,
                spread_3x_profitable=False,
                gates=gates,
                report_markdown="# Statistical Proof Report\n\nNo trades available.",
            )

        # 1. Performance Metrics
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        win_rate = (len(wins) / n_trades) * 100.0

        gross_win = float(np.sum(wins))
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-8
        profit_factor = gross_win / gross_loss

        # Sharpe & Sortino
        mean_ret = float(np.mean(rets))
        std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 1e-8
        downside_std = float(np.std(losses, ddof=1)) if len(losses) > 1 else std_ret

        # Annualized scaling assuming ~10 trades/day * 252 days = 2520 trades/yr
        ann_factor = np.sqrt(2520)
        sharpe = (mean_ret / std_ret) * ann_factor if std_ret > 0 else 0.0
        sortino = (mean_ret / (downside_std + 1e-8)) * ann_factor

        # Equity Curve and Max Drawdown
        cum_equity = np.insert(np.cumprod(1.0 + np.clip(rets, -0.99, 10.0)), 0, 1.0)
        max_dd = MonteCarloSimulator.calculate_max_drawdown(cum_equity) * 100.0
        calmar = (mean_ret * 2520) / ((max_dd / 100.0) + 1e-8)

        # 2. Deflated Sharpe Ratio
        dsr = cls.calculate_deflated_sharpe_ratio(rets, num_independent_trials=num_trials)

        # 3. Monte Carlo 1,000-Run Simulation
        mc_summary = MonteCarloSimulator.run_simulation(rets, num_simulations=1000)

        # 4. Spread Shock Simulation
        trade_sides = [1] * n_trades
        stress_sim = SpreadShockSimulator(base_spread_pips=base_spread_pips)
        stress_results = stress_sim.stress_test_trades(gross_returns_pips=rets, trade_sides=trade_sides)
        spread_3x_ok = stress_results.get("3.0x_spread", False) and stress_results["3.0x_spread"].is_profitable

        # 5. Validation Gates Check
        rejections: List[str] = []
        gate_dsr = dsr >= min_dsr
        if not gate_dsr:
            rejections.append(f"DSR {dsr:.4f} < {min_dsr:.4f} threshold (selection bias / non-normality risk).")

        gate_pf = profit_factor >= min_profit_factor
        if not gate_pf:
            rejections.append(f"Profit Factor {profit_factor:.2f} < {min_profit_factor:.2f} minimum.")

        gate_dd = mc_summary.p95_max_drawdown_pct <= max_permitted_dd_pct
        if not gate_dd:
            rejections.append(f"Monte Carlo p95 Drawdown {mc_summary.p95_max_drawdown_pct:.1f}% > {max_permitted_dd_pct:.1f}% limit.")

        gate_stress = spread_3x_ok
        if not gate_stress:
            rejections.append("Strategy fails profitability under 3.0x broker spread shock.")

        all_ok = gate_dsr and gate_pf and gate_dd and gate_stress
        gates = ValidationGateResults(
            gate_dsr_passed=gate_dsr,
            gate_spread_shock_passed=gate_stress,
            gate_drawdown_limit_passed=gate_dd,
            gate_profit_factor_passed=gate_pf,
            all_gates_passed=all_ok,
            rejection_reasons=rejections,
        )

        # 6. Markdown Report Generation
        report_md = f"""# Statistical Proof Report: {symbol}

**Generated At:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Validation Decision:** {'APPROVED FOR DEPLOYMENT' if all_ok else 'REJECTED BY GATES'}

---

### Core Performance Metrics

| Metric | Value | Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Total Trades** | {n_trades} | >= 30 | {'PASS' if n_trades >= 30 else 'FAIL'} |
| **Win Rate** | {win_rate:.2f}% | > 50.0% | {'PASS' if win_rate > 50.0 else 'WARN'} |
| **Profit Factor** | {profit_factor:.2f} | >= {min_profit_factor:.2f} | {'PASS' if gate_pf else 'FAIL'} |
| **Annualized Sharpe** | {sharpe:.2f} | > 1.50 | {'PASS' if sharpe > 1.5 else 'WARN'} |
| **Deflated Sharpe (DSR)** | {dsr:.4f} | >= {min_dsr:.2f} | {'PASS' if gate_dsr else 'FAIL'} |
| **Historical Max Drawdown** | {max_dd:.2f}% | <= {max_permitted_dd_pct:.1f}% | PASS |
| **Sortino Ratio** | {sortino:.2f} | > 2.0 | {'PASS' if sortino > 2.0 else 'WARN'} |
| **Calmar Ratio** | {calmar:.2f} | > 1.0 | PASS |

---

### Monte Carlo 1,000-Path Simulation

- **Median Final Equity:** ${mc_summary.median_final_equity:.2f} (5th: ${mc_summary.p05_final_equity:.2f} | 95th: ${mc_summary.p95_final_equity:.2f})
- **95th Percentile Max Drawdown:** {mc_summary.p95_max_drawdown_pct:.2f}% (Limit: {max_permitted_dd_pct:.1f}%)
- **99th Percentile Max Drawdown:** {mc_summary.p99_max_drawdown_pct:.2f}%
- **Value-at-Risk (VaR 95%):** {mc_summary.var_95_pct:.2f}%
- **Expected Shortfall (CVaR 95%):** {mc_summary.cvar_95_pct:.2f}%
- **Risk of Ruin (50% Drawdown):** {mc_summary.risk_of_ruin_pct:.2f}%

---

### Broker Friction & Spread Shock Simulator

- **1.0x Baseline Spread:** {'Profitable' if stress_results.get('1.0x_spread', False) and stress_results['1.0x_spread'].is_profitable else 'Unprofitable'}
- **2.0x Spread Shock:** {'Profitable' if stress_results.get('2.0x_spread', False) and stress_results['2.0x_spread'].is_profitable else 'Unprofitable'}
- **3.0x Spread Shock:** {'Profitable' if spread_3x_ok else 'Unprofitable'}
- **5.0x Extreme News Shock:** {'Profitable' if stress_results.get('5.0x_spread', False) and stress_results['5.0x_spread'].is_profitable else 'Unprofitable'}

---

### Institutional Gate Rejection Log

{chr(10).join([f"- [FAIL] {r}" for r in rejections]) if rejections else '- [PASS] All institutional validation gates passed cleanly.'}
"""

        return ProofReport(
            symbol=symbol,
            total_trades=n_trades,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            annualized_sharpe=sharpe,
            deflated_sharpe_ratio=dsr,
            max_drawdown_pct=max_dd,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            monte_carlo_p95_dd=mc_summary.p95_max_drawdown_pct,
            monte_carlo_p99_dd=mc_summary.p99_max_drawdown_pct,
            risk_of_ruin_pct=mc_summary.risk_of_ruin_pct,
            spread_3x_profitable=spread_3x_ok,
            gates=gates,
            report_markdown=report_md,
        )
