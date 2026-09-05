"""Triple-Barrier labeling engine for quantitative trading strategy research.

Implements Marcos López de Prado's Triple-Barrier method with Take-Profit,
Stop-Loss, and Vertical Time Expiration horizons including broker spread friction.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("clow.models.triple_barrier")


class BarrierOutcome(str, Enum):
    """Outcome of a trade across the triple barriers."""

    HIT_TP = "HIT_TP"  # Upper barrier / profit target hit
    HIT_SL = "HIT_SL"  # Lower barrier / stop loss hit
    TIMEOUT = "TIMEOUT"  # Vertical time barrier expired


@dataclass
class BarrierResult:
    """Detailed result of a single triple-barrier evaluation."""

    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    outcome: BarrierOutcome
    holding_period_bars: int
    gross_return: float
    net_return: float
    tp_price: float
    sl_price: float


class TripleBarrierEngine:
    """Evaluates Take-Profit, Stop-Loss, and Time Expiration barriers for historical bars."""

    @staticmethod
    def evaluate_barriers(
        df: pd.DataFrame,
        entry_idx: int,
        side: int,  # +1 for BUY, -1 for SELL
        entry_price: float,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.0,
        horizon_bars: int = 12,
        atr_col: str = "atr",
        spread_pips: float = 0.00015,  # 1.5 pips default spread friction
    ) -> BarrierResult:
        """Evaluates triple barrier outcome for a single trade entry.
        
        Args:
            df: OHLCV DataFrame
            entry_idx: Bar index where entry occurs
            side: +1 (BUY) or -1 (SELL)
            entry_price: Actual execution entry price
            tp_atr_mult: Take-profit distance in ATR multiples
            sl_atr_mult: Stop-loss distance in ATR multiples
            horizon_bars: Max bars before vertical timeout
            atr_col: Column name containing ATR values
            spread_pips: Spread friction deducted on round-trip
            
        Returns:
            BarrierResult object
        """
        n_rows = len(df)
        if entry_idx >= n_rows - 1:
            return BarrierResult(
                entry_idx=entry_idx,
                exit_idx=entry_idx,
                entry_price=entry_price,
                exit_price=entry_price,
                outcome=BarrierOutcome.TIMEOUT,
                holding_period_bars=0,
                gross_return=0.0,
                net_return=-spread_pips,
                tp_price=entry_price,
                sl_price=entry_price,
            )

        # Get ATR at entry
        atr = df[atr_col].iloc[entry_idx] if atr_col in df.columns else (entry_price * 0.001)
        if np.isnan(atr) or atr <= 0:
            atr = entry_price * 0.001

        # Calculate price levels
        if side > 0:  # BUY
            tp_price = entry_price + (tp_atr_mult * atr)
            sl_price = entry_price - (sl_atr_mult * atr)
        else:  # SELL
            tp_price = entry_price - (tp_atr_mult * atr)
            sl_price = entry_price + (sl_atr_mult * atr)

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        max_search = min(n_rows, entry_idx + 1 + horizon_bars)

        for curr_idx in range(entry_idx + 1, max_search):
            curr_high = highs[curr_idx]
            curr_low = lows[curr_idx]
            holding_bars = curr_idx - entry_idx

            if side > 0:  # BUY
                hit_tp = curr_high >= tp_price
                hit_sl = curr_low <= sl_price

                if hit_sl and hit_tp:
                    # Conservative assumption: Stop Loss was hit first
                    gross_ret = (sl_price - entry_price) / entry_price
                    return BarrierResult(
                        entry_idx=entry_idx,
                        exit_idx=curr_idx,
                        entry_price=entry_price,
                        exit_price=sl_price,
                        outcome=BarrierOutcome.HIT_SL,
                        holding_period_bars=holding_bars,
                        gross_return=gross_ret,
                        net_return=gross_ret - (spread_pips / entry_price),
                        tp_price=tp_price,
                        sl_price=sl_price,
                    )
                elif hit_sl:
                    gross_ret = (sl_price - entry_price) / entry_price
                    return BarrierResult(
                        entry_idx=entry_idx,
                        exit_idx=curr_idx,
                        entry_price=entry_price,
                        exit_price=sl_price,
                        outcome=BarrierOutcome.HIT_SL,
                        holding_period_bars=holding_bars,
                        gross_return=gross_ret,
                        net_return=gross_ret - (spread_pips / entry_price),
                        tp_price=tp_price,
                        sl_price=sl_price,
                    )
                elif hit_tp:
                    gross_ret = (tp_price - entry_price) / entry_price
                    return BarrierResult(
                        entry_idx=entry_idx,
                        exit_idx=curr_idx,
                        entry_price=entry_price,
                        exit_price=tp_price,
                        outcome=BarrierOutcome.HIT_TP,
                        holding_period_bars=holding_bars,
                        gross_return=gross_ret,
                        net_return=gross_ret - (spread_pips / entry_price),
                        tp_price=tp_price,
                        sl_price=sl_price,
                    )
            else:  # SELL
                hit_tp = curr_low <= tp_price
                hit_sl = curr_high >= sl_price

                if hit_sl and hit_tp:
                    gross_ret = (entry_price - sl_price) / entry_price
                    return BarrierResult(
                        entry_idx=entry_idx,
                        exit_idx=curr_idx,
                        entry_price=entry_price,
                        exit_price=sl_price,
                        outcome=BarrierOutcome.HIT_SL,
                        holding_period_bars=holding_bars,
                        gross_return=gross_ret,
                        net_return=gross_ret - (spread_pips / entry_price),
                        tp_price=tp_price,
                        sl_price=sl_price,
                    )
                elif hit_sl:
                    gross_ret = (entry_price - sl_price) / entry_price
                    return BarrierResult(
                        entry_idx=entry_idx,
                        exit_idx=curr_idx,
                        entry_price=entry_price,
                        exit_price=sl_price,
                        outcome=BarrierOutcome.HIT_SL,
                        holding_period_bars=holding_bars,
                        gross_return=gross_ret,
                        net_return=gross_ret - (spread_pips / entry_price),
                        tp_price=tp_price,
                        sl_price=sl_price,
                    )
                elif hit_tp:
                    gross_ret = (entry_price - tp_price) / entry_price
                    return BarrierResult(
                        entry_idx=entry_idx,
                        exit_idx=curr_idx,
                        entry_price=entry_price,
                        exit_price=tp_price,
                        outcome=BarrierOutcome.HIT_TP,
                        holding_period_bars=holding_bars,
                        gross_return=gross_ret,
                        net_return=gross_ret - (spread_pips / entry_price),
                        tp_price=tp_price,
                        sl_price=sl_price,
                    )

        # Reached time horizon without hitting TP or SL -> Vertical Timeout
        exit_idx = max_search - 1
        exit_price = closes[exit_idx]
        holding_bars = exit_idx - entry_idx
        gross_ret = (exit_price - entry_price) / entry_price if side > 0 else (entry_price - exit_price) / entry_price

        return BarrierResult(
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            entry_price=entry_price,
            exit_price=exit_price,
            outcome=BarrierOutcome.TIMEOUT,
            holding_period_bars=holding_bars,
            gross_return=gross_ret,
            net_return=gross_ret - (spread_pips / entry_price),
            tp_price=tp_price,
            sl_price=sl_price,
        )

    @classmethod
    def label_dataset(
        cls,
        df: pd.DataFrame,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.0,
        horizon_bars: int = 12,
        spread_pips: float = 0.00015,
    ) -> pd.DataFrame:
        """Labels an entire historical dataset with both Buy and Sell barrier outcomes."""
        if df.empty:
            return pd.DataFrame()

        res = df.copy()
        n = len(res)

        buy_outcomes: List[str] = []
        buy_net_rets: List[float] = []
        sell_outcomes: List[str] = []
        sell_net_rets: List[float] = []

        closes = res["close"].values

        for i in range(n):
            entry_p = closes[i]

            # Long trade evaluation
            buy_res = cls.evaluate_barriers(
                df=res,
                entry_idx=i,
                side=+1,
                entry_price=entry_p,
                tp_atr_mult=tp_atr_mult,
                sl_atr_mult=sl_atr_mult,
                horizon_bars=horizon_bars,
                spread_pips=spread_pips,
            )
            buy_outcomes.append(buy_res.outcome.value)
            buy_net_rets.append(buy_res.net_return)

            # Short trade evaluation
            sell_res = cls.evaluate_barriers(
                df=res,
                entry_idx=i,
                side=-1,
                entry_price=entry_p,
                tp_atr_mult=tp_atr_mult,
                sl_atr_mult=sl_atr_mult,
                horizon_bars=horizon_bars,
                spread_pips=spread_pips,
            )
            sell_outcomes.append(sell_res.outcome.value)
            sell_net_rets.append(sell_res.net_return)

        res["tb_buy_outcome"] = buy_outcomes
        res["tb_buy_net_ret"] = buy_net_rets
        res["tb_sell_outcome"] = sell_outcomes
        res["tb_sell_net_ret"] = sell_net_rets

        return res
