"""Tactical Order Policy Engine (Model 2) for Clow.

Constructs optimal pending orders (BUY_LIMIT, SELL_LIMIT, dynamic Stop-Loss,
Take-Profit, and Expiration Horizons) with mathematical expectancy validation.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from training.models.inference import NextCandleForecast

logger = logging.getLogger("clow.models.order_policy")


class OrderType(str, Enum):
    """Institutional execution order types."""

    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"
    BUY_MARKET = "BUY_MARKET"
    SELL_MARKET = "SELL_MARKET"
    HOLD_NO_ACTION = "HOLD_NO_ACTION"


@dataclass
class TacticalOrderProposal:
    """Complete institutional pending order proposal from Model 2."""

    order_type: OrderType
    symbol: str
    current_price: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    expiration_bars: int
    risk_pips: float
    reward_pips: float
    risk_reward_ratio: float
    win_probability: float
    expected_value: float
    passed_expectancy_filter: bool
    rejection_reason: str | None = None


class TacticalOrderPolicy:
    """Constructs and validates tactical execution orders based on Model 1 forecasts."""

    def __init__(
        self,
        min_directional_confidence: float = 0.55,
        min_risk_reward_ratio: float = 1.20,
        min_expectancy_pips: float = 0.0,
        max_expiration_bars: int = 6,
        default_spread_pips: float = 0.00015,
        limit_entry_discount_atr: float = 0.30,
        sl_buffer_atr: float = 1.0,
        tp_target_atr: float = 2.0,
    ) -> None:
        self.min_directional_confidence = min_directional_confidence
        self.min_risk_reward_ratio = min_risk_reward_ratio
        self.min_expectancy_pips = min_expectancy_pips
        self.max_expiration_bars = max_expiration_bars
        self.default_spread_pips = default_spread_pips
        self.limit_entry_discount_atr = limit_entry_discount_atr
        self.sl_buffer_atr = sl_buffer_atr
        self.tp_target_atr = tp_target_atr

    def calculate_expiration_horizon(
        self,
        range_to_atr: float,
        volatility_expansion: float = 1.0,
    ) -> int:
        """Calculates dynamic order cancellation timeout.
        
        High volatility / expansion -> shorter horizon (faster fill or cancel).
        Low volatility / compression -> longer horizon.
        """
        if volatility_expansion > 1.5 or range_to_atr > 1.5:
            return max(2, self.max_expiration_bars // 2)
        elif volatility_expansion < 0.8:
            return self.max_expiration_bars
        return max(3, self.max_expiration_bars)

    def calculate_expectancy(
        self,
        win_prob: float,
        reward_amount: float,
        risk_amount: float,
        friction: float,
    ) -> float:
        """Calculates mathematical expected value (EV) with broker spread friction deducted."""
        loss_prob = 1.0 - win_prob
        ev = (win_prob * reward_amount) - (loss_prob * risk_amount) - friction
        return ev

    def generate_order_proposal(
        self,
        forecast: NextCandleForecast,
        current_price: float,
        atr: float,
        symbol: str = "EURUSD",
        spread: float | None = None,
        volatility_expansion: float = 1.0,
    ) -> TacticalOrderProposal:
        """Constructs an optimal pending order proposal with strict expectancy filtering."""
        spread_friction = spread if spread is not None else self.default_spread_pips
        prob_bull = forecast.direction_prob
        prob_bear = 1.0 - prob_bull

        # 1. Directional signal check
        if prob_bull >= self.min_directional_confidence:
            order_type = OrderType.BUY_LIMIT
            win_prob = prob_bull
            # Entry level: Limit discount below current close
            discount = self.limit_entry_discount_atr * atr
            entry_price = current_price - discount

            # Stop Loss & Take Profit
            sl_price = entry_price - (self.sl_buffer_atr * atr)
            # Use forecast quantile high if available, else standard TP ATR
            tp_dist = max(forecast.quantiles_high[1] * atr if len(forecast.quantiles_high) > 1 else 0.0, self.tp_target_atr * atr)
            tp_price = entry_price + tp_dist

            risk = entry_price - sl_price
            reward = tp_price - entry_price

        elif prob_bear >= self.min_directional_confidence:
            order_type = OrderType.SELL_LIMIT
            win_prob = prob_bear
            # Entry level: Limit premium above current close
            premium = self.limit_entry_discount_atr * atr
            entry_price = current_price + premium

            # Stop Loss & Take Profit
            sl_price = entry_price + (self.sl_buffer_atr * atr)
            tp_dist = max(forecast.quantiles_low[1] * atr if len(forecast.quantiles_low) > 1 else 0.0, self.tp_target_atr * atr)
            tp_price = entry_price - tp_dist

            risk = sl_price - entry_price
            reward = entry_price - tp_price

        else:
            return TacticalOrderProposal(
                order_type=OrderType.HOLD_NO_ACTION,
                symbol=symbol,
                current_price=current_price,
                entry_price=current_price,
                stop_loss_price=current_price,
                take_profit_price=current_price,
                expiration_bars=0,
                risk_pips=0.0,
                reward_pips=0.0,
                risk_reward_ratio=0.0,
                win_probability=max(prob_bull, prob_bear),
                expected_value=0.0,
                passed_expectancy_filter=False,
                rejection_reason="Directional confidence below threshold.",
            )

        # 2. Risk-Reward Ratio
        rrr = reward / (risk + 1e-8)

        # 3. Dynamic Expiration Horizon
        exp_bars = self.calculate_expiration_horizon(
            range_to_atr=forecast.range_to_atr,
            volatility_expansion=volatility_expansion,
        )

        # 4. Mathematical Expectancy
        ev = self.calculate_expectancy(
            win_prob=win_prob,
            reward_amount=reward,
            risk_amount=risk,
            friction=spread_friction,
        )

        # 5. Gate Validation
        passed = True
        rejection_reason = None

        if rrr < self.min_risk_reward_ratio:
            passed = False
            rejection_reason = f"Risk-Reward Ratio {rrr:.2f} < minimum {self.min_risk_reward_ratio:.2f}"
        elif ev <= self.min_expectancy_pips:
            passed = False
            rejection_reason = f"Expected Value {ev:.5f} <= minimum {self.min_expectancy_pips:.5f}"

        return TacticalOrderProposal(
            order_type=order_type,
            symbol=symbol,
            current_price=current_price,
            entry_price=entry_price,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            expiration_bars=exp_bars,
            risk_pips=risk,
            reward_pips=reward,
            risk_reward_ratio=rrr,
            win_probability=win_prob,
            expected_value=ev,
            passed_expectancy_filter=passed,
            rejection_reason=rejection_reason,
        )
