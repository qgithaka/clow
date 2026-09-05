"""Unit tests for TacticalOrderPolicy, dynamic expiration, and expectancy gating."""

from training.models.inference import NextCandleForecast
from training.models.order_policy import OrderType, TacticalOrderPolicy


def test_buy_limit_order_generation() -> None:
    """Verify BUY_LIMIT generation when directional confidence is high."""
    policy = TacticalOrderPolicy(
        min_directional_confidence=0.55,
        min_risk_reward_ratio=1.5,
        min_expectancy_pips=0.0,
        limit_entry_discount_atr=0.25,
        sl_buffer_atr=1.0,
        tp_target_atr=2.0,
    )

    forecast_bull = NextCandleForecast(
        direction_prob=0.70,
        is_bullish=True,
        body_ratio=0.6,
        upper_wick_ratio=0.2,
        lower_wick_ratio=0.2,
        range_to_atr=1.1,
        quantiles_high=[0.5, 2.0, 3.5],
        quantiles_low=[0.2, 0.5, 1.0],
        latency_ms=1.5,
    )

    proposal = policy.generate_order_proposal(
        forecast=forecast_bull,
        current_price=1.0850,
        atr=0.0010,
        symbol="EURUSD",
        spread=0.00010,
    )

    assert proposal.order_type == OrderType.BUY_LIMIT
    assert proposal.entry_price < 1.0850  # Discounted entry
    assert proposal.stop_loss_price < proposal.entry_price
    assert proposal.take_profit_price > proposal.entry_price
    assert proposal.risk_reward_ratio >= 1.5
    assert proposal.expected_value > 0.0
    assert proposal.passed_expectancy_filter is True
    assert proposal.rejection_reason is None


def test_sell_limit_order_generation() -> None:
    """Verify SELL_LIMIT generation when bearish confidence is high."""
    policy = TacticalOrderPolicy(
        min_directional_confidence=0.55,
        min_risk_reward_ratio=1.5,
    )

    forecast_bear = NextCandleForecast(
        direction_prob=0.20,  # 80% Bearish
        is_bullish=False,
        body_ratio=-0.7,
        upper_wick_ratio=0.1,
        lower_wick_ratio=0.2,
        range_to_atr=1.0,
        quantiles_high=[0.2, 0.5, 0.8],
        quantiles_low=[0.5, 2.0, 3.5],
        latency_ms=1.2,
    )

    proposal = policy.generate_order_proposal(
        forecast=forecast_bear,
        current_price=1.0850,
        atr=0.0010,
        symbol="EURUSD",
        spread=0.00010,
    )

    assert proposal.order_type == OrderType.SELL_LIMIT
    assert proposal.entry_price > 1.0850  # Premium entry
    assert proposal.stop_loss_price > proposal.entry_price
    assert proposal.take_profit_price < proposal.entry_price
    assert proposal.passed_expectancy_filter is True


def test_low_confidence_and_expectancy_rejections() -> None:
    """Verify low confidence returns HOLD and negative expectancy is rejected."""
    policy = TacticalOrderPolicy(min_directional_confidence=0.55)

    # 1. Low confidence -> HOLD
    forecast_neutral = NextCandleForecast(
        direction_prob=0.51,
        is_bullish=True,
        body_ratio=0.1,
        upper_wick_ratio=0.4,
        lower_wick_ratio=0.5,
        range_to_atr=0.8,
        quantiles_high=[0.5, 1.0, 1.5],
        quantiles_low=[0.5, 1.0, 1.5],
        latency_ms=1.0,
    )
    prop_neutral = policy.generate_order_proposal(
        forecast=forecast_neutral, current_price=1.0850, atr=0.0010
    )
    assert prop_neutral.order_type == OrderType.HOLD_NO_ACTION
    assert prop_neutral.passed_expectancy_filter is False

    # 2. Insane spread friction -> Negative EV rejection
    forecast_high_friction = NextCandleForecast(
        direction_prob=0.58,
        is_bullish=True,
        body_ratio=0.5,
        upper_wick_ratio=0.2,
        lower_wick_ratio=0.3,
        range_to_atr=1.0,
        quantiles_high=[0.5, 1.2, 2.0],
        quantiles_low=[0.5, 1.0, 1.5],
        latency_ms=1.0,
    )
    prop_high_spread = policy.generate_order_proposal(
        forecast=forecast_high_friction,
        current_price=1.0850,
        atr=0.0010,
        spread=0.0050,  # 50 pip spread makes EV negative
    )
    assert prop_high_spread.passed_expectancy_filter is False
    assert "Expected Value" in str(prop_high_spread.rejection_reason)


def test_dynamic_expiration_horizon() -> None:
    """Verify expiration horizon scales inversely with volatility expansion."""
    policy = TacticalOrderPolicy(max_expiration_bars=6)

    # High volatility -> Fast expiration (e.g. 3 bars)
    exp_high_vol = policy.calculate_expiration_horizon(range_to_atr=1.8, volatility_expansion=2.0)
    assert exp_high_vol <= 3

    # Low volatility -> Patient expiration (e.g. 6 bars)
    exp_low_vol = policy.calculate_expiration_horizon(range_to_atr=0.5, volatility_expansion=0.6)
    assert exp_low_vol == 6
