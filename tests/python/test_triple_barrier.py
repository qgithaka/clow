"""Unit tests for TripleBarrierEngine Take-Profit, Stop-Loss, and Time Expiration labeling."""

import numpy as np
import pandas as pd

from training.models.triple_barrier import BarrierOutcome, TripleBarrierEngine


def test_long_tp_and_sl_outcomes() -> None:
    """Verify clean Take-Profit hit, Stop-Loss hit, and Time expiration for Long trades."""
    # Construct scenario: Entry at bar 0 (close=1.0850, atr=0.0010)
    # TP = 1.0850 + 2*0.0010 = 1.0870
    # SL = 1.0850 - 1*0.0010 = 1.0840
    dates = pd.date_range("2023-01-01", periods=10, freq="5min", tz="UTC")

    # 1. TP scenario: Bar 2 reaches High=1.0875
    df_tp = pd.DataFrame({
        "timestamp_utc": dates,
        "open": [1.0850, 1.0855, 1.0865] + [1.0870] * 7,
        "high": [1.0852, 1.0860, 1.0875] + [1.0875] * 7,
        "low": [1.0848, 1.0852, 1.0860] + [1.0865] * 7,
        "close": [1.0850, 1.0858, 1.0870] + [1.0870] * 7,
        "atr": [0.0010] * 10,
    })

    res_tp = TripleBarrierEngine.evaluate_barriers(df_tp, entry_idx=0, side=+1, entry_price=1.0850)
    assert res_tp.outcome == BarrierOutcome.HIT_TP
    assert res_tp.exit_idx == 2
    assert res_tp.holding_period_bars == 2
    assert res_tp.net_return > 0.0

    # 2. SL scenario: Bar 1 drops to Low=1.0835 (SL is 1.0840)
    df_sl = pd.DataFrame({
        "timestamp_utc": dates,
        "open": [1.0850, 1.0845] + [1.0830] * 8,
        "high": [1.0852, 1.0848] + [1.0835] * 8,
        "low": [1.0848, 1.0835] + [1.0825] * 8,
        "close": [1.0850, 1.0840] + [1.0830] * 8,
        "atr": [0.0010] * 10,
    })

    res_sl = TripleBarrierEngine.evaluate_barriers(df_sl, entry_idx=0, side=+1, entry_price=1.0850)
    assert res_sl.outcome == BarrierOutcome.HIT_SL
    assert res_sl.exit_idx == 1
    assert res_sl.holding_period_bars == 1
    assert res_sl.net_return < 0.0

    # 3. Timeout scenario: Price stays in narrow band for 5 bars
    df_timeout = pd.DataFrame({
        "timestamp_utc": dates,
        "open": [1.0850] * 10,
        "high": [1.0855] * 10,
        "low": [1.0845] * 10,
        "close": [1.0852] * 10,
        "atr": [0.0010] * 10,
    })

    res_timeout = TripleBarrierEngine.evaluate_barriers(
        df_timeout, entry_idx=0, side=+1, entry_price=1.0850, horizon_bars=4
    )
    assert res_timeout.outcome == BarrierOutcome.TIMEOUT
    assert res_timeout.exit_idx == 4
    assert res_timeout.holding_period_bars == 4


def test_short_barrier_outcomes() -> None:
    """Verify Short trade barrier evaluation (Low touches TP, High touches SL)."""
    dates = pd.date_range("2023-01-01", periods=5, freq="5min", tz="UTC")
    # Entry at 1.0850, ATR=0.0010 -> TP=1.0830, SL=1.0860
    df_short_tp = pd.DataFrame({
        "timestamp_utc": dates,
        "open": [1.0850, 1.0840, 1.0830, 1.0825, 1.0820],
        "high": [1.0855, 1.0845, 1.0835, 1.0830, 1.0825],
        "low": [1.0845, 1.0835, 1.0825, 1.0820, 1.0815],
        "close": [1.0850, 1.0838, 1.0828, 1.0822, 1.0818],
        "atr": [0.0010] * 5,
    })

    res_short = TripleBarrierEngine.evaluate_barriers(df_short_tp, entry_idx=0, side=-1, entry_price=1.0850)
    assert res_short.outcome == BarrierOutcome.HIT_TP
    assert res_short.exit_idx == 2
    assert res_short.net_return > 0.0


def test_label_dataset_batch() -> None:
    """Verify batch dataset labeling adds buy and sell barrier columns."""
    dates = pd.date_range("2023-01-01", periods=30, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "open": np.linspace(1.0800, 1.0850, 30),
        "high": np.linspace(1.0805, 1.0855, 30),
        "low": np.linspace(1.0795, 1.0845, 30),
        "close": np.linspace(1.0802, 1.0852, 30),
        "atr": [0.0010] * 30,
    })

    labeled = TripleBarrierEngine.label_dataset(df, horizon_bars=5)
    assert "tb_buy_outcome" in labeled.columns
    assert "tb_buy_net_ret" in labeled.columns
    assert "tb_sell_outcome" in labeled.columns
    assert "tb_sell_net_ret" in labeled.columns
    assert len(labeled) == 30

    empty_labeled = TripleBarrierEngine.label_dataset(pd.DataFrame())
    assert empty_labeled.empty
