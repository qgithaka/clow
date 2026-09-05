#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

#include "../../src/risk/kill_switch.h"
#include "../../src/risk/order_state_machine.h"
#include "../../src/risk/position_sizer.h"
#include "../../src/risk/risk_audit_logger.h"
#include "../../src/risk/risk_manager.h"

void test_position_sizer_formulas() {
    using namespace clow::risk;

    PositionSizer sizer(SizingMethod::FixedFractional, 1.0, 50.0);
    SymbolSpecification spec;
    spec.symbol = "EURUSD";
    spec.pip_size = 0.0001;
    spec.pip_value_per_lot = 10.0;
    spec.lot_step = 0.01;
    spec.min_lot = 0.01;
    spec.max_lot = 50.0;

    // 1% risk on $10,000 equity = $100 cash risk
    // 20 pips stop loss (0.0020) = $200 risk per standard lot
    // Expected lot size = $100 / $200 = 0.50 lots
    double equity = 10000.0;
    double stop_dist = 0.0020; // 20 pips
    auto res = sizer.calculate_lots(equity, stop_dist, spec);

    assert(res.valid);
    assert(std::abs(res.lot_size - 0.50) < 1e-4);
    assert(std::abs(res.cash_risk - 100.0) < 1e-4);
    assert(std::abs(res.calculated_risk_pct - 1.0) < 1e-4);

    // Test Half-Kelly Sizing
    sizer.set_method(SizingMethod::HalfKelly);
    // p = 0.60, b = 2.0 (RR=2:1), q = 0.40
    // Full Kelly f* = (0.60 * 2 - 0.40) / 2 = 0.80 / 2 = 0.40
    // Half Kelly = 0.50 * 0.40 = 0.20 (20% dynamic risk)
    // Capped by max_risk_pct = 1.0%
    auto res_kelly = sizer.calculate_lots(equity, stop_dist, spec, 0.60, 2.0);
    assert(res_kelly.valid);
    assert(std::abs(res_kelly.lot_size - 0.50) < 1e-4);

    // Test negative expectancy Kelly rejection
    // p = 0.30, b = 1.0 -> f* = (0.30 - 0.70) / 1.0 = -0.40 <= 0 -> rejected
    auto res_neg = sizer.calculate_lots(equity, stop_dist, spec, 0.30, 1.0);
    assert(!res_neg.valid);

    std::cout << "[PASS] test_position_sizer_formulas" << std::endl;
}

void test_sovereign_risk_gates() {
    using namespace clow::risk;

    RiskManager mgr(4.0, 1.0, 3, 2.5, 0.55);
    SymbolSpecification spec;

    AccountState account;
    account.balance = 10000.0;
    account.equity = 10000.0;
    account.current_spread_pips = 1.2;
    account.open_positions_count = 0;
    account.pending_orders_count = 0;

    OrderProposal proposal;
    proposal.symbol = "EURUSD";
    proposal.order_type = "BUY_LIMIT";
    proposal.current_price = 1.0850;
    proposal.entry_price = 1.0840;
    proposal.stop_loss = 1.0820; // 20 pips stop
    proposal.take_profit = 1.0880; // 40 pips target
    proposal.win_probability = 0.60;
    proposal.expected_value = 0.0015;

    // Gate 1: Disabled mode rejection
    mgr.set_trading_mode(TradingMode::Disabled);
    auto d1 = mgr.evaluate_order(proposal, account, spec);
    assert(!d1.approved);
    assert(d1.triggered_gate == "TradingModeGate");

    // Enable Paper mode
    mgr.set_trading_mode(TradingMode::Paper);
    auto d2 = mgr.evaluate_order(proposal, account, spec);
    assert(d2.approved);
    assert(d2.approved_lot_size > 0.0);

    // Gate 2: Spread Shock filter rejection (> 2.5 pips)
    account.current_spread_pips = 3.5;
    auto d3 = mgr.evaluate_order(proposal, account, spec);
    assert(!d3.approved);
    assert(d3.triggered_gate == "MaxSpreadGate");
    account.current_spread_pips = 1.2;

    // Gate 3: Max Open Trades rejection (>= 3)
    account.open_positions_count = 2;
    account.pending_orders_count = 1; // Total = 3
    auto d4 = mgr.evaluate_order(proposal, account, spec);
    assert(!d4.approved);
    assert(d4.triggered_gate == "MaxOpenTradesGate");
    account.open_positions_count = 0;
    account.pending_orders_count = 0;

    // Gate 4: Intraday Drawdown breach (equity drops to $9,500 = 5% DD > 4% cap)
    account.equity = 9500.0;
    auto d5 = mgr.evaluate_order(proposal, account, spec);
    assert(!d5.approved);
    assert(d5.triggered_gate == "DailyDrawdownGate" || d5.triggered_gate == "SystemHaltGate");
    assert(mgr.is_halted());

    // Reset daily stats to restore trading
    mgr.reset_daily_stats(10000.0);
    assert(!mgr.is_halted());

    std::cout << "[PASS] test_sovereign_risk_gates" << std::endl;
}

void test_order_state_machine_lifecycle() {
    using namespace clow::risk;

    OrderStateMachine sm;
    OrderProposal proposal;
    proposal.symbol = "EURUSD";
    proposal.order_type = "BUY_LIMIT";
    proposal.entry_price = 1.0840;
    proposal.stop_loss = 1.0820;
    proposal.take_profit = 1.0880;
    proposal.expiration_bars = 2; // Expire after 2 bars

    int64_t id = sm.create_order(proposal, 0.50);
    assert(id > 0);
    const auto* o = sm.get_order(id);
    assert(o != nullptr);
    assert(o->state == OrderState::Created);

    // Transition: Created -> Submitted -> Pending
    assert(sm.transition_to(id, OrderState::Submitted));
    assert(sm.transition_to(id, OrderState::Pending));
    assert(sm.get_pending_orders().size() == 1);

    // Bar 1 tick: bars_elapsed = 1, not expired
    auto exp1 = sm.on_bar_tick("EURUSD");
    assert(exp1.empty());
    assert(sm.get_order(id)->state == OrderState::Pending);

    // Bar 2 tick: bars_elapsed = 2 >= 2 -> expired
    auto exp2 = sm.on_bar_tick("EURUSD");
    assert(exp2.size() == 1);
    assert(exp2[0] == id);
    assert(sm.get_order(id)->state == OrderState::Expired);
    assert(sm.get_pending_orders().empty());

    // Test fill lifecycle
    int64_t id2 = sm.create_order(proposal, 0.50);
    sm.transition_to(id2, OrderState::Submitted);
    sm.transition_to(id2, OrderState::Pending);
    sm.on_order_fill(id2, 99901, 1.0840);
    assert(sm.get_order(id2)->state == OrderState::Filled);
    assert(sm.get_active_positions().size() == 1);

    // Close position
    sm.on_order_close(id2, 1.0880, 200.0);
    assert(sm.get_order(id2)->state == OrderState::Closed);
    assert(sm.get_active_positions().empty());

    std::cout << "[PASS] test_order_state_machine_lifecycle" << std::endl;
}

void test_kill_switch_panic_liquidation() {
    using namespace clow::risk;

    RiskManager mgr;
    mgr.set_trading_mode(TradingMode::Paper);
    OrderStateMachine sm;

    OrderProposal p;
    p.symbol = "EURUSD";
    p.order_type = "BUY_LIMIT";
    p.entry_price = 1.0840;
    p.stop_loss = 1.0820;
    p.take_profit = 1.0880;

    // Create 2 pending orders
    int64_t id1 = sm.create_order(p, 0.20);
    sm.transition_to(id1, OrderState::Submitted);
    sm.transition_to(id1, OrderState::Pending);

    int64_t id2 = sm.create_order(p, 0.30);
    sm.transition_to(id2, OrderState::Submitted);
    sm.transition_to(id2, OrderState::Pending);

    // Create 1 filled position
    int64_t id3 = sm.create_order(p, 0.50);
    sm.transition_to(id3, OrderState::Submitted);
    sm.on_order_fill(id3, 88801, 1.0840);

    assert(sm.get_pending_orders().size() == 2);
    assert(sm.get_active_positions().size() == 1);

    KillSwitch ks(mgr, sm);
    bool broker_close_called = false;
    ks.set_broker_close_handler([&](int64_t ticket, const std::string&, double) {
        if (ticket == 88801) broker_close_called = true;
        return true;
    });

    // Trigger panic kill-switch
    auto summary = ks.trigger_panic("Emergency Broker Disconnect Event");

    assert(summary.execution_success);
    assert(summary.pending_orders_cancelled == 2);
    assert(summary.active_positions_closed == 1);
    assert(broker_close_called);
    assert(sm.get_pending_orders().empty());
    assert(sm.get_active_positions().empty());
    assert(mgr.is_halted());
    assert(mgr.trading_mode() == TradingMode::Disabled);

    std::cout << "[PASS] test_kill_switch_panic_liquidation" << std::endl;
}

void test_risk_audit_logger() {
    using namespace clow::risk;

    RiskAuditLogger logger("logs/test_risk_audit.log");
    logger.clear_in_memory();

    logger.log_rejection("EURUSD", "MaxSpreadGate", "Spread 4.0 > 2.5 pips");
    logger.log_state_transition(1001, "EURUSD", "PENDING", "EXPIRED");

    assert(logger.records_count() == 2);
    auto recs = logger.get_records();
    assert(recs[0].event_type == AuditEventType::OrderRejected);
    assert(recs[0].gate_or_state == "MaxSpreadGate");
    assert(recs[1].event_type == AuditEventType::StateTransition);

    std::cout << "[PASS] test_risk_audit_logger" << std::endl;
}
