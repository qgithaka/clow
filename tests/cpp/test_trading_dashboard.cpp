#include <cassert>
#include <iostream>
#include <cmath>
#include "../../src/ui/tactical_radar.h"
#include "../../src/ui/copilot_panel.h"
#include "../../src/ui/autopilot_loop.h"
#include "../../src/ui/positions_table.h"
#include "../../src/ui/execution_controller.h"
#include "../../src/risk/risk_manager.h"
#include "../../src/risk/order_state_machine.h"
#include "../../src/risk/kill_switch.h"
#include "../../src/broker/mt5_bridge.h"
#include "../../src/core/paper_simulation.h"

void test_tactical_radar_telemetry() {
    using namespace clow::ui;

    TacticalRadar radar(0.70);
    assert(!radar.has_prediction());
    assert(!radar.is_high_conviction());

    auto default_state = radar.get_telemetry_state();
    assert(default_state.symbol == "N/A");
    assert(!default_state.is_high_conviction);
    assert(!default_state.is_actionable);

    // Feed Bullish high-conviction signal
    RadarPrediction bull_pred;
    bull_pred.symbol = "EURUSD";
    bull_pred.timeframe = "M15";
    bull_pred.direction = RadarDirection::Bullish;
    bull_pred.bullish_probability = 0.76;
    bull_pred.bearish_probability = 0.24;
    bull_pred.confidence = 0.76;
    bull_pred.quantile_10 = 1.0830;
    bull_pred.quantile_50 = 1.0855;
    bull_pred.quantile_90 = 1.0890;
    bull_pred.has_tactical_proposal = true;
    bull_pred.proposal.symbol = "EURUSD";
    bull_pred.proposal.order_type = "BUY_LIMIT";
    bull_pred.proposal.entry_price = 1.0840;
    bull_pred.proposal.stop_loss = 1.0820;
    bull_pred.proposal.take_profit = 1.0880;

    radar.update_prediction(bull_pred);
    assert(radar.has_prediction());
    assert(radar.is_high_conviction());

    auto bull_state = radar.get_telemetry_state();
    assert(bull_state.symbol == "EURUSD");
    assert(bull_state.direction_str == "BULLISH");
    assert(std::abs(bull_state.confidence_pct - 76.0) < 1e-4);
    assert(bull_state.color_hex == "#00F59B");
    assert(bull_state.confidence_badge == "[HIGH CONVICTION]");
    assert(std::abs(bull_state.stop_pips - 20.0) < 1e-4);
    assert(std::abs(bull_state.target_pips - 40.0) < 1e-4);
    assert(std::abs(bull_state.reward_risk_ratio - 2.0) < 1e-4);
    assert(bull_state.is_actionable);

    std::string summary = radar.format_radar_summary();
    assert(summary.find("EURUSD") != std::string::npos);
    assert(summary.find("BULLISH") != std::string::npos);

    std::string breakdown = radar.format_tactical_breakdown();
    assert(breakdown.find("ACTIONABLE SETUP READY") != std::string::npos);

    std::cout << "[PASS] test_tactical_radar_telemetry" << std::endl;
}

void test_copilot_order_approval_workflow() {
    using namespace clow::ui;
    using namespace clow::risk;

    CopilotPanel panel;
    OrderStateMachine osm;

    panel.set_submission_handler([&osm](const OrderProposal& prop, double vol) -> int64_t {
        int64_t cid = osm.create_order(prop, vol);
        osm.transition_to(cid, OrderState::Submitted, "Co-Pilot human approved");
        osm.transition_to(cid, OrderState::Pending, "Order pending in market");
        return cid;
    });

    bool callback_fired = false;
    panel.set_action_callback([&callback_fired](const CopilotProposalCard& card) {
        if (card.action == CopilotAction::Approved) {
            callback_fired = true;
        }
    });

    OrderProposal prop;
    prop.symbol = "GBPUSD";
    prop.order_type = "SELL_LIMIT";
    prop.entry_price = 1.2850;
    prop.stop_loss = 1.2880;
    prop.take_profit = 1.2790;
    prop.win_probability = 0.65;

    int64_t pid = panel.queue_proposal(prop, 0.50, 1.0, 10000.0);
    assert(panel.pending_count() == 1);

    const auto* card = panel.get_proposal(pid);
    assert(card != nullptr);
    assert(card->action == CopilotAction::PendingReview);
    assert(std::abs(card->reward_risk_ratio - 2.0) < 1e-4);
    assert(std::abs(card->estimated_risk_usd - 100.0) < 1e-4);
    assert(std::abs(card->estimated_reward_usd - 200.0) < 1e-4);

    // Approve proposal
    [[maybe_unused]] bool ok = panel.approve_proposal(pid);
    assert(ok);
    assert(callback_fired);
    assert(panel.pending_count() == 0);
    assert(card->action == CopilotAction::Approved);
    assert(card->client_order_id > 0);

    [[maybe_unused]] const auto* osm_order = osm.get_order(card->client_order_id);
    assert(osm_order != nullptr);
    assert(osm_order->state == OrderState::Pending);

    // Rejection test on a second proposal
    int64_t pid2 = panel.queue_proposal(prop, 0.20, 0.5, 10000.0);
    assert(panel.pending_count() == 1);
    [[maybe_unused]] bool rej_ok = panel.reject_proposal(pid2, "High volatility news embargo");
    assert(rej_ok);
    assert(panel.get_proposal(pid2)->action == CopilotAction::Rejected);
    assert(panel.get_proposal(pid2)->rejection_reason == "High volatility news embargo");

    std::cout << "[PASS] test_copilot_order_approval_workflow" << std::endl;
}

void test_autopilot_execution_loop() {
    using namespace clow::ui;
    using namespace clow::risk;

    RiskManager risk_mgr;
    risk_mgr.set_trading_mode(TradingMode::Paper);
    OrderStateMachine osm;

    AutopilotConfig config;
    config.min_confidence = 0.70;
    config.min_reward_risk_ratio = 1.5;
    config.max_orders_per_hour = 5;

    AutopilotLoop autopilot(risk_mgr, osm, config);

    AccountState account;
    account.balance = 50000.0;
    account.equity = 50000.0;
    account.current_spread_pips = 1.0;

    SymbolSpecification spec;
    spec.symbol = "EURUSD";
    spec.pip_size = 0.0001;
    spec.pip_value_per_lot = 10.0;

    RadarPrediction low_conf_pred;
    low_conf_pred.symbol = "EURUSD";
    low_conf_pred.confidence = 0.62; // < 0.70
    low_conf_pred.has_tactical_proposal = true;
    low_conf_pred.proposal.symbol = "EURUSD";
    low_conf_pred.proposal.entry_price = 1.0840;
    low_conf_pred.proposal.stop_loss = 1.0820;
    low_conf_pred.proposal.take_profit = 1.0880;

    // Disabled test
    auto evt0 = autopilot.process_radar_signal(low_conf_pred, account, spec, 1000);
    assert(!evt0.threshold_passed);
    assert(evt0.status_message.find("disabled") != std::string::npos);

    // Enable Autopilot
    autopilot.set_enabled(true);

    // Low confidence signal rejected
    auto evt1 = autopilot.process_radar_signal(low_conf_pred, account, spec, 1000);
    assert(!evt1.threshold_passed);
    assert(!evt1.risk_approved);
    assert(evt1.client_order_id == 0);

    // High confidence signal passed
    RadarPrediction high_conf_pred;
    high_conf_pred.symbol = "EURUSD";
    high_conf_pred.confidence = 0.78; // > 0.70
    high_conf_pred.has_tactical_proposal = true;
    high_conf_pred.proposal.symbol = "EURUSD";
    high_conf_pred.proposal.order_type = "BUY_LIMIT";
    high_conf_pred.proposal.entry_price = 1.0840;
    high_conf_pred.proposal.stop_loss = 1.0820;
    high_conf_pred.proposal.take_profit = 1.0880; // R:R = 2.0 > 1.5

    auto evt2 = autopilot.process_radar_signal(high_conf_pred, account, spec, 2000);
    assert(evt2.threshold_passed);
    assert(evt2.risk_approved);
    assert(evt2.client_order_id > 0);
    assert(evt2.executed_lot_size > 0.0);
    assert(autopilot.total_dispatches_count() == 1);

    [[maybe_unused]] const auto* managed_ord = osm.get_order(evt2.client_order_id);
    assert(managed_ord != nullptr);
    assert(managed_ord->state == OrderState::Pending);

    // Risk gate rejection test: widened spread shock
    AccountState bad_spread_acc = account;
    bad_spread_acc.current_spread_pips = 4.5; // > max 2.5 pips
    auto evt3 = autopilot.process_radar_signal(high_conf_pred, bad_spread_acc, spec, 3000);
    assert(evt3.threshold_passed);
    assert(!evt3.risk_approved);
    assert(evt3.client_order_id == 0);
    assert(evt3.status_message.find("Risk Gate Rejection") != std::string::npos);

    std::cout << "[PASS] test_autopilot_execution_loop" << std::endl;
}

void test_positions_table_mark_to_market() {
    using namespace clow::ui;

    PositionsTable table;
    assert(table.total_count() == 0);

    // 1. Add BUY position on EURUSD (Open @ 1.0850, 1.00 Lot)
    PositionRow pos1;
    pos1.ticket = 7001;
    pos1.symbol = "EURUSD";
    pos1.order_type = "BUY";
    pos1.volume = 1.00;
    pos1.open_price = 1.08500;
    pos1.current_price = 1.08500;
    pos1.stop_loss = 1.08200;
    pos1.take_profit = 1.09100;
    pos1.is_position = true;
    table.add_or_update(pos1);

    // 2. Add SELL position on EURUSD (Open @ 1.0850, 0.50 Lot)
    PositionRow pos2;
    pos2.ticket = 7002;
    pos2.symbol = "EURUSD";
    pos2.order_type = "SELL";
    pos2.volume = 0.50;
    pos2.open_price = 1.08500;
    pos2.current_price = 1.08500;
    pos2.stop_loss = 1.08800;
    pos2.take_profit = 1.08000;
    pos2.is_position = true;
    table.add_or_update(pos2);

    // 3. Add Pending BUY_LIMIT on EURUSD (Entry 1.0830)
    PositionRow ord1;
    ord1.ticket = 8001;
    ord1.symbol = "EURUSD";
    ord1.order_type = "BUY_LIMIT";
    ord1.volume = 0.25;
    ord1.open_price = 1.08300;
    ord1.current_price = 1.08300;
    ord1.stop_loss = 1.08100;
    ord1.take_profit = 1.08700;
    ord1.is_position = false;
    table.add_or_update(ord1);

    assert(table.total_count() == 3);
    assert(table.get_open_positions().size() == 2);
    assert(table.get_pending_orders().size() == 1);

    // Update quote: EURUSD rises to Bid 1.08700 / Ask 1.08712 (+20 pips for BUY, -21.2 pips for SELL)
    table.update_quote("EURUSD", 1.08700, 1.08712);

    [[maybe_unused]] const auto* updated_pos1 = table.get(7001);
    assert(updated_pos1 != nullptr);
    assert(std::abs(updated_pos1->floating_pips - 20.0) < 1e-3);
    assert(std::abs(updated_pos1->floating_pnl - 200.0) < 1e-2);

    [[maybe_unused]] const auto* updated_pos2 = table.get(7002);
    assert(updated_pos2 != nullptr);
    assert(std::abs(updated_pos2->floating_pips - (-21.2)) < 1e-3);
    assert(std::abs(updated_pos2->floating_pnl - (-106.0)) < 1e-2);

    [[maybe_unused]] const auto* updated_ord1 = table.get(8001);
    assert(updated_ord1 != nullptr);
    assert(updated_ord1->floating_pnl == 0.0);
    assert(updated_ord1->floating_pips == 0.0);

    auto summary = table.get_summary();
    assert(summary.open_positions_count == 2);
    assert(summary.pending_orders_count == 1);
    assert(std::abs(summary.total_open_lots - 1.50) < 1e-4);
    assert(std::abs(summary.total_floating_pnl - 94.0) < 1e-2);

    std::string ascii_table = table.format_table_ascii();
    assert(ascii_table.find("LIVE TRADING DOCK") != std::string::npos);
    assert(ascii_table.find("7001") != std::string::npos);

    table.remove(7001);
    assert(table.get_open_positions().size() == 1);

    std::cout << "[PASS] test_positions_table_mark_to_market" << std::endl;
}

void test_execution_controller_lifecycle() {
    using namespace clow::ui;
    using namespace clow::risk;
    using namespace clow::broker;

    MT5Bridge bridge;
    bridge.initialize(123456, "demo_pass", "MetaQuotes-Demo");

    RiskManager risk_mgr;
    risk_mgr.set_trading_mode(TradingMode::Paper);

    OrderStateMachine osm;
    KillSwitch kill_switch(risk_mgr, osm);
    PositionsTable table;

    ExecutionController controller(bridge, osm, risk_mgr, kill_switch, table);

    // Create a position
    OrderProposal prop;
    prop.symbol = "EURUSD";
    prop.order_type = "BUY";
    prop.entry_price = 1.0850;
    prop.stop_loss = 1.0820;
    prop.take_profit = 1.0900;

    int64_t cid = osm.create_order(prop, 0.50);
    osm.transition_to(cid, OrderState::Submitted);
    osm.on_order_fill(cid, 99001, 1.0850);

    // Create a pending order
    int64_t cid_pending = osm.create_order(prop, 0.25);
    osm.transition_to(cid_pending, OrderState::Submitted);
    osm.transition_to(cid_pending, OrderState::Pending);

    controller.sync_state();
    assert(table.total_count() == 2);
    assert(table.get_open_positions().size() == 1);
    assert(table.get_pending_orders().size() == 1);

    // Modify protection
    [[maybe_unused]] bool mod_ok = controller.modify_protection(99001, 1.0830, 1.0920);
    assert(mod_ok);
    assert(table.get(99001)->stop_loss == 1.0830);
    assert(table.get(99001)->take_profit == 1.0920);

    // Single-click close position
    [[maybe_unused]] bool close_ok = controller.close_position(99001);
    assert(close_ok);
    assert(table.get_open_positions().empty());
    assert(osm.get_order(cid)->state == OrderState::Closed);

    // Single-click cancel pending order
    [[maybe_unused]] bool cancel_ok = controller.cancel_pending_order(cid_pending);
    assert(cancel_ok);
    assert(table.get_pending_orders().empty());
    assert(osm.get_order(cid_pending)->state == OrderState::Cancelled);

    // Emergency Panic Liquidation test
    int64_t cid3 = osm.create_order(prop, 1.0);
    osm.transition_to(cid3, OrderState::Submitted);
    osm.on_order_fill(cid3, 99003, 1.0850);
    controller.sync_state();
    assert(table.total_count() == 1);

    auto summary = controller.panic_liquidate("Test Panic");
    assert(summary.execution_success);
    assert(table.total_count() == 0);
    assert(kill_switch.is_active());

    std::cout << "[PASS] test_execution_controller_lifecycle" << std::endl;
}

void test_end_to_end_paper_simulation() {
    using namespace clow::core;

    MultiPairSimConfig config;
    config.symbols = {"EURUSD", "GBPUSD", "USDJPY"};
    config.ticks_per_pair = 200;
    config.initial_balance = 25000.0;
    config.min_conviction_threshold = 0.70;

    PaperSimulationEngine engine(config);
    auto report = engine.run_simulation();

    assert(report.total_ticks_ingested == 600);
    assert(report.inference_signals_evaluated == 12);
    assert(report.orders_dispatched > 0);
    assert(report.positions_filled == report.orders_dispatched);
    assert(report.positions_closed == report.positions_filled);
    assert(report.ending_equity >= report.starting_equity);
    assert(report.zero_gate_breaches);
    assert(report.summary_text.find("VERIFIED") != std::string::npos);

    std::cout << "[PASS] test_end_to_end_paper_simulation" << std::endl;
}
