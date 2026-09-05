#include "risk/risk_manager.h"
#include "core/logger.h"

#include <algorithm>
#include <cmath>

namespace clow::risk {

RiskManager::RiskManager(
    double max_daily_drawdown_pct,
    double max_account_risk_pct,
    int max_open_trades,
    double max_spread_pips,
    double min_confidence
)
    : m_max_daily_drawdown_pct(max_daily_drawdown_pct),
      m_max_account_risk_pct(max_account_risk_pct),
      m_max_open_trades(max_open_trades),
      m_max_spread_pips(max_spread_pips),
      m_min_confidence(min_confidence),
      m_sizer(SizingMethod::FixedFractional, max_account_risk_pct) {}

void RiskManager::update_equity(double equity) {
    if (equity <= 0.0) return;

    if (m_day_start_equity <= 0.0) {
        m_day_start_equity = equity;
    }
    if (equity > m_peak_daily_equity) {
        m_peak_daily_equity = equity;
    }

    double dd_from_start = (m_day_start_equity - equity) / m_day_start_equity * 100.0;
    m_current_daily_drawdown_pct = std::max(0.0, dd_from_start);

    if (m_current_daily_drawdown_pct >= m_max_daily_drawdown_pct) {
        halt_trading("Daily maximum drawdown threshold breached (" +
                     std::to_string(m_current_daily_drawdown_pct) + "% >= " +
                     std::to_string(m_max_daily_drawdown_pct) + "%)");
    }
}

void RiskManager::reset_daily_stats(double starting_equity) {
    m_day_start_equity = starting_equity;
    m_peak_daily_equity = starting_equity;
    m_current_daily_drawdown_pct = 0.0;
    if (m_halted && m_halt_reason.rfind("Daily maximum drawdown", 0) == 0) {
        resume_trading();
    }
    CLOW_LOG_INFO("Reset daily risk statistics with baseline equity: " + std::to_string(starting_equity));
}

void RiskManager::halt_trading(const std::string& reason) {
    m_halted = true;
    m_halt_reason = reason;
    CLOW_LOG_CRITICAL("SOVEREIGN RISK HALT TRIGGERED: " + reason);
}

void RiskManager::resume_trading() {
    m_halted = false;
    m_halt_reason.clear();
    CLOW_LOG_INFO("Trading permissions resumed.");
}

RiskDecision RiskManager::evaluate_order(
    const OrderProposal& proposal,
    const AccountState& account,
    const SymbolSpecification& symbol_spec
) {
    RiskDecision decision;
    decision.approved = false;

    // Gate 1: Trading Mode Permission Gate
    if (m_mode == TradingMode::Disabled) {
        decision.triggered_gate = "TradingModeGate";
        decision.rejection_reason = "Live and automated trading is disabled";
        return decision;
    }

    // Gate 2: Sovereign Panic / System Halt Gate
    if (m_halted) {
        decision.triggered_gate = "SystemHaltGate";
        decision.rejection_reason = "Trading halted: " + m_halt_reason;
        return decision;
    }

    // Gate 3: Spread Shock Filter Gate
    if (account.current_spread_pips > m_max_spread_pips) {
        decision.triggered_gate = "MaxSpreadGate";
        decision.rejection_reason = "Current broker spread (" +
                                    std::to_string(account.current_spread_pips) +
                                    " pips) exceeds max threshold (" +
                                    std::to_string(m_max_spread_pips) + " pips)";
        return decision;
    }

    // Gate 4: Intraday Drawdown Gate
    update_equity(account.equity);
    if (m_current_daily_drawdown_pct >= m_max_daily_drawdown_pct) {
        decision.triggered_gate = "DailyDrawdownGate";
        decision.rejection_reason = "Daily loss limit breached (" +
                                    std::to_string(m_current_daily_drawdown_pct) + "% >= " +
                                    std::to_string(m_max_daily_drawdown_pct) + "%)";
        return decision;
    }

    // Gate 5: Concurrent Open Trades Gate
    int total_active = account.open_positions_count + account.pending_orders_count;
    if (total_active >= m_max_open_trades) {
        decision.triggered_gate = "MaxOpenTradesGate";
        decision.rejection_reason = "Concurrent active orders/positions (" +
                                    std::to_string(total_active) +
                                    ") reached max limit (" +
                                    std::to_string(m_max_open_trades) + ")";
        return decision;
    }

    // Gate 6: Minimum Directional Confidence Gate
    if (proposal.win_probability < m_min_confidence) {
        decision.triggered_gate = "MinConfidenceGate";
        decision.rejection_reason = "Model confidence (" +
                                    std::to_string(proposal.win_probability) +
                                    ") below threshold (" +
                                    std::to_string(m_min_confidence) + ")";
        return decision;
    }

    // Gate 7: Mathematical Expectancy Gate
    if (proposal.expected_value <= 0.0) {
        decision.triggered_gate = "ExpectancyGate";
        decision.rejection_reason = "Expected value is non-positive (" +
                                    std::to_string(proposal.expected_value) + ")";
        return decision;
    }

    // Gate 8: Position Sizing & Stop Distance Gate
    double stop_dist_price = std::abs(proposal.entry_price - proposal.stop_loss);
    double target_dist_price = std::abs(proposal.take_profit - proposal.entry_price);
    double rr_ratio = (stop_dist_price > 0.0) ? (target_dist_price / stop_dist_price) : 1.0;

    m_sizer.set_max_risk_pct(m_max_account_risk_pct);
    auto sizing = m_sizer.calculate_lots(
        account.equity,
        stop_dist_price,
        symbol_spec,
        proposal.win_probability,
        rr_ratio
    );

    if (!sizing.valid) {
        decision.triggered_gate = "PositionSizingGate";
        decision.rejection_reason = sizing.rejection_reason;
        return decision;
    }

    decision.approved = true;
    decision.approved_lot_size = sizing.lot_size;
    decision.approved_risk_pct = sizing.calculated_risk_pct;

    CLOW_LOG_INFO("Order approved for " + proposal.symbol +
                  " [" + proposal.order_type + "] Lots=" + std::to_string(decision.approved_lot_size) +
                  " Risk=" + std::to_string(decision.approved_risk_pct) + "%");

    return decision;
}

} // namespace clow::risk
