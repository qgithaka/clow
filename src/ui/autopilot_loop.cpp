#include "ui/autopilot_loop.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace clow::ui {

AutopilotLoop::AutopilotLoop(
    clow::risk::RiskManager& risk_mgr,
    clow::risk::OrderStateMachine& state_machine,
    AutopilotConfig config
)
    : m_risk_mgr(risk_mgr),
      m_state_machine(state_machine),
      m_config(config) {}

bool AutopilotLoop::check_rate_limit(int64_t now_ms) {
    int64_t one_hour_ago = now_ms - 3600000;
    m_recent_dispatch_timestamps.erase(
        std::remove_if(m_recent_dispatch_timestamps.begin(), m_recent_dispatch_timestamps.end(),
            [one_hour_ago](int64_t ts) { return ts < one_hour_ago; }),
        m_recent_dispatch_timestamps.end()
    );

    return static_cast<int>(m_recent_dispatch_timestamps.size()) < m_config.max_orders_per_hour;
}

AutopilotTriggerEvent AutopilotLoop::process_radar_signal(
    const RadarPrediction& prediction,
    const clow::risk::AccountState& account,
    const clow::risk::SymbolSpecification& symbol_spec,
    int64_t current_timestamp_ms
) {
    AutopilotTriggerEvent event;
    event.event_id = m_next_event_id++;
    event.timestamp_ms = current_timestamp_ms > 0 ? current_timestamp_ms : prediction.timestamp_ms;
    event.symbol = prediction.symbol;
    event.confidence = prediction.confidence;

    if (!m_enabled) {
        event.status_message = "Auto-Pilot is currently disabled";
        m_history.push_back(event);
        return event;
    }

    if (m_config.require_active_trading_mode && m_risk_mgr.trading_mode() == clow::risk::TradingMode::Disabled) {
        event.status_message = "Execution rejected: Trading mode is Disabled";
        m_history.push_back(event);
        return event;
    }

    if (!prediction.has_tactical_proposal) {
        event.status_message = "No tactical proposal in signal";
        m_history.push_back(event);
        return event;
    }

    if (prediction.confidence < m_config.min_confidence) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1);
        oss << "Confidence " << (prediction.confidence * 100.0) << "% below "
            << (m_config.min_confidence * 100.0) << "% threshold";
        event.status_message = oss.str();
        m_history.push_back(event);
        return event;
    }

    event.threshold_passed = true;

    const auto& prop = prediction.proposal;
    double stop_dist = std::abs(prop.entry_price - prop.stop_loss);
    double target_dist = std::abs(prop.take_profit - prop.entry_price);
    double rr = (stop_dist > 1e-6) ? (target_dist / stop_dist) : 0.0;

    if (rr < m_config.min_reward_risk_ratio) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "Reward:Risk ratio " << rr << " below " << m_config.min_reward_risk_ratio << " requirement";
        event.status_message = oss.str();
        m_history.push_back(event);
        return event;
    }

    if (!check_rate_limit(event.timestamp_ms)) {
        event.status_message = "Hourly dispatch rate limit reached";
        m_history.push_back(event);
        return event;
    }

    // Sovereign Risk Gate Evaluation
    auto decision = m_risk_mgr.evaluate_order(prop, account, symbol_spec);
    if (!decision.approved) {
        event.risk_approved = false;
        event.status_message = "Risk Gate Rejection: " + decision.rejection_reason;
        m_history.push_back(event);
        if (m_event_callback) {
            m_event_callback(event);
        }
        return event;
    }

    event.risk_approved = true;
    event.executed_lot_size = decision.approved_lot_size;

    // Autonomous submission and transition into pending state
    int64_t client_id = m_state_machine.create_order(prop, decision.approved_lot_size);
    m_state_machine.transition_to(client_id, clow::risk::OrderState::Submitted, "Auto-Pilot autonomous dispatch");
    m_state_machine.transition_to(client_id, clow::risk::OrderState::Pending, "Auto-Pilot pending order active");

    event.client_order_id = client_id;
    event.status_message = "Order autonomously submitted and pending";

    m_recent_dispatch_timestamps.push_back(event.timestamp_ms);
    ++m_dispatches_count;
    m_history.push_back(event);

    if (m_event_callback) {
        m_event_callback(event);
    }

    return event;
}

void AutopilotLoop::reset_history() noexcept {
    m_history.clear();
    m_recent_dispatch_timestamps.clear();
    m_dispatches_count = 0;
}

} // namespace clow::ui
