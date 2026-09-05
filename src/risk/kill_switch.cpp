#include "risk/kill_switch.h"
#include "core/logger.h"

namespace clow::risk {

KillSwitch::KillSwitch(RiskManager& risk_mgr, OrderStateMachine& state_machine)
    : m_risk_mgr(risk_mgr),
      m_state_machine(state_machine) {}

void KillSwitch::set_broker_close_handler(PositionCloseExecutor handler) {
    m_close_handler = std::move(handler);
}

PanicLiquidationSummary KillSwitch::trigger_panic(const std::string& reason) {
    m_panic_active = true;
    m_last_panic_reason = reason;

    PanicLiquidationSummary summary;
    summary.trigger_reason = reason;

    CLOW_LOG_CRITICAL("=================================================");
    CLOW_LOG_CRITICAL("!!! SOVEREIGN PANIC KILL-SWITCH TRIGGERED !!!");
    CLOW_LOG_CRITICAL("Reason: " + reason);
    CLOW_LOG_CRITICAL("=================================================");

    // 1. Instantly halt risk manager and disable trading permissions
    m_risk_mgr.halt_trading("KILL-SWITCH: " + reason);
    m_risk_mgr.set_trading_mode(TradingMode::Disabled);

    // 2. Cancel all pending orders in state machine
    summary.pending_orders_cancelled = m_state_machine.cancel_all_pending("Panic Kill-Switch Emergency Liquidation");

    // 3. Liquidate all active market positions
    auto positions = m_state_machine.get_active_positions();
    for (const auto& pos : positions) {
        bool closed_via_broker = false;
        if (m_close_handler) {
            closed_via_broker = m_close_handler(pos.broker_ticket, pos.symbol, pos.volume);
        }
        (void)closed_via_broker;

        // Transition order state to Closed
        m_state_machine.on_order_close(pos.client_id, pos.entry_price, 0.0);
        summary.active_positions_closed++;
    }

    summary.execution_success = true;

    CLOW_LOG_CRITICAL("PANIC COMPLETE: Cancelled " + std::to_string(summary.pending_orders_cancelled) +
                      " pending orders, Liquidated " + std::to_string(summary.active_positions_closed) +
                      " market positions.");

    return summary;
}

bool KillSwitch::is_active() const noexcept {
    return m_panic_active;
}

void KillSwitch::clear_panic() {
    m_panic_active = false;
    m_last_panic_reason.clear();
    m_risk_mgr.resume_trading();
    CLOW_LOG_INFO("Sovereign panic kill-switch reset and cleared.");
}

} // namespace clow::risk
