#include "ui/main_window.h"
#include "core/logger.h"

namespace clow::ui {

MainWindowController::MainWindowController()
    : m_risk(4.0, 1.0, 3, 2.5, 0.55),
      m_state_machine(),
      m_kill_switch(m_risk, m_state_machine) {
    
    // Connect state machine callbacks to toast manager
    m_state_machine.set_state_change_callback([this](const risk::ManagedOrder& order, risk::OrderState, risk::OrderState new_state) {
        if (new_state == risk::OrderState::Filled) {
            m_toasts.show_toast(
                ToastType::Success,
                "Order Filled",
                "Order #" + std::to_string(order.client_id) + " (" + order.symbol + ") filled @ " + std::to_string(order.fill_price)
            );
        } else if (new_state == risk::OrderState::Expired) {
            m_toasts.show_toast(
                ToastType::Warning,
                "Order Expired",
                "Pending order #" + std::to_string(order.client_id) + " reached expiration timeout"
            );
        } else if (new_state == risk::OrderState::Cancelled) {
            m_toasts.show_toast(
                ToastType::Info,
                "Order Cancelled",
                "Order #" + std::to_string(order.client_id) + " cancelled"
            );
        }
    });
}

bool MainWindowController::switch_symbol(const std::string& symbol) {
    if (m_sidebar.set_symbol(symbol)) {
        CLOW_LOG_INFO("UI: Active symbol switched to " + symbol);
        m_toasts.show_toast(ToastType::Info, "Symbol Changed", "Active chart switched to " + symbol, 2000);
        return true;
    }
    return false;
}

bool MainWindowController::switch_timeframe(const std::string& timeframe) {
    if (m_sidebar.set_timeframe(timeframe)) {
        CLOW_LOG_INFO("UI: Active timeframe switched to " + timeframe);
        m_toasts.show_toast(ToastType::Info, "Timeframe Changed", "Timeframe updated to " + timeframe, 2000);
        return true;
    }
    return false;
}

risk::PanicLiquidationSummary MainWindowController::on_panic_button_clicked() {
    auto summary = m_kill_switch.trigger_panic("Manual Emergency Kill-Switch button clicked in UI");
    m_toasts.show_toast(
        ToastType::Critical,
        "SOVEREIGN PANIC ACTIVATED",
        "Cancelled " + std::to_string(summary.pending_orders_cancelled) +
        " pending orders, Liquidated " + std::to_string(summary.active_positions_closed) + " positions.",
        10000
    );
    return summary;
}

void MainWindowController::on_equity_update(double equity, double balance) {
    TelemetryDisplayData t = m_telemetry.data();
    t.equity = equity;
    t.balance = balance;
    m_risk.update_equity(equity);
    t.daily_drawdown_pct = m_risk.current_daily_drawdown_pct();
    m_telemetry.update_telemetry(t);
}

} // namespace clow::ui
