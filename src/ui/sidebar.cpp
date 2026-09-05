#include "ui/sidebar.h"
#include <algorithm>

namespace clow::ui {

SidebarController::SidebarController() = default;

bool SidebarController::set_symbol(const std::string& symbol) {
    auto it = std::find(m_state.available_symbols.begin(), m_state.available_symbols.end(), symbol);
    if (it != m_state.available_symbols.end()) {
        m_state.active_symbol = symbol;
        return true;
    }
    return false;
}

bool SidebarController::set_timeframe(const std::string& timeframe) {
    auto it = std::find(m_state.available_timeframes.begin(), m_state.available_timeframes.end(), timeframe);
    if (it != m_state.available_timeframes.end()) {
        m_state.active_timeframe = timeframe;
        return true;
    }
    return false;
}

bool SidebarController::set_model(const std::string& model_id) {
    auto it = std::find(m_state.available_models.begin(), m_state.available_models.end(), model_id);
    if (it != m_state.available_models.end()) {
        m_state.active_model = model_id;
        return true;
    }
    return false;
}

bool SidebarController::set_execution_mode(const std::string& mode) {
    if (mode == "DISABLED" || mode == "PAPER" || mode == "LIVE") {
        m_state.execution_mode = mode;
        return true;
    }
    return false;
}

void SidebarController::set_risk_per_trade(double risk_pct) {
    m_state.risk_per_trade_pct = std::clamp(risk_pct, 0.1, 5.0);
}

void SidebarController::set_max_open_trades(int max_trades) {
    m_state.max_open_trades = std::clamp(max_trades, 1, 20);
}

void SidebarController::set_max_spread(double max_spread) {
    m_state.max_spread_filter = std::clamp(max_spread, 0.5, 20.0);
}

} // namespace clow::ui
