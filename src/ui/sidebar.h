#pragma once

#include <string>
#include <vector>

namespace clow::ui {

struct SidebarState {
    std::string active_symbol{"EURUSD"};
    std::string active_timeframe{"M5"};
    std::string active_model{"clow_forecaster_v1"};
    std::string execution_mode{"PAPER"}; // DISABLED, PAPER, LIVE
    double risk_per_trade_pct{1.0};
    int max_open_trades{3};
    double max_spread_filter{2.5};

    std::vector<std::string> available_symbols{"EURUSD", "GBPUSD", "USDJPY", "XAUUSD"};
    std::vector<std::string> available_timeframes{"M1", "M5", "M15", "H1", "H4", "D1"};
    std::vector<std::string> available_models{"clow_forecaster_v1", "chronos_bolt_v1", "tactical_policy_v1"};
};

/**
 * @brief Left Navigation & Control Sidebar presenter.
 */
class SidebarController {
public:
    SidebarController();
    ~SidebarController() = default;

    [[nodiscard]] const SidebarState& state() const noexcept { return m_state; }

    bool set_symbol(const std::string& symbol);
    bool set_timeframe(const std::string& timeframe);
    bool set_model(const std::string& model_id);
    bool set_execution_mode(const std::string& mode);
    void set_risk_per_trade(double risk_pct);
    void set_max_open_trades(int max_trades);
    void set_max_spread(double max_spread);

private:
    SidebarState m_state;
};

} // namespace clow::ui
