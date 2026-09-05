#pragma once

#include "risk/kill_switch.h"
#include "risk/order_state_machine.h"
#include "risk/risk_manager.h"
#include "ui/account_manager.h"
#include "ui/sidebar.h"
#include "ui/telemetry_header.h"
#include "ui/theme.h"
#include "ui/toast_manager.h"

#include <memory>
#include <string>

namespace clow::ui {

/**
 * @brief Central Desktop Terminal Controller & Multi-Viewport Coordinator.
 */
class MainWindowController {
public:
    MainWindowController();
    ~MainWindowController() = default;

    [[nodiscard]] TelemetryHeader& telemetry() noexcept { return m_telemetry; }
    [[nodiscard]] SidebarController& sidebar() noexcept { return m_sidebar; }
    [[nodiscard]] AccountManager& accounts() noexcept { return m_accounts; }
    [[nodiscard]] ToastManager& toasts() noexcept { return m_toasts; }
    [[nodiscard]] risk::RiskManager& risk() noexcept { return m_risk; }
    [[nodiscard]] risk::OrderStateMachine& state_machine() noexcept { return m_state_machine; }
    [[nodiscard]] risk::KillSwitch& kill_switch() noexcept { return m_kill_switch; }

    /**
     * @brief Coordinates pair switching and updates sub-components.
     */
    bool switch_symbol(const std::string& symbol);

    /**
     * @brief Coordinates timeframe switching and updates sub-components.
     */
    bool switch_timeframe(const std::string& timeframe);

    /**
     * @brief Handles panic kill-switch button click event from the UI header.
     */
    risk::PanicLiquidationSummary on_panic_button_clicked();

    /**
     * @brief Synchronizes account equity changes into telemetry and risk manager.
     */
    void on_equity_update(double equity, double balance);

private:
    TelemetryHeader m_telemetry;
    SidebarController m_sidebar;
    AccountManager m_accounts;
    ToastManager m_toasts;

    risk::RiskManager m_risk;
    risk::OrderStateMachine m_state_machine;
    risk::KillSwitch m_kill_switch;
};

} // namespace clow::ui
