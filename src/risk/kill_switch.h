#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include "risk/order_state_machine.h"
#include "risk/risk_manager.h"

namespace clow::risk {

struct PanicLiquidationSummary {
    size_t pending_orders_cancelled{0};
    size_t active_positions_closed{0};
    double total_estimated_pnl{0.0};
    bool execution_success{false};
    std::string trigger_reason;
};

using PositionCloseExecutor = std::function<bool(int64_t broker_ticket, const std::string& symbol, double volume)>;

/**
 * @brief Sovereign Instant Panic Kill-Switch.
 * 
 * Provides atomic, one-touch emergency liquidation: cancels all pending orders,
 * closes all open market positions via broker bridge, locks risk gates, and audits the event.
 */
class KillSwitch {
public:
    KillSwitch(RiskManager& risk_mgr, OrderStateMachine& state_machine);
    ~KillSwitch() = default;

    /**
     * @brief Registers broker position closure execution handler.
     */
    void set_broker_close_handler(PositionCloseExecutor handler);

    /**
     * @brief Executes instant panic shutdown and liquidation.
     * @param reason Sovereign emergency reason.
     * @return PanicLiquidationSummary report.
     */
    PanicLiquidationSummary trigger_panic(const std::string& reason = "Manual Sovereign Panic Kill-Switch Triggered");

    /**
     * @brief Checks if kill-switch is currently active.
     */
    [[nodiscard]] bool is_active() const noexcept;

    /**
     * @brief Sovereign authorization to clear panic state after risk assessment.
     */
    void clear_panic();

private:
    RiskManager& m_risk_mgr;
    OrderStateMachine& m_state_machine;
    PositionCloseExecutor m_close_handler;
    bool m_panic_active{false};
    std::string m_last_panic_reason;
};

} // namespace clow::risk
