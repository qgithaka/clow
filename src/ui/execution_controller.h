#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include "broker/mt5_bridge.h"
#include "risk/order_state_machine.h"
#include "risk/risk_manager.h"
#include "risk/kill_switch.h"
#include "ui/positions_table.h"

namespace clow::ui {

/**
 * @brief High-level Execution Controller for single-click order and position management.
 * 
 * Coordinates actions between the MT5 IPC Bridge, Risk State Machine,
 * Sovereign Risk Manager, and the Live Positions Dock Table.
 */
class ExecutionController {
public:
    ExecutionController(
        clow::broker::MT5Bridge& bridge,
        clow::risk::OrderStateMachine& state_machine,
        clow::risk::RiskManager& risk_mgr,
        clow::risk::KillSwitch& kill_switch,
        PositionsTable& positions_table
    );
    ~ExecutionController() = default;

    /**
     * @brief Single-click close position by broker ticket.
     */
    bool close_position(int64_t ticket);

    /**
     * @brief Single-click cancel pending order by broker ticket.
     */
    bool cancel_pending_order(int64_t ticket);

    /**
     * @brief Modifies Stop Loss and Take Profit levels for an open position or order.
     */
    bool modify_protection(int64_t ticket, double new_sl, double new_tp);

    /**
     * @brief Triggers instant emergency panic liquidation of all positions and orders.
     */
    clow::risk::PanicLiquidationSummary panic_liquidate(const std::string& reason = "User Terminal Panic Liquidation");

    /**
     * @brief Synchronizes PositionsTable with the current OrderStateMachine and MT5 quotes.
     */
    void sync_state();

private:
    clow::broker::MT5Bridge& m_bridge;
    clow::risk::OrderStateMachine& m_state_machine;
    clow::risk::RiskManager& m_risk_mgr;
    clow::risk::KillSwitch& m_kill_switch;
    PositionsTable& m_positions_table;
};

} // namespace clow::ui
