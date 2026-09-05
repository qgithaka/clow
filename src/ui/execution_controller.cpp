#include "ui/execution_controller.h"
#include "core/logger.h"

namespace clow::ui {

ExecutionController::ExecutionController(
    clow::broker::MT5Bridge& bridge,
    clow::risk::OrderStateMachine& state_machine,
    clow::risk::RiskManager& risk_mgr,
    clow::risk::KillSwitch& kill_switch,
    PositionsTable& positions_table
)
    : m_bridge(bridge),
      m_state_machine(state_machine),
      m_risk_mgr(risk_mgr),
      m_kill_switch(kill_switch),
      m_positions_table(positions_table) {}

bool ExecutionController::close_position(int64_t ticket) {
    const auto* row = m_positions_table.get(ticket);
    std::string symbol = row ? row->symbol : "EURUSD";
    double volume = row ? row->volume : 0.01;

    auto resp = m_bridge.close_position(ticket, symbol, volume);
    if (!resp.success) {
        CLOW_LOG_ERROR("Failed to close position #" + std::to_string(ticket) + ": " + resp.message);
        return false;
    }

    if (row && row->client_id > 0) {
        m_state_machine.on_order_close(row->client_id, resp.execution_price, resp.realized_pnl);
    }

    m_positions_table.remove(ticket);
    CLOW_LOG_INFO("Successfully closed position #" + std::to_string(ticket));
    return true;
}

bool ExecutionController::cancel_pending_order(int64_t ticket) {
    const auto* row = m_positions_table.get(ticket);
    std::string symbol = row ? row->symbol : "";

    auto resp = m_bridge.cancel_order(ticket, symbol);
    if (!resp.success) {
        CLOW_LOG_ERROR("Failed to cancel pending order #" + std::to_string(ticket) + ": " + resp.message);
        return false;
    }

    if (row && row->client_id > 0) {
        m_state_machine.transition_to(row->client_id, clow::risk::OrderState::Cancelled, "Cancelled via Terminal Dock");
    }

    m_positions_table.remove(ticket);
    CLOW_LOG_INFO("Successfully cancelled pending order #" + std::to_string(ticket));
    return true;
}

bool ExecutionController::modify_protection(int64_t ticket, double new_sl, double new_tp) {
    auto resp = m_bridge.modify_order(ticket, new_sl, new_tp);
    if (!resp.success) {
        CLOW_LOG_ERROR("Failed to modify protection for ticket #" + std::to_string(ticket) + ": " + resp.message);
        return false;
    }

    const auto* row = m_positions_table.get(ticket);
    if (row) {
        PositionRow updated = *row;
        updated.stop_loss = new_sl;
        updated.take_profit = new_tp;
        m_positions_table.add_or_update(updated);
    }

    return true;
}

clow::risk::PanicLiquidationSummary ExecutionController::panic_liquidate(const std::string& reason) {
    auto summary = m_kill_switch.trigger_panic(reason);
    m_positions_table.clear();
    return summary;
}

void ExecutionController::sync_state() {
    auto active_positions = m_state_machine.get_active_positions();
    for (const auto& pos : active_positions) {
        PositionRow row;
        row.ticket = pos.broker_ticket > 0 ? pos.broker_ticket : pos.client_id;
        row.client_id = pos.client_id;
        row.symbol = pos.symbol;
        row.order_type = pos.order_type;
        row.volume = pos.volume;
        row.open_price = pos.fill_price > 0.0 ? pos.fill_price : pos.entry_price;
        row.current_price = row.open_price;
        row.stop_loss = pos.stop_loss;
        row.take_profit = pos.take_profit;
        row.is_position = true;
        row.status = "OPEN";
        m_positions_table.add_or_update(row);
    }

    auto pending_orders = m_state_machine.get_pending_orders();
    for (const auto& ord : pending_orders) {
        PositionRow row;
        row.ticket = ord.broker_ticket > 0 ? ord.broker_ticket : ord.client_id;
        row.client_id = ord.client_id;
        row.symbol = ord.symbol;
        row.order_type = ord.order_type;
        row.volume = ord.volume;
        row.open_price = ord.entry_price;
        row.current_price = ord.entry_price;
        row.stop_loss = ord.stop_loss;
        row.take_profit = ord.take_profit;
        row.is_position = false;
        row.status = "PENDING";
        m_positions_table.add_or_update(row);
    }
}

} // namespace clow::ui
