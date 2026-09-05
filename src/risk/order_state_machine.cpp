#include "risk/order_state_machine.h"
#include "core/logger.h"

namespace clow::risk {

const char* order_state_to_string(OrderState state) noexcept {
    switch (state) {
        case OrderState::Created:   return "CREATED";
        case OrderState::Submitted: return "SUBMITTED";
        case OrderState::Pending:   return "PENDING";
        case OrderState::Filled:    return "FILLED";
        case OrderState::Expired:   return "EXPIRED";
        case OrderState::Cancelled: return "CANCELLED";
        case OrderState::Rejected:  return "REJECTED";
        case OrderState::Closed:    return "CLOSED";
        default:                    return "UNKNOWN";
    }
}

OrderStateMachine::OrderStateMachine() = default;

void OrderStateMachine::set_state_change_callback(StateChangeCallback cb) {
    m_callback = std::move(cb);
}

bool OrderStateMachine::is_valid_transition(OrderState from, OrderState to) {
    if (from == to) return true;

    switch (from) {
        case OrderState::Created:
            return (to == OrderState::Submitted || to == OrderState::Rejected || to == OrderState::Cancelled);

        case OrderState::Submitted:
            return (to == OrderState::Pending || to == OrderState::Filled || to == OrderState::Rejected || to == OrderState::Cancelled);

        case OrderState::Pending:
            return (to == OrderState::Filled || to == OrderState::Expired || to == OrderState::Cancelled || to == OrderState::Rejected);

        case OrderState::Filled:
            return (to == OrderState::Closed);

        case OrderState::Expired:
        case OrderState::Cancelled:
        case OrderState::Rejected:
        case OrderState::Closed:
            return false; // Terminal states

        default:
            return false;
    }
}

int64_t OrderStateMachine::create_order(const OrderProposal& proposal, double volume) {
    int64_t id = m_next_client_id++;
    
    ManagedOrder order;
    order.client_id = id;
    order.symbol = proposal.symbol;
    order.order_type = proposal.order_type;
    order.entry_price = proposal.entry_price;
    order.stop_loss = proposal.stop_loss;
    order.take_profit = proposal.take_profit;
    order.volume = volume;
    order.max_expiration_bars = std::max(1, proposal.expiration_bars);
    order.bars_elapsed = 0;
    order.state = OrderState::Created;

    m_orders[id] = order;

    CLOW_LOG_INFO("Created order #" + std::to_string(id) + " for " + order.symbol +
                  " [" + order.order_type + "] Vol=" + std::to_string(order.volume) +
                  " Timeout=" + std::to_string(order.max_expiration_bars) + " bars");

    return id;
}

bool OrderStateMachine::transition_to(int64_t client_id, OrderState new_state, const std::string& reason) {
    auto it = m_orders.find(client_id);
    if (it == m_orders.end()) {
        CLOW_LOG_ERROR("Order #" + std::to_string(client_id) + " not found in state machine");
        return false;
    }

    ManagedOrder& order = it->second;
    OrderState old_state = order.state;

    if (!is_valid_transition(old_state, new_state)) {
        CLOW_LOG_ERROR("Illegal state transition for order #" + std::to_string(client_id) +
                       " from " + order_state_to_string(old_state) +
                       " to " + order_state_to_string(new_state));
        return false;
    }

    order.state = new_state;
    order.status_reason = reason;

    CLOW_LOG_INFO("Order #" + std::to_string(client_id) + " state changed: " +
                  order_state_to_string(old_state) + " -> " + order_state_to_string(new_state) +
                  (reason.empty() ? "" : " (" + reason + ")"));

    if (m_callback) {
        m_callback(order, old_state, new_state);
    }

    return true;
}

std::vector<int64_t> OrderStateMachine::on_bar_tick(const std::string& symbol) {
    std::vector<int64_t> expired_ids;

    for (auto& [id, order] : m_orders) {
        if ((order.state == OrderState::Pending || order.state == OrderState::Submitted) &&
            order.symbol == symbol) {
            order.bars_elapsed++;
            if (order.bars_elapsed >= order.max_expiration_bars) {
                transition_to(id, OrderState::Expired, "Order reached timeout horizon of " +
                              std::to_string(order.max_expiration_bars) + " bars");
                expired_ids.push_back(id);
            }
        }
    }

    return expired_ids;
}

void OrderStateMachine::on_order_fill(int64_t client_id, int64_t broker_ticket, double fill_price) {
    auto it = m_orders.find(client_id);
    if (it == m_orders.end()) return;

    it->second.broker_ticket = broker_ticket;
    it->second.fill_price = fill_price;
    transition_to(client_id, OrderState::Filled, "Broker fill confirmed @ " + std::to_string(fill_price));
}

void OrderStateMachine::on_order_close(int64_t client_id, double close_price, double realized_pnl) {
    auto it = m_orders.find(client_id);
    if (it == m_orders.end()) return;

    it->second.close_price = close_price;
    it->second.realized_pnl = realized_pnl;
    transition_to(client_id, OrderState::Closed, "Position closed @ " + std::to_string(close_price) +
                  " PnL=" + std::to_string(realized_pnl));
}

const ManagedOrder* OrderStateMachine::get_order(int64_t client_id) const {
    auto it = m_orders.find(client_id);
    if (it != m_orders.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<ManagedOrder> OrderStateMachine::get_pending_orders() const {
    std::vector<ManagedOrder> list;
    for (const auto& [id, order] : m_orders) {
        if (order.state == OrderState::Pending || order.state == OrderState::Submitted) {
            list.push_back(order);
        }
    }
    return list;
}

std::vector<ManagedOrder> OrderStateMachine::get_active_positions() const {
    std::vector<ManagedOrder> list;
    for (const auto& [id, order] : m_orders) {
        if (order.state == OrderState::Filled) {
            list.push_back(order);
        }
    }
    return list;
}

size_t OrderStateMachine::cancel_all_pending(const std::string& reason) {
    size_t count = 0;
    for (auto& [id, order] : m_orders) {
        if (order.state == OrderState::Pending || order.state == OrderState::Submitted || order.state == OrderState::Created) {
            if (transition_to(id, OrderState::Cancelled, reason)) {
                count++;
            }
        }
    }
    return count;
}

} // namespace clow::risk
