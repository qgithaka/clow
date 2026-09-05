#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include "risk/risk_manager.h"

namespace clow::risk {

enum class OrderState {
    Created,
    Submitted,
    Pending,
    Filled,
    Expired,
    Cancelled,
    Rejected,
    Closed
};

const char* order_state_to_string(OrderState state) noexcept;

struct ManagedOrder {
    int64_t client_id{0};
    int64_t broker_ticket{0};
    std::string symbol;
    std::string order_type;
    double entry_price{0.0};
    double stop_loss{0.0};
    double take_profit{0.0};
    double volume{0.0};
    int max_expiration_bars{3};
    int bars_elapsed{0};
    OrderState state{OrderState::Created};
    double fill_price{0.0};
    double close_price{0.0};
    double realized_pnl{0.0};
    std::string status_reason;
};

using StateChangeCallback = std::function<void(const ManagedOrder&, OrderState old_state, OrderState new_state)>;

/**
 * @brief High-precision asynchronous order lifecycle state machine.
 * 
 * Manages atomic transitions between Created, Submitted, Pending, Filled, Expired,
 * Cancelled, and Closed order states with deterministic timeout expiration logic.
 */
class OrderStateMachine {
public:
    OrderStateMachine();
    ~OrderStateMachine() = default;

    /**
     * @brief Registers state transition listener callback.
     */
    void set_state_change_callback(StateChangeCallback cb);

    /**
     * @brief Spawns a newly managed order from an approved proposal.
     * @return Unique client order ID.
     */
    int64_t create_order(const OrderProposal& proposal, double volume);

    /**
     * @brief Executes validated state transition.
     */
    bool transition_to(int64_t client_id, OrderState new_state, const std::string& reason = "");

    /**
     * @brief Increments bar countdown for all pending limit/stop orders and expires stale orders.
     * @param symbol Symbol for which a new candle bar completed.
     * @return List of client order IDs that expired during this tick.
     */
    std::vector<int64_t> on_bar_tick(const std::string& symbol);

    /**
     * @brief Notifies state machine of a successful order fill from the broker.
     */
    void on_order_fill(int64_t client_id, int64_t broker_ticket, double fill_price);

    /**
     * @brief Notifies state machine of a position close and realized PnL.
     */
    void on_order_close(int64_t client_id, double close_price, double realized_pnl);

    [[nodiscard]] const ManagedOrder* get_order(int64_t client_id) const;
    [[nodiscard]] std::vector<ManagedOrder> get_pending_orders() const;
    [[nodiscard]] std::vector<ManagedOrder> get_active_positions() const;
    [[nodiscard]] size_t total_orders_count() const noexcept { return m_orders.size(); }

    /**
     * @brief Cancels all pending orders across all symbols.
     * @return Count of cancelled pending orders.
     */
    size_t cancel_all_pending(const std::string& reason = "Manual/Risk cancellation");

private:
    int64_t m_next_client_id{10001};
    std::unordered_map<int64_t, ManagedOrder> m_orders;
    StateChangeCallback m_callback;

    static bool is_valid_transition(OrderState from, OrderState to);
};

} // namespace clow::risk
