#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace clow::ui {

struct PositionRow {
    int64_t ticket{0};
    int64_t client_id{0};
    std::string symbol{"EURUSD"};
    std::string order_type{"BUY"};
    double volume{0.01};
    double open_price{0.0};
    double current_price{0.0};
    double stop_loss{0.0};
    double take_profit{0.0};
    double floating_pnl{0.0};
    double floating_pips{0.0};
    bool is_position{true}; // true = open market position, false = pending order
    std::string status{"OPEN"};
    int64_t open_time_utc{0};
};

struct PositionsSummary {
    double total_floating_pnl{0.0};
    double total_open_lots{0.0};
    size_t open_positions_count{0};
    size_t pending_orders_count{0};
    double total_risk_pips{0.0};
};

/**
 * @brief Live MT5 Position and Order Table Dock model.
 * 
 * Aggregates live market positions and pending orders, tracks real-time
 * mark-to-market floating P/L with tick price updates, and provides summary metrics.
 */
class PositionsTable {
public:
    PositionsTable() = default;
    ~PositionsTable() = default;

    /**
     * @brief Adds or updates a position/order record.
     */
    void add_or_update(const PositionRow& row);

    /**
     * @brief Removes a closed position or cancelled order.
     */
    bool remove(int64_t ticket);

    /**
     * @brief Ingests a new live quote for a symbol and updates floating PnL across all holdings.
     */
    void update_quote(const std::string& symbol, double bid, double ask);

    [[nodiscard]] const PositionRow* get(int64_t ticket) const;
    [[nodiscard]] std::vector<PositionRow> get_all_rows() const;
    [[nodiscard]] std::vector<PositionRow> get_open_positions() const;
    [[nodiscard]] std::vector<PositionRow> get_pending_orders() const;

    [[nodiscard]] PositionsSummary get_summary() const;
    [[nodiscard]] size_t total_count() const noexcept { return m_rows.size(); }

    /**
     * @brief Formats an ASCII table representation for telemetry or terminal display.
     */
    [[nodiscard]] std::string format_table_ascii() const;

    void clear() noexcept;

private:
    std::unordered_map<int64_t, PositionRow> m_rows;
    std::vector<int64_t> m_ticket_order;

    static void recalculate_row_pnl(PositionRow& row, double bid, double ask);
};

} // namespace clow::ui
