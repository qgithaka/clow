#pragma once

#include "risk/order_state_machine.h"
#include "ui/chart_series.h"
#include <string>
#include <vector>

namespace clow::ui {

struct OrderCanvasLine {
    int64_t order_id{0};
    std::string line_type; // "ENTRY", "SL", "TP"
    double price{0.0};
    double y_px{0.0};
    std::string color_hex;
    std::string badge_text;
};

/**
 * @brief Renders interactive order level lines (Limit Entry, Stop Loss, Take Profit) directly on the chart canvas.
 */
class OrderCanvasOverlay {
public:
    OrderCanvasOverlay();
    ~OrderCanvasOverlay() = default;

    void update_orders(const std::vector<risk::ManagedOrder>& active_orders);

    [[nodiscard]] std::vector<OrderCanvasLine> compute_lines(
        const ViewportTransform& transform,
        double pip_size = 0.0001
    ) const;

    [[nodiscard]] size_t order_count() const noexcept { return m_orders.size(); }

private:
    std::vector<risk::ManagedOrder> m_orders;
};

} // namespace clow::ui
