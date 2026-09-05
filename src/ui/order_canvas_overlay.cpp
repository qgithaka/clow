#include "ui/order_canvas_overlay.h"
#include "ui/theme.h"
#include <cmath>
#include <iomanip>
#include <sstream>

namespace clow::ui {

OrderCanvasOverlay::OrderCanvasOverlay() = default;

void OrderCanvasOverlay::update_orders(const std::vector<risk::ManagedOrder>& active_orders) {
    m_orders = active_orders;
}

std::vector<OrderCanvasLine> OrderCanvasOverlay::compute_lines(
    const ViewportTransform& transform,
    double pip_size
) const {
    std::vector<OrderCanvasLine> lines;
    double p_size = (pip_size > 0.0) ? pip_size : 0.0001;

    for (const auto& order : m_orders) {
        if (order.state == risk::OrderState::Closed ||
            order.state == risk::OrderState::Cancelled ||
            order.state == risk::OrderState::Expired ||
            order.state == risk::OrderState::Rejected) {
            continue;
        }

        // 1. Entry Line
        if (order.entry_price > 0.0) {
            OrderCanvasLine entry_line;
            entry_line.order_id = order.client_id;
            entry_line.line_type = "ENTRY";
            entry_line.price = order.entry_price;
            entry_line.y_px = transform.price_to_y(order.entry_price);
            entry_line.color_hex = Theme::COLOR_ACCENT_CYAN;

            std::stringstream ss;
            ss << "#" << order.client_id << " " << order.order_type << " " << order.volume << "L @ " << std::fixed << std::setprecision(5) << order.entry_price;
            entry_line.badge_text = ss.str();
            lines.push_back(entry_line);
        }

        // 2. Stop Loss Line
        if (order.stop_loss > 0.0) {
            OrderCanvasLine sl_line;
            sl_line.order_id = order.client_id;
            sl_line.line_type = "SL";
            sl_line.price = order.stop_loss;
            sl_line.y_px = transform.price_to_y(order.stop_loss);
            sl_line.color_hex = Theme::COLOR_BEARISH_RED;

            double sl_pips = std::abs(order.entry_price - order.stop_loss) / p_size;
            std::stringstream ss;
            ss << "SL -" << std::fixed << std::setprecision(1) << sl_pips << "p (" << std::setprecision(5) << order.stop_loss << ")";
            sl_line.badge_text = ss.str();
            lines.push_back(sl_line);
        }

        // 3. Take Profit Line
        if (order.take_profit > 0.0) {
            OrderCanvasLine tp_line;
            tp_line.order_id = order.client_id;
            tp_line.line_type = "TP";
            tp_line.price = order.take_profit;
            tp_line.y_px = transform.price_to_y(order.take_profit);
            tp_line.color_hex = Theme::COLOR_BULLISH_GREEN;

            double tp_pips = std::abs(order.take_profit - order.entry_price) / p_size;
            std::stringstream ss;
            ss << "TP +" << std::fixed << std::setprecision(1) << tp_pips << "p (" << std::setprecision(5) << order.take_profit << ")";
            tp_line.badge_text = ss.str();
            lines.push_back(tp_line);
        }
    }

    return lines;
}

} // namespace clow::ui
