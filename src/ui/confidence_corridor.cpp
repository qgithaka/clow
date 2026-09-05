#include "ui/confidence_corridor.h"
#include <cmath>

namespace clow::ui {

ConfidenceCorridor::ConfidenceCorridor() = default;

void ConfidenceCorridor::set_corridor(
    double anchor_price,
    double q10_low,
    double q50_mid,
    double q90_high,
    size_t forecast_steps
) {
    m_anchor_price = anchor_price;
    m_q10_low = q10_low;
    m_q50_mid = q50_mid;
    m_q90_high = q90_high;
    m_forecast_steps = std::max(size_t{1}, forecast_steps);
    m_visible = (m_q90_high >= m_q10_low && m_anchor_price > 0.0);
}

CorridorPolygon ConfidenceCorridor::compute_polygon(
    const ViewportTransform& transform,
    size_t start_bar_index
) const {
    CorridorPolygon poly;
    if (!m_visible) return poly;

    poly.visible = true;
    poly.fill_color_rgba = "rgba(0, 176, 255, 0.15)"; // Cyan translucent corridor

    double x_start = transform.index_to_x(start_bar_index);
    double y_anchor = transform.price_to_y(m_anchor_price);

    poly.upper_points_px.push_back({x_start, y_anchor});
    poly.lower_points_px.push_back({x_start, y_anchor});

    for (size_t step = 1; step <= m_forecast_steps; ++step) {
        double x_step = transform.index_to_x(start_bar_index + step);
        // Cone expands with sqrt of time steps
        double expansion = std::sqrt(static_cast<double>(step));

        double high_p = m_anchor_price + (m_q90_high - m_anchor_price) * expansion;
        double low_p = m_anchor_price - (m_anchor_price - m_q10_low) * expansion;

        poly.upper_points_px.push_back({x_step, transform.price_to_y(high_p)});
        poly.lower_points_px.push_back({x_step, transform.price_to_y(low_p)});
    }

    return poly;
}

} // namespace clow::ui
