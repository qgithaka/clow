#include "ui/ghost_candle_overlay.h"
#include "ui/theme.h"
#include <algorithm>
#include <cmath>

namespace clow::ui {

GhostCandleOverlay::GhostCandleOverlay() = default;

void GhostCandleOverlay::update_forecast(
    double latest_close,
    double current_atr,
    const ai::ForecasterPrediction& prediction
) {
    if (!prediction.valid || latest_close <= 0.0) {
        m_ghost.active = false;
        return;
    }

    double atr = (current_atr > 0.0) ? current_atr : (latest_close * 0.0015);

    m_ghost.open = latest_close;
    m_ghost.win_probability = static_cast<double>(prediction.direction_prob);
    m_ghost.is_bullish = (prediction.direction_prob >= 0.5f);

    // High and Low excursions from median quantile (index 1 if 3 quantiles, else 0)
    size_t q_mid_idx = prediction.quantiles_high.size() >= 2 ? 1 : 0;
    double high_exc = static_cast<double>(prediction.quantiles_high.empty() ? 1.0f : prediction.quantiles_high[q_mid_idx]) * atr;
    double low_exc = static_cast<double>(prediction.quantiles_low.empty() ? 1.0f : prediction.quantiles_low[q_mid_idx]) * atr;

    m_ghost.high = m_ghost.open + high_exc;
    m_ghost.low = m_ghost.open - low_exc;

    // Body ratio from anatomy[0]
    double body_ratio = 0.5;
    if (prediction.anatomy.size() >= 1) {
        body_ratio = std::clamp(static_cast<double>(prediction.anatomy[0]), 0.1, 0.9);
    }

    double full_range = m_ghost.high - m_ghost.low;
    if (m_ghost.is_bullish) {
        m_ghost.close = m_ghost.open + (full_range * body_ratio * 0.7);
        m_ghost.glow_color_hex = Theme::COLOR_BULLISH_GREEN;
    } else {
        m_ghost.close = m_ghost.open - (full_range * body_ratio * 0.7);
        m_ghost.glow_color_hex = Theme::COLOR_BEARISH_RED;
    }

    m_ghost.opacity = 0.70;
    m_ghost.active = true;
}

GhostRenderRect GhostCandleOverlay::compute_render_geometry(
    const ViewportTransform& transform,
    size_t next_bar_index
) const {
    GhostRenderRect rect;
    if (!m_ghost.active) return rect;

    double x = transform.index_to_x(next_bar_index);
    double y_open = transform.price_to_y(m_ghost.open);
    double y_close = transform.price_to_y(m_ghost.close);
    double y_high = transform.price_to_y(m_ghost.high);
    double y_low = transform.price_to_y(m_ghost.low);

    double top_y = std::min(y_open, y_close);
    double bot_y = std::max(y_open, y_close);
    double height = std::max(2.0, bot_y - top_y);

    rect.body_x = x - (transform.bar_width_px / 2.0);
    rect.body_y = top_y;
    rect.body_width = transform.bar_width_px;
    rect.body_height = height;

    rect.wick_x = x;
    rect.wick_top_y = y_high;
    rect.wick_bottom_y = y_low;

    rect.stroke_color_hex = m_ghost.glow_color_hex;
    rect.fill_color_hex = m_ghost.glow_color_hex;

    return rect;
}

} // namespace clow::ui
