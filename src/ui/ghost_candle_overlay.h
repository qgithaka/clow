#pragma once

#include "ai/onnx_engine.h"
#include "ui/chart_series.h"
#include <string>

namespace clow::ui {

struct GhostCandle {
    double open{0.0};
    double high{0.0};
    double low{0.0};
    double close{0.0};
    double win_probability{0.50};
    bool is_bullish{true};
    std::string glow_color_hex{"#00E676"};
    double opacity{0.65};
    bool active{false};
};

struct GhostRenderRect {
    double body_x{0.0};
    double body_y{0.0};
    double body_width{0.0};
    double body_height{0.0};
    double wick_x{0.0};
    double wick_top_y{0.0};
    double wick_bottom_y{0.0};
    std::string stroke_color_hex;
    std::string fill_color_hex;
};

/**
 * @brief Predictive Ghost-Candle overlay renderer for upcoming candle bar.
 */
class GhostCandleOverlay {
public:
    GhostCandleOverlay();
    ~GhostCandleOverlay() = default;

    /**
     * @brief Computes predicted candle geometry from latest candle close and neural forecast.
     */
    void update_forecast(
        double latest_close,
        double current_atr,
        const ai::ForecasterPrediction& prediction
    );

    [[nodiscard]] const GhostCandle& ghost_candle() const noexcept { return m_ghost; }
    [[nodiscard]] bool is_active() const noexcept { return m_ghost.active; }

    /**
     * @brief Computes canvas pixel coordinates for rendering the ghost candle.
     * @param transform Active chart viewport coordinate transformer.
     * @param next_bar_index Next bar index on the canvas (e.g. visible_count).
     */
    [[nodiscard]] GhostRenderRect compute_render_geometry(
        const ViewportTransform& transform,
        size_t next_bar_index
    ) const;

private:
    GhostCandle m_ghost;
};

} // namespace clow::ui
