#pragma once

#include "ui/chart_series.h"
#include <string>
#include <vector>

namespace clow::ui {

struct CorridorPolygon {
    std::vector<std::pair<double, double>> upper_points_px;
    std::vector<std::pair<double, double>> lower_points_px;
    std::string fill_color_rgba;
    bool visible{false};
};

/**
 * @brief Renders the multi-horizon Quantile Confidence Corridor (10th-90th percentile cone).
 */
class ConfidenceCorridor {
public:
    ConfidenceCorridor();
    ~ConfidenceCorridor() = default;

    /**
     * @brief Configures forecasted quantile boundaries.
     * @param anchor_price Latest closing price.
     * @param q10_low 10th percentile lower boundary price.
     * @param q50_mid 50th percentile median forecasted price.
     * @param q90_high 90th percentile upper boundary price.
     * @param forecast_steps Number of future periods (e.g. 1 to 5).
     */
    void set_corridor(
        double anchor_price,
        double q10_low,
        double q50_mid,
        double q90_high,
        size_t forecast_steps = 3
    );

    [[nodiscard]] bool is_visible() const noexcept { return m_visible; }
    [[nodiscard]] double upper_bound() const noexcept { return m_q90_high; }
    [[nodiscard]] double lower_bound() const noexcept { return m_q10_low; }

    /**
     * @brief Computes screen-space polygon vertices for shaded rendering.
     */
    [[nodiscard]] CorridorPolygon compute_polygon(
        const ViewportTransform& transform,
        size_t start_bar_index
    ) const;

private:
    double m_anchor_price{0.0};
    double m_q10_low{0.0};
    double m_q50_mid{0.0};
    double m_q90_high{0.0};
    size_t m_forecast_steps{3};
    bool m_visible{false};
};

} // namespace clow::ui
