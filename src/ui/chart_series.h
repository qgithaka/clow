#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace clow::ui {

struct CandleBar {
    int64_t timestamp_ms{0};
    double open{0.0};
    double high{0.0};
    double low{0.0};
    double close{0.0};
    double volume{0.0};

    [[nodiscard]] bool is_bullish() const noexcept {
        return close >= open;
    }
};

struct ViewportTransform {
    double min_price{1.0};
    double max_price{1.1};
    size_t visible_bars_start{0};
    size_t visible_bars_count{50};
    int canvas_width_px{1200};
    int canvas_height_px{700};
    double bar_width_px{12.0};
    double bar_spacing_px{4.0};

    [[nodiscard]] double price_to_y(double price) const noexcept {
        if (max_price <= min_price) return static_cast<double>(canvas_height_px) / 2.0;
        double normalized = (price - min_price) / (max_price - min_price);
        // Canvas Y=0 is top, Y=height is bottom
        return static_cast<double>(canvas_height_px) * (1.0 - normalized);
    }

    [[nodiscard]] double y_to_price(double y) const noexcept {
        if (canvas_height_px <= 0) return min_price;
        double normalized = 1.0 - (y / static_cast<double>(canvas_height_px));
        return min_price + normalized * (max_price - min_price);
    }

    [[nodiscard]] double index_to_x(size_t viewport_bar_idx) const noexcept {
        return static_cast<double>(viewport_bar_idx) * (bar_width_px + bar_spacing_px) + (bar_width_px / 2.0);
    }
};

/**
 * @brief Candlestick time-series buffer and viewport bounds manager.
 */
class ChartSeries {
public:
    explicit ChartSeries(size_t max_bars = 2000);
    ~ChartSeries() = default;

    void set_bars(const std::vector<CandleBar>& bars);
    void push_bar(const CandleBar& bar);
    void update_latest_tick(double bid, double ask, int64_t timestamp_ms);

    [[nodiscard]] size_t size() const noexcept { return m_bars.size(); }
    [[nodiscard]] bool empty() const noexcept { return m_bars.empty(); }
    [[nodiscard]] const CandleBar& get_bar(size_t index) const;
    [[nodiscard]] const std::vector<CandleBar>& bars() const noexcept { return m_bars; }

    /**
     * @brief Computes dynamic price range for visible bar window with padding.
     */
    void compute_viewport_bounds(size_t start_idx, size_t count, double& out_min, double& out_max, double padding_pct = 0.05) const;

private:
    size_t m_max_bars;
    std::vector<CandleBar> m_bars;
};

} // namespace clow::ui
