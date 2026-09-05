#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include "ui/chart_series.h"

namespace clow::ui {

enum class TimeframePeriod {
    M1 = 60000,        // 60,000 ms
    M5 = 300000,       // 300,000 ms
    M15 = 900000,      // 900,000 ms
    H1 = 3600000,      // 3,600,000 ms
    H4 = 14400000,     // 14,400,000 ms
    D1 = 86400000      // 86,400,000 ms
};

/**
 * @brief Thread-safe high-frequency tick aggregator and bar rollup engine.
 */
class ChartFeed {
public:
    explicit ChartFeed(TimeframePeriod tf = TimeframePeriod::M5);
    ~ChartFeed() = default;

    void set_timeframe(TimeframePeriod tf);
    [[nodiscard]] TimeframePeriod timeframe() const noexcept { return m_timeframe; }

    /**
     * @brief Ingests an incoming raw broker tick and updates or rolls the candlestick series.
     * @param symbol Asset symbol.
     * @param bid Current bid price.
     * @param ask Current ask price.
     * @param timestamp_ms Current tick timestamp in ms.
     * @param series Destination chart series receiving the candle updates.
     * @return true if a new bar closed and opened on this tick.
     */
    bool on_tick(
        const std::string& symbol,
        double bid,
        double ask,
        int64_t timestamp_ms,
        ChartSeries& series
    );

    [[nodiscard]] int64_t current_bar_open_time() const noexcept { return m_current_bar_open_time; }
    [[nodiscard]] double current_price() const noexcept { return m_last_price; }

private:
    TimeframePeriod m_timeframe;
    int64_t m_current_bar_open_time{0};
    double m_last_price{0.0};
    std::mutex m_mutex;
};

} // namespace clow::ui
