#include "ui/chart_feed.h"

namespace clow::ui {

ChartFeed::ChartFeed(TimeframePeriod tf)
    : m_timeframe(tf) {}

void ChartFeed::set_timeframe(TimeframePeriod tf) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_timeframe = tf;
    m_current_bar_open_time = 0;
}

bool ChartFeed::on_tick(
    const std::string&,
    double bid,
    double ask,
    int64_t timestamp_ms,
    ChartSeries& series
) {
    std::lock_guard<std::mutex> lock(m_mutex);

    double price = (bid + ask) / 2.0;
    m_last_price = price;

    int64_t tf_ms = static_cast<int64_t>(m_timeframe);
    int64_t bar_slot = (timestamp_ms / tf_ms) * tf_ms;

    if (m_current_bar_open_time == 0) {
        m_current_bar_open_time = bar_slot;
        series.push_bar(CandleBar{bar_slot, price, price, price, price, 1.0});
        return false;
    }

    if (bar_slot > m_current_bar_open_time) {
        // New bar opened
        m_current_bar_open_time = bar_slot;
        series.push_bar(CandleBar{bar_slot, price, price, price, price, 1.0});
        return true;
    } else {
        // Update current open bar
        series.update_latest_tick(bid, ask, timestamp_ms);
        return false;
    }
}

} // namespace clow::ui
