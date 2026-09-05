#include "ui/chart_series.h"

namespace clow::ui {

ChartSeries::ChartSeries(size_t max_bars)
    : m_max_bars(std::max(size_t{100}, max_bars)) {
    m_bars.reserve(m_max_bars);
}

void ChartSeries::set_bars(const std::vector<CandleBar>& bars) {
    if (bars.size() <= m_max_bars) {
        m_bars = bars;
    } else {
        m_bars.assign(bars.end() - static_cast<std::ptrdiff_t>(m_max_bars), bars.end());
    }
}

void ChartSeries::push_bar(const CandleBar& bar) {
    if (m_bars.size() >= m_max_bars) {
        m_bars.erase(m_bars.begin());
    }
    m_bars.push_back(bar);
}

void ChartSeries::update_latest_tick(double bid, double ask, int64_t timestamp_ms) {
    if (m_bars.empty()) {
        double mid = (bid + ask) / 2.0;
        push_bar(CandleBar{timestamp_ms, mid, mid, mid, mid, 1.0});
        return;
    }

    double price = (bid + ask) / 2.0;
    CandleBar& latest = m_bars.back();
    latest.close = price;
    latest.high = std::max(latest.high, price);
    latest.low = std::min(latest.low, price);
    latest.volume += 1.0;
    latest.timestamp_ms = timestamp_ms;
}

const CandleBar& ChartSeries::get_bar(size_t index) const {
    static const CandleBar empty_bar{};
    if (index < m_bars.size()) {
        return m_bars[index];
    }
    return empty_bar;
}

void ChartSeries::compute_viewport_bounds(
    size_t start_idx,
    size_t count,
    double& out_min,
    double& out_max,
    double padding_pct
) const {
    if (m_bars.empty()) {
        out_min = 1.0;
        out_max = 1.1;
        return;
    }

    size_t start = std::min(start_idx, m_bars.size() - 1);
    size_t end = std::min(start + count, m_bars.size());

    double min_p = m_bars[start].low;
    double max_p = m_bars[start].high;

    for (size_t i = start; i < end; ++i) {
        min_p = std::min(min_p, m_bars[i].low);
        max_p = std::max(max_p, m_bars[i].high);
    }

    double span = max_p - min_p;
    if (span <= 0.0) span = min_p * 0.001; // Fallback tiny span

    out_min = min_p - span * padding_pct;
    out_max = max_p + span * padding_pct;
}

} // namespace clow::ui
