#include "ui/telemetry_header.h"
#include "ui/theme.h"
#include <iomanip>
#include <sstream>

namespace clow::ui {

TelemetryHeader::TelemetryHeader() = default;

void TelemetryHeader::update_telemetry(const TelemetryDisplayData& data) {
    m_data = data;
}

std::string TelemetryHeader::get_status_summary() const {
    std::stringstream ss;
    ss << (m_data.is_connected ? "[CONNECTED] " : "[DISCONNECTED] ")
       << m_data.broker_server << " | "
       << "Equity: $" << std::fixed << std::setprecision(2) << m_data.equity << " | "
       << "Mode: " << m_data.trading_mode << " | "
       << "DD: " << std::setprecision(2) << m_data.daily_drawdown_pct << "%";
    return ss.str();
}

std::string TelemetryHeader::get_connection_badge_color() const {
    if (!m_data.is_connected) {
        return Theme::COLOR_BEARISH_RED;
    }
    return (m_data.ping_ms < 50.0) ? Theme::COLOR_BULLISH_GREEN : Theme::COLOR_WARNING_AMBER;
}

std::string TelemetryHeader::get_trading_mode_badge_color() const {
    if (m_data.trading_mode == "LIVE") {
        return Theme::COLOR_BULLISH_GREEN;
    } else if (m_data.trading_mode == "PAPER") {
        return Theme::COLOR_ACCENT_CYAN;
    }
    return Theme::COLOR_BEARISH_RED;
}

} // namespace clow::ui
