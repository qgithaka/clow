#pragma once

#include <string>

namespace clow::ui {

struct TelemetryDisplayData {
    std::string broker_server{"MetaQuotes-Demo"};
    std::string broker_account{"10082910"};
    bool is_connected{false};
    double ping_ms{18.4};
    double balance{10000.0};
    double equity{10000.0};
    double free_margin{9500.0};
    double daily_pnl_pct{0.0};
    double daily_drawdown_pct{0.0};
    std::string trading_mode{"PAPER"}; // DISABLED, PAPER, LIVE
    bool is_kill_switch_active{false};
};

/**
 * @brief Top Telemetry Header presenter and state manager.
 */
class TelemetryHeader {
public:
    TelemetryHeader();
    ~TelemetryHeader() = default;

    void update_telemetry(const TelemetryDisplayData& data);
    [[nodiscard]] const TelemetryDisplayData& data() const noexcept { return m_data; }

    [[nodiscard]] std::string get_status_summary() const;
    [[nodiscard]] std::string get_connection_badge_color() const;
    [[nodiscard]] std::string get_trading_mode_badge_color() const;

private:
    TelemetryDisplayData m_data;
};

} // namespace clow::ui
