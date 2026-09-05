#pragma once

#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include "risk/risk_manager.h"

namespace clow::ui {

enum class RadarDirection {
    Bullish,
    Bearish,
    Neutral
};

const char* radar_direction_to_string(RadarDirection dir) noexcept;

struct RadarPrediction {
    std::string symbol{"EURUSD"};
    std::string timeframe{"M15"};
    double bullish_probability{0.50};
    double bearish_probability{0.50};
    double confidence{0.50};
    RadarDirection direction{RadarDirection::Neutral};
    double forecasted_return{0.0};
    double forecasted_volatility{0.0};
    double quantile_10{0.0};
    double quantile_50{0.0};
    double quantile_90{0.0};
    bool has_tactical_proposal{false};
    clow::risk::OrderProposal proposal;
    int64_t timestamp_ms{0};
};

struct RadarTelemetryState {
    std::string symbol;
    std::string direction_str;
    double confidence_pct{0.0};
    std::string confidence_badge;
    std::string color_hex;
    double target_pips{0.0};
    double stop_pips{0.0};
    double reward_risk_ratio{0.0};
    bool is_high_conviction{false};
    bool is_actionable{false};
};

/**
 * @brief Tactical AI Radar panel model and visualization calculator.
 * 
 * Computes live multi-timeframe directional conviction gauges, excursion targets,
 * signal severity color codes, and tactical order proposal telemetry.
 */
class TacticalRadar {
public:
    explicit TacticalRadar(double high_conviction_threshold = 0.70);
    ~TacticalRadar() = default;

    /**
     * @brief Updates radar with a newly generated AI inference prediction.
     */
    void update_prediction(const RadarPrediction& prediction);

    /**
     * @brief Evaluates current radar telemetry and UI presentation state.
     */
    [[nodiscard]] RadarTelemetryState get_telemetry_state() const;

    [[nodiscard]] const RadarPrediction& current_prediction() const noexcept { return m_current_prediction; }
    [[nodiscard]] bool has_prediction() const noexcept { return m_has_prediction; }
    [[nodiscard]] bool is_high_conviction() const noexcept;
    [[nodiscard]] double conviction_threshold() const noexcept { return m_conviction_threshold; }
    void set_conviction_threshold(double threshold) noexcept { m_conviction_threshold = threshold; }

    /**
     * @brief Generates summary status text for radar UI header.
     */
    [[nodiscard]] std::string format_radar_summary() const;

    /**
     * @brief Formats tactical trade setup breakdown.
     */
    [[nodiscard]] std::string format_tactical_breakdown() const;

private:
    double m_conviction_threshold{0.70};
    RadarPrediction m_current_prediction;
    bool m_has_prediction{false};
};

} // namespace clow::ui
