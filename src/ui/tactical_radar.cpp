#include "ui/tactical_radar.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace clow::ui {

const char* radar_direction_to_string(RadarDirection dir) noexcept {
    switch (dir) {
        case RadarDirection::Bullish: return "BULLISH";
        case RadarDirection::Bearish: return "BEARISH";
        case RadarDirection::Neutral: return "NEUTRAL";
        default: return "UNKNOWN";
    }
}

TacticalRadar::TacticalRadar(double high_conviction_threshold)
    : m_conviction_threshold(high_conviction_threshold) {}

void TacticalRadar::update_prediction(const RadarPrediction& prediction) {
    m_current_prediction = prediction;
    m_has_prediction = true;
}

bool TacticalRadar::is_high_conviction() const noexcept {
    return m_has_prediction && (m_current_prediction.confidence >= m_conviction_threshold);
}

RadarTelemetryState TacticalRadar::get_telemetry_state() const {
    RadarTelemetryState state;
    if (!m_has_prediction) {
        state.symbol = "N/A";
        state.direction_str = "IDLE";
        state.confidence_pct = 0.0;
        state.confidence_badge = "[NO SIGNAL]";
        state.color_hex = "#8A99AD";
        state.is_high_conviction = false;
        state.is_actionable = false;
        return state;
    }

    state.symbol = m_current_prediction.symbol;
    state.direction_str = radar_direction_to_string(m_current_prediction.direction);
    state.confidence_pct = m_current_prediction.confidence * 100.0;

    if (m_current_prediction.direction == RadarDirection::Bullish) {
        state.color_hex = "#00F59B";
    } else if (m_current_prediction.direction == RadarDirection::Bearish) {
        state.color_hex = "#FF3B69";
    } else {
        state.color_hex = "#8A99AD";
    }

    if (m_current_prediction.confidence >= 0.80) {
        state.confidence_badge = "[ULTRA CONVICTION]";
    } else if (m_current_prediction.confidence >= m_conviction_threshold) {
        state.confidence_badge = "[HIGH CONVICTION]";
    } else if (m_current_prediction.confidence >= 0.55) {
        state.confidence_badge = "[MODERATE]";
    } else {
        state.confidence_badge = "[LOW / NOISE]";
    }

    state.is_high_conviction = (m_current_prediction.confidence >= m_conviction_threshold);

    if (m_current_prediction.has_tactical_proposal) {
        const auto& prop = m_current_prediction.proposal;
        double pip_scale = (prop.symbol.find("JPY") != std::string::npos) ? 100.0 : 10000.0;
        state.target_pips = std::abs(prop.take_profit - prop.entry_price) * pip_scale;
        state.stop_pips = std::abs(prop.entry_price - prop.stop_loss) * pip_scale;

        if (state.stop_pips > 1e-5) {
            state.reward_risk_ratio = state.target_pips / state.stop_pips;
        } else {
            state.reward_risk_ratio = 0.0;
        }

        state.is_actionable = state.is_high_conviction && (state.reward_risk_ratio >= 1.2);
    } else {
        state.target_pips = 0.0;
        state.stop_pips = 0.0;
        state.reward_risk_ratio = 0.0;
        state.is_actionable = false;
    }

    return state;
}

std::string TacticalRadar::format_radar_summary() const {
    auto state = get_telemetry_state();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << state.symbol << " | " << state.direction_str << " ("
        << state.confidence_pct << "% " << state.confidence_badge << ")";
    return oss.str();
}

std::string TacticalRadar::format_tactical_breakdown() const {
    if (!m_has_prediction) {
        return "Tactical Radar: No active inference signal available.";
    }

    auto state = get_telemetry_state();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "=== TACTICAL RADAR TELEMETRY ===\n";
    oss << "Symbol: " << state.symbol << " [" << m_current_prediction.timeframe << "]\n";
    oss << "Conviction: " << state.direction_str << " at " << std::setprecision(1) << state.confidence_pct << "% " << state.confidence_badge << "\n";
    oss << std::setprecision(4);
    oss << "Quantiles [10/50/90]: " << m_current_prediction.quantile_10 << " / "
        << m_current_prediction.quantile_50 << " / " << m_current_prediction.quantile_90 << "\n";

    if (m_current_prediction.has_tactical_proposal) {
        const auto& prop = m_current_prediction.proposal;
        oss << "Proposal: " << prop.order_type << " @ " << prop.entry_price
            << " | SL: " << prop.stop_loss << " | TP: " << prop.take_profit << "\n";
        oss << std::setprecision(1);
        oss << "Excursion: Target +" << state.target_pips << " pips | Risk -" << state.stop_pips
            << " pips | R:R 1:" << std::setprecision(2) << state.reward_risk_ratio << "\n";
        oss << "Status: " << (state.is_actionable ? "ACTIONABLE SETUP READY" : "FILTERED BY GATES");
    } else {
        oss << "Proposal: None (Awaiting tactical setup trigger)";
    }

    return oss.str();
}

} // namespace clow::ui
