#include "risk/position_sizer.h"

#include <algorithm>
#include <cmath>

namespace clow::risk {

PositionSizer::PositionSizer(
    SizingMethod method,
    double max_risk_pct,
    double max_lot_cap
)
    : m_method(method),
      m_max_risk_pct(std::max(0.01, max_risk_pct)),
      m_max_lot_cap(std::max(0.01, max_lot_cap)) {}

double PositionSizer::quantize_lot(double raw_lot, double step, double min_lot, double max_lot) {
    if (step <= 0.0) step = 0.01;
    double steps = std::floor(raw_lot / step);
    double quantized = steps * step;
    return std::clamp(quantized, min_lot, max_lot);
}

SizingResult PositionSizer::calculate_lots(
    double equity,
    double stop_distance_price,
    const SymbolSpecification& symbol_spec,
    double win_prob,
    double risk_reward_ratio
) const {
    SizingResult result;
    result.valid = false;

    if (equity <= 0.0) {
        result.rejection_reason = "Account equity is zero or negative";
        return result;
    }

    if (stop_distance_price <= 0.0 || symbol_spec.pip_size <= 0.0) {
        result.rejection_reason = "Invalid stop loss distance or pip size";
        return result;
    }

    double stop_distance_pips = stop_distance_price / symbol_spec.pip_size;
    if (stop_distance_pips < 0.5) {
        result.rejection_reason = "Stop distance too tight (< 0.5 pips)";
        return result;
    }

    double effective_risk_pct = m_max_risk_pct;

    if (m_method == SizingMethod::HalfKelly || m_method == SizingMethod::QuarterKelly) {
        double b = std::max(0.1, risk_reward_ratio);
        double p = std::clamp(win_prob, 0.0, 1.0);
        double q = 1.0 - p;

        // Kelly criterion: f* = (p * b - q) / b
        double kelly_f = (p * b - q) / b;
        result.raw_kelly_fraction = kelly_f;

        if (kelly_f <= 0.0) {
            result.rejection_reason = "Negative mathematical expectancy (Kelly fraction <= 0)";
            return result;
        }

        double mult = (m_method == SizingMethod::HalfKelly) ? 0.50 : 0.25;
        double dynamic_risk = kelly_f * mult * 100.0; // Convert to percentage
        effective_risk_pct = std::clamp(dynamic_risk, 0.1, m_max_risk_pct);
    } else if (m_method == SizingMethod::FixedLot) {
        result.lot_size = quantize_lot(0.10, symbol_spec.lot_step, symbol_spec.min_lot, symbol_spec.max_lot);
        result.cash_risk = stop_distance_pips * symbol_spec.pip_value_per_lot * result.lot_size;
        result.calculated_risk_pct = (result.cash_risk / equity) * 100.0;
        result.valid = true;
        return result;
    }

    double cash_risk = equity * (effective_risk_pct / 100.0);
    double risk_per_lot = stop_distance_pips * symbol_spec.pip_value_per_lot;

    if (risk_per_lot <= 0.0) {
        result.rejection_reason = "Risk per lot calculation produced non-positive value";
        return result;
    }

    double raw_lot = cash_risk / risk_per_lot;
    double max_allowed = std::min(symbol_spec.max_lot, m_max_lot_cap);
    double final_lot = quantize_lot(raw_lot, symbol_spec.lot_step, symbol_spec.min_lot, max_allowed);

    result.lot_size = final_lot;
    result.cash_risk = stop_distance_pips * symbol_spec.pip_value_per_lot * final_lot;
    result.calculated_risk_pct = (result.cash_risk / equity) * 100.0;
    result.valid = true;

    return result;
}

} // namespace clow::risk
