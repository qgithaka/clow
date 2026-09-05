#pragma once

#include <cmath>
#include <cstddef>
#include <string>

namespace clow::risk {

enum class SizingMethod {
    FixedFractional,    // Fixed percentage of account equity
    HalfKelly,          // Half-Kelly optimal growth sizing
    QuarterKelly,       // Conservative quarter-Kelly sizing
    FixedLot            // Fixed static lot size
};

struct SymbolSpecification {
    std::string symbol{"EURUSD"};
    double contract_size{100000.0}; // Standard lot units (e.g. 100k for FX)
    double pip_size{0.0001};        // 0.0001 for EURUSD, 0.01 for USDJPY
    double pip_value_per_lot{10.0}; // $10 per pip per 1.0 standard lot in USD
    double min_lot{0.01};
    double max_lot{50.0};
    double lot_step{0.01};
};

struct SizingResult {
    double lot_size{0.01};
    double cash_risk{0.0};
    double calculated_risk_pct{0.0};
    double raw_kelly_fraction{0.0};
    bool valid{false};
    std::string rejection_reason;
};

/**
 * @brief Dynamic institutional lot sizing engine.
 * 
 * Computes exact position volume given account equity, stop distance, win probability,
 * and risk-reward ratio, enforcing hard lot boundaries and quantization steps.
 */
class PositionSizer {
public:
    PositionSizer(
        SizingMethod method = SizingMethod::FixedFractional,
        double max_risk_pct = 1.0,
        double max_lot_cap = 20.0
    );

    ~PositionSizer() = default;

    /**
     * @brief Calculates optimal trade volume.
     * @param equity Current account equity.
     * @param stop_distance_price Stop loss distance in absolute price units (e.g. 0.0020 for 20 pips).
     * @param symbol_spec Specifications of traded symbol.
     * @param win_prob Model 1 directional win probability (required for Kelly sizing).
     * @param risk_reward_ratio Planned reward/risk ratio (required for Kelly sizing).
     * @return SizingResult containing validated lot size and risk metrics.
     */
    SizingResult calculate_lots(
        double equity,
        double stop_distance_price,
        const SymbolSpecification& symbol_spec,
        double win_prob = 0.55,
        double risk_reward_ratio = 1.5
    ) const;

    /**
     * @brief Rounds raw lot size to legal broker increment step (e.g. 0.01).
     */
    static double quantize_lot(double raw_lot, double step, double min_lot, double max_lot);

    [[nodiscard]] SizingMethod method() const noexcept { return m_method; }
    [[nodiscard]] double max_risk_pct() const noexcept { return m_max_risk_pct; }

    void set_method(SizingMethod method) noexcept { m_method = method; }
    void set_max_risk_pct(double pct) noexcept { m_max_risk_pct = pct; }

private:
    SizingMethod m_method;
    double m_max_risk_pct;
    double m_max_lot_cap;
};

} // namespace clow::risk
