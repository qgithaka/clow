#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include "risk/position_sizer.h"

namespace clow::risk {

enum class TradingMode {
    Disabled,   // Default: live execution strictly prohibited
    Paper,      // Simulated paper execution
    Live        // Sovereign authorized live broker execution
};

struct AccountState {
    double balance{10000.0};
    double equity{10000.0};
    double initial_day_equity{10000.0};
    double realized_daily_pnl{0.0};
    double unrealized_pnl{0.0};
    int open_positions_count{0};
    int pending_orders_count{0};
    double current_spread_pips{1.2};
};

struct OrderProposal {
    std::string symbol{"EURUSD"};
    std::string order_type{"BUY_LIMIT"};
    double current_price{1.0850};
    double entry_price{1.0840};
    double stop_loss{1.0820};
    double take_profit{1.0880};
    double win_probability{0.60};
    double expected_value{0.0015};
    int expiration_bars{3};
};

struct RiskDecision {
    bool approved{false};
    double approved_lot_size{0.0};
    double approved_risk_pct{0.0};
    std::string triggered_gate;
    std::string rejection_reason;
};

/**
 * @brief Sovereign risk governance layer.
 * 
 * Enforces strict risk gates (drawdown limits, spread limits, concurrent trades, kill-switch status)
 * before any order can be dispatched to execution bridges.
 */
class RiskManager {
public:
    RiskManager(
        double max_daily_drawdown_pct = 4.0,
        double max_account_risk_pct = 1.0,
        int max_open_trades = 3,
        double max_spread_pips = 2.5,
        double min_confidence = 0.55
    );

    ~RiskManager() = default;

    /**
     * @brief Evaluates an incoming order proposal against all sovereign risk gates.
     * @param proposal Incoming candidate order proposal.
     * @param account Current account telemetry snapshot.
     * @param symbol_spec Symbol contract specification.
     * @return RiskDecision struct detailing approval or gate rejection.
     */
    RiskDecision evaluate_order(
        const OrderProposal& proposal,
        const AccountState& account,
        const SymbolSpecification& symbol_spec
    );

    /**
     * @brief Updates daily equity baseline and calculates intraday drawdown.
     */
    void update_equity(double equity);

    /**
     * @brief Resets intraday daily drawdown and PnL trackers (e.g. at UTC midnight).
     */
    void reset_daily_stats(double starting_equity);

    /**
     * @brief Halts all new trade authorizations.
     */
    void halt_trading(const std::string& reason);

    /**
     * @brief Resumes trading if safety conditions are satisfied.
     */
    void resume_trading();

    [[nodiscard]] bool is_halted() const noexcept { return m_halted; }
    [[nodiscard]] TradingMode trading_mode() const noexcept { return m_mode; }
    [[nodiscard]] const std::string& halt_reason() const noexcept { return m_halt_reason; }
    [[nodiscard]] double current_daily_drawdown_pct() const noexcept { return m_current_daily_drawdown_pct; }

    void set_trading_mode(TradingMode mode) noexcept { m_mode = mode; }
    void set_max_daily_drawdown_pct(double pct) noexcept { m_max_daily_drawdown_pct = pct; }
    void set_max_spread_pips(double pips) noexcept { m_max_spread_pips = pips; }
    void set_max_open_trades(int count) noexcept { m_max_open_trades = count; }

private:
    double m_max_daily_drawdown_pct;
    double m_max_account_risk_pct;
    int m_max_open_trades;
    double m_max_spread_pips;
    double m_min_confidence;

    TradingMode m_mode{TradingMode::Disabled};
    bool m_halted{false};
    std::string m_halt_reason;

    double m_day_start_equity{10000.0};
    double m_peak_daily_equity{10000.0};
    double m_current_daily_drawdown_pct{0.0};

    PositionSizer m_sizer;
};

} // namespace clow::risk
