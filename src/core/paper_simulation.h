#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include "broker/mt5_bridge.h"
#include "risk/risk_manager.h"
#include "risk/order_state_machine.h"
#include "risk/kill_switch.h"
#include "ui/tactical_radar.h"
#include "ui/autopilot_loop.h"
#include "ui/positions_table.h"
#include "ui/execution_controller.h"

namespace clow::core {

struct MultiPairSimConfig {
    std::vector<std::string> symbols{"EURUSD", "GBPUSD", "USDJPY"};
    int ticks_per_pair{500};
    double initial_balance{50000.0};
    double max_daily_drawdown_pct{4.0};
    double max_account_risk_pct{1.0};
    double min_conviction_threshold{0.70};
};

struct MultiPairSimReport {
    size_t total_ticks_ingested{0};
    size_t inference_signals_evaluated{0};
    size_t orders_dispatched{0};
    size_t positions_filled{0};
    size_t positions_closed{0};
    double starting_equity{0.0};
    double ending_equity{0.0};
    double total_realized_pnl{0.0};
    double max_observed_drawdown_pct{0.0};
    double win_rate_pct{0.0};
    bool zero_gate_breaches{true};
    std::string summary_text;
};

/**
 * @brief End-to-End Multi-Pair Paper Trading Simulation Orchestrator.
 * 
 * Drives full monorepo subsystem integration: MT5 tick feed aggregation,
 * AI inference evaluation, sovereign risk gates, autonomous autopilot dispatches,
 * mark-to-market positions table tracking, and PnL accounting.
 */
class PaperSimulationEngine {
public:
    explicit PaperSimulationEngine(MultiPairSimConfig config = MultiPairSimConfig{});
    ~PaperSimulationEngine() = default;

    /**
     * @brief Executes multi-pair paper simulation run.
     */
    MultiPairSimReport run_simulation();

    [[nodiscard]] const MultiPairSimConfig& config() const noexcept { return m_config; }

private:
    MultiPairSimConfig m_config;
};

} // namespace clow::core
