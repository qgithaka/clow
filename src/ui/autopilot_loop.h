#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include "ui/tactical_radar.h"
#include "risk/risk_manager.h"
#include "risk/order_state_machine.h"

namespace clow::ui {

struct AutopilotConfig {
    double min_confidence{0.70};
    double min_reward_risk_ratio{1.5};
    int max_orders_per_hour{10};
    bool require_active_trading_mode{true};
};

struct AutopilotTriggerEvent {
    int64_t event_id{0};
    int64_t timestamp_ms{0};
    std::string symbol;
    double confidence{0.0};
    bool threshold_passed{false};
    bool risk_approved{false};
    int64_t client_order_id{0};
    double executed_lot_size{0.0};
    std::string status_message;
};

using AutopilotEventCallback = std::function<void(const AutopilotTriggerEvent& event)>;

/**
 * @brief Autonomous Auto-Pilot execution workflow controller.
 * 
 * Ingests tactical radar signals, verifies high-conviction threshold (> 70%),
 * passes setups through sovereign risk gates, and atomically dispatches pending
 * orders to the order state machine.
 */
class AutopilotLoop {
public:
    AutopilotLoop(
        clow::risk::RiskManager& risk_mgr,
        clow::risk::OrderStateMachine& state_machine,
        AutopilotConfig config = AutopilotConfig{}
    );
    ~AutopilotLoop() = default;

    /**
     * @brief Evaluates an incoming radar prediction for autonomous execution.
     */
    AutopilotTriggerEvent process_radar_signal(
        const RadarPrediction& prediction,
        const clow::risk::AccountState& account,
        const clow::risk::SymbolSpecification& symbol_spec,
        int64_t current_timestamp_ms = 0
    );

    void set_enabled(bool enabled) noexcept { m_enabled = enabled; }
    [[nodiscard]] bool is_enabled() const noexcept { return m_enabled; }

    void set_config(const AutopilotConfig& config) noexcept { m_config = config; }
    [[nodiscard]] const AutopilotConfig& config() const noexcept { return m_config; }

    void set_event_callback(AutopilotEventCallback cb) { m_event_callback = std::move(cb); }

    [[nodiscard]] const std::vector<AutopilotTriggerEvent>& trigger_history() const noexcept { return m_history; }
    [[nodiscard]] size_t total_dispatches_count() const noexcept { return m_dispatches_count; }

    void reset_history() noexcept;

private:
    clow::risk::RiskManager& m_risk_mgr;
    clow::risk::OrderStateMachine& m_state_machine;
    AutopilotConfig m_config;
    bool m_enabled{false};
    int64_t m_next_event_id{1};
    size_t m_dispatches_count{0};
    std::vector<AutopilotTriggerEvent> m_history;
    std::vector<int64_t> m_recent_dispatch_timestamps;
    AutopilotEventCallback m_event_callback;

    bool check_rate_limit(int64_t now_ms);
};

} // namespace clow::ui
