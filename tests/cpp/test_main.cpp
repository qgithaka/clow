#include <cassert>
#include <iostream>
#include "../../src/core/config.h"
#include "../../src/core/logger.h"

// Forward declarations of M06 AI test functions
void test_sliding_window_buffer();
void test_feature_normalizer();
void test_onnx_inference_engine_execution();
void test_inference_benchmark_latency_gate();

// Forward declarations of M07 Risk test functions
void test_position_sizer_formulas();
void test_sovereign_risk_gates();
void test_order_state_machine_lifecycle();
void test_kill_switch_panic_liquidation();
void test_risk_audit_logger();

// Forward declarations of M08 Desktop UI test functions
void test_theme_styling();
void test_telemetry_header();
void test_sidebar_controller();
void test_account_manager();
void test_toast_manager();
void test_main_window_controller();

// Forward declarations of M09 Chart Engine test functions
void test_chart_series_and_viewport();
void test_chart_feed_tick_aggregation();
void test_ghost_candle_overlay();
void test_confidence_corridor();
void test_order_canvas_overlay();
void test_chart_rendering_fps_benchmark();

void test_config_defaults() {
    auto cfg = clow::core::ClowConfig::load_defaults();
    assert(cfg.mt5().server == "MetaQuotes-Demo");
    assert(cfg.risk().max_daily_drawdown_pct == 4.0);
    assert(cfg.risk().max_account_risk_pct == 1.0);
    assert(cfg.ai().context_length == 64);
    std::cout << "[PASS] test_config_defaults" << std::endl;
}

void test_logger_emission() {
    CLOW_LOG_INFO("C++ unit test log message verification");
    std::cout << "[PASS] test_logger_emission" << std::endl;
}

int main() {
    std::cout << "Running Clow C++ test suite..." << std::endl;
    
    // Core tests
    test_config_defaults();
    test_logger_emission();

    // M06 AI tests
    test_sliding_window_buffer();
    test_feature_normalizer();
    test_onnx_inference_engine_execution();
    test_inference_benchmark_latency_gate();

    // M07 Risk tests
    test_position_sizer_formulas();
    test_sovereign_risk_gates();
    test_order_state_machine_lifecycle();
    test_kill_switch_panic_liquidation();
    test_risk_audit_logger();

    // M08 Desktop UI tests
    test_theme_styling();
    test_telemetry_header();
    test_sidebar_controller();
    test_account_manager();
    test_toast_manager();
    test_main_window_controller();

    // M09 Chart Engine tests
    test_chart_series_and_viewport();
    test_chart_feed_tick_aggregation();
    test_ghost_candle_overlay();
    test_confidence_corridor();
    test_order_canvas_overlay();
    test_chart_rendering_fps_benchmark();

    std::cout << "All C++ unit tests passed successfully!" << std::endl;
    return 0;
}
