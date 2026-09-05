#include <cassert>
#include <iostream>
#include "../../src/core/config.h"
#include "../../src/core/logger.h"

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
    test_config_defaults();
    test_logger_emission();
    std::cout << "All C++ unit tests passed!" << std::endl;
    return 0;
}
