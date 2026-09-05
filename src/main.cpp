#include <iostream>
#include <string>
#include <vector>
#include "core/config.h"
#include "core/logger.h"

int main(int argc, char* argv[]) {
    CLOW_LOG_INFO("Initializing Clow Quantitative Terminal v0.1.0 (C++20)...");

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--version" || arg == "-v") {
            std::cout << "Clow Terminal v0.1.0" << std::endl;
            return 0;
        }
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: clow_terminal [options]\n"
                      << "Options:\n"
                      << "  -v, --version    Display version information\n"
                      << "  -h, --help       Display this help message\n"
                      << "  --headless       Run in background headless mode\n";
            return 0;
        }
    }

    auto config = clow::core::ClowConfig::load_defaults();
    CLOW_LOG_INFO("Configuration loaded successfully. MT5 Server: " + config.mt5().server);
    CLOW_LOG_INFO("Sovereign Risk Gate active (Max Drawdown: " + 
                  std::to_string(config.risk().max_daily_drawdown_pct) + "%).");

    return 0;
}
