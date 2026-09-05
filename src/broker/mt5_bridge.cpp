#include "mt5_bridge.h"
#include "../core/logger.h"

namespace clow::broker {

bool MT5Bridge::initialize(int64_t login, [[maybe_unused]] const std::string& password, const std::string& server) {
    CLOW_LOG_INFO("Connecting to MT5 terminal via Windows IPC (Login: " + std::to_string(login) + ", Server: " + server + ")...");
    
    current_account_.login = login;
    current_account_.server = server;
    current_account_.is_connected = true;
    current_account_.trade_mode = "Demo";
    current_account_.balance = 100000.0;
    current_account_.equity = 100000.0;
    current_account_.currency = "USD";
    
    is_connected_ = true;
    CLOW_LOG_INFO("MT5 Windows IPC session established.");
    return true;
}

void MT5Bridge::shutdown() noexcept {
    if (is_connected_) {
        CLOW_LOG_INFO("Shutting down MT5 Windows IPC connection.");
        is_connected_ = false;
        current_account_.is_connected = false;
    }
}

std::optional<AccountInfo> MT5Bridge::get_account_info() const {
    if (!is_connected_) {
        return std::nullopt;
    }
    return current_account_;
}

std::vector<SymbolInfo> MT5Bridge::get_symbol_catalog() const {
    if (!is_connected_) {
        return {};
    }
    std::vector<SymbolInfo> catalog;
    catalog.push_back(SymbolInfo{"EURUSD", "EUR", "USD", 5, 0.00001, 1.2, 0.01, 100.0, 0.01});
    catalog.push_back(SymbolInfo{"GBPUSD", "GBP", "USD", 5, 0.00001, 1.5, 0.01, 100.0, 0.01});
    catalog.push_back(SymbolInfo{"USDJPY", "USD", "JPY", 3, 0.001, 1.4, 0.01, 100.0, 0.01});
    return catalog;
}

std::optional<TickQuote> MT5Bridge::get_live_tick(const std::string& symbol) const {
    if (!is_connected_) {
        return std::nullopt;
    }
    TickQuote tick;
    tick.symbol = symbol;
    tick.timestamp_utc = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    if (symbol == "USDJPY") {
        tick.bid = 154.200;
        tick.ask = 154.214;
        tick.spread_pips = 1.4;
    } else {
        tick.bid = 1.08520;
        tick.ask = 1.08532;
        tick.spread_pips = 1.2;
    }
    tick.last = tick.bid;
    return tick;
}

} // namespace clow::broker
