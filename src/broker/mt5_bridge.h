#pragma once

#include <string>
#include <vector>
#include <optional>
#include <chrono>

namespace clow::broker {

struct AccountInfo {
    int64_t login = 0;
    std::string server = "";
    std::string name = "";
    std::string company = "";
    std::string currency = "USD";
    double balance = 0.0;
    double equity = 0.0;
    double margin = 0.0;
    double free_margin = 0.0;
    double leverage = 100.0;
    std::string trade_mode = "Demo";
    bool is_connected = false;
};

struct SymbolInfo {
    std::string name = "";
    std::string currency_base = "EUR";
    std::string currency_profit = "USD";
    int digits = 5;
    double point = 0.00001;
    double spread = 1.0;
    double volume_min = 0.01;
    double volume_max = 100.0;
    double volume_step = 0.01;
};

struct TickQuote {
    std::string symbol = "";
    int64_t timestamp_utc = 0;
    double bid = 0.0;
    double ask = 0.0;
    double last = 0.0;
    double spread_pips = 0.0;
};

class MT5Bridge {
public:
    MT5Bridge() = default;
    ~MT5Bridge() = default;

    bool initialize(int64_t login, const std::string& password, const std::string& server);
    void shutdown() noexcept;

    [[nodiscard]] bool is_connected() const noexcept { return is_connected_; }
    [[nodiscard]] std::optional<AccountInfo> get_account_info() const;
    [[nodiscard]] std::vector<SymbolInfo> get_symbol_catalog() const;
    [[nodiscard]] std::optional<TickQuote> get_live_tick(const std::string& symbol) const;

private:
    bool is_connected_ = false;
    AccountInfo current_account_;
};

} // namespace clow::broker
