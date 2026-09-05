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

struct ExecutionResponse {
    bool success{false};
    int64_t ticket{0};
    double execution_price{0.0};
    double volume{0.0};
    double realized_pnl{0.0};
    std::string message;
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

    /**
     * @brief Single-click execution methods via direct MT5 Windows IPC bridge.
     */
    ExecutionResponse send_order(const std::string& symbol, const std::string& order_type, double volume, double price, double sl = 0.0, double tp = 0.0);
    ExecutionResponse close_position(int64_t ticket, const std::string& symbol = "", double volume = 0.0);
    ExecutionResponse cancel_order(int64_t ticket, const std::string& symbol = "");
    ExecutionResponse modify_order(int64_t ticket, double new_sl, double new_tp);

private:
    bool is_connected_ = false;
    AccountInfo current_account_;
    int64_t next_ticket_{90001};
};

} // namespace clow::broker
