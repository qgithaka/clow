#pragma once

#include <string>
#include <vector>

namespace clow::core {

struct MT5Config {
    std::string server = "MetaQuotes-Demo";
    int64_t login = 0;
    std::string password = "";
    std::string terminal_path = "";
    int timeout_seconds = 30;
};

struct RiskConfig {
    double max_account_risk_pct = 1.0;
    double max_daily_drawdown_pct = 4.0;
    int max_open_trades = 3;
    double max_spread_pips = 2.5;
    double min_confidence_threshold = 0.65;
    int default_expiration_bars = 3;
};

struct AIConfig {
    std::string model1_path = "models/clow_forecaster_m5.onnx";
    std::string model2_path = "models/clow_tactical_v1.onnx";
    int context_length = 64;
    int num_threads = 4;
};

class ClowConfig {
public:
    ClowConfig() = default;

    static ClowConfig load_defaults();
    static ClowConfig load_from_file(const std::string& filepath);

    [[nodiscard]] const MT5Config& mt5() const noexcept { return mt5_; }
    [[nodiscard]] const RiskConfig& risk() const noexcept { return risk_; }
    [[nodiscard]] const AIConfig& ai() const noexcept { return ai_; }

    void set_mt5(const MT5Config& cfg) { mt5_ = cfg; }
    void set_risk(const RiskConfig& cfg) { risk_ = cfg; }
    void set_ai(const AIConfig& cfg) { ai_ = cfg; }

private:
    MT5Config mt5_;
    RiskConfig risk_;
    AIConfig ai_;
};

} // namespace clow::core
