#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace clow::ui {

struct BrokerProfile {
    std::string profile_id{"default_demo"};
    std::string name{"IC Markets Demo"};
    std::string server{"ICMarkets-Demo"};
    int64_t login{10082910};
    std::string terminal_path{"C:/Program Files/MetaTrader 5/terminal64.exe"};
    bool is_live{false};
    bool is_connected{false};
    double balance{10000.0};
    double equity{10000.0};
};

/**
 * @brief Broker Account Manager modal presenter and account switcher.
 */
class AccountManager {
public:
    AccountManager();
    ~AccountManager() = default;

    void add_profile(const BrokerProfile& profile);
    bool select_active_profile(const std::string& profile_id);
    [[nodiscard]] const BrokerProfile* active_profile() const;
    [[nodiscard]] const std::vector<BrokerProfile>& profiles() const noexcept { return m_profiles; }

    void update_connection_status(const std::string& profile_id, bool connected, double balance, double equity);

private:
    std::vector<BrokerProfile> m_profiles;
    std::string m_active_profile_id;
};

} // namespace clow::ui
