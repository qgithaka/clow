#include "ui/account_manager.h"
#include <algorithm>

namespace clow::ui {

AccountManager::AccountManager() {
    // Default demo profile
    BrokerProfile demo;
    demo.profile_id = "default_demo";
    demo.name = "MetaQuotes Demo";
    demo.server = "MetaQuotes-Demo";
    demo.login = 10082910;
    demo.is_live = false;
    demo.is_connected = false;
    demo.balance = 10000.0;
    demo.equity = 10000.0;
    m_profiles.push_back(demo);
    m_active_profile_id = demo.profile_id;
}

void AccountManager::add_profile(const BrokerProfile& profile) {
    m_profiles.push_back(profile);
}

bool AccountManager::select_active_profile(const std::string& profile_id) {
    auto it = std::find_if(m_profiles.begin(), m_profiles.end(),
                           [&](const BrokerProfile& p) { return p.profile_id == profile_id; });
    if (it != m_profiles.end()) {
        m_active_profile_id = profile_id;
        return true;
    }
    return false;
}

const BrokerProfile* AccountManager::active_profile() const {
    auto it = std::find_if(m_profiles.begin(), m_profiles.end(),
                           [&](const BrokerProfile& p) { return p.profile_id == m_active_profile_id; });
    if (it != m_profiles.end()) {
        return &(*it);
    }
    return nullptr;
}

void AccountManager::update_connection_status(
    const std::string& profile_id,
    bool connected,
    double balance,
    double equity
) {
    for (auto& profile : m_profiles) {
        if (profile.profile_id == profile_id) {
            profile.is_connected = connected;
            profile.balance = balance;
            profile.equity = equity;
            break;
        }
    }
}

} // namespace clow::ui
