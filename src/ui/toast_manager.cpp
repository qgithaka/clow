#include "ui/toast_manager.h"
#include <chrono>

namespace clow::ui {

const char* toast_type_to_string(ToastType type) noexcept {
    switch (type) {
        case ToastType::Info:     return "INFO";
        case ToastType::Success:  return "SUCCESS";
        case ToastType::Warning:  return "WARNING";
        case ToastType::Error:    return "ERROR";
        case ToastType::Critical: return "CRITICAL";
        default:                  return "UNKNOWN";
    }
}

ToastManager::ToastManager() = default;

int64_t ToastManager::show_toast(ToastType type, const std::string& title, const std::string& message, int duration_ms) {
    int64_t id = m_next_id++;
    uint64_t now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );

    ToastNotification toast;
    toast.id = id;
    toast.type = type;
    toast.title = title;
    toast.message = message;
    toast.timestamp_ms = now_ms;
    toast.duration_ms = duration_ms;
    toast.dismissed = false;

    m_all_toasts.push_back(toast);
    return id;
}

void ToastManager::dismiss_toast(int64_t toast_id) {
    for (auto& toast : m_all_toasts) {
        if (toast.id == toast_id) {
            toast.dismissed = true;
            break;
        }
    }
}

void ToastManager::clear_all() {
    for (auto& toast : m_all_toasts) {
        toast.dismissed = true;
    }
}

std::vector<ToastNotification> ToastManager::active_toasts() const {
    std::vector<ToastNotification> active;
    for (const auto& toast : m_all_toasts) {
        if (!toast.dismissed) {
            active.push_back(toast);
        }
    }
    return active;
}

} // namespace clow::ui
