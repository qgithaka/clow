#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace clow::ui {

enum class ToastType {
    Info,
    Success,
    Warning,
    Error,
    Critical
};

const char* toast_type_to_string(ToastType type) noexcept;

struct ToastNotification {
    int64_t id{0};
    ToastType type{ToastType::Info};
    std::string title;
    std::string message;
    uint64_t timestamp_ms{0};
    int duration_ms{4000};
    bool dismissed{false};
};

/**
 * @brief Toast and Notification queue manager for trade executions and risk alerts.
 */
class ToastManager {
public:
    ToastManager();
    ~ToastManager() = default;

    int64_t show_toast(ToastType type, const std::string& title, const std::string& message, int duration_ms = 4000);

    void dismiss_toast(int64_t toast_id);
    void clear_all();

    [[nodiscard]] std::vector<ToastNotification> active_toasts() const;
    [[nodiscard]] size_t total_history_count() const noexcept { return m_all_toasts.size(); }

private:
    int64_t m_next_id{1};
    std::vector<ToastNotification> m_all_toasts;
};

} // namespace clow::ui
