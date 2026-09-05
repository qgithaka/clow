#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <mutex>
#include <chrono>
#include <iomanip>

namespace clow::core {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERR,
    CRITICAL
};

class Logger {
public:
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void set_level(LogLevel level) noexcept {
        current_level_ = level;
    }

    void log(LogLevel level, const std::string& message) {
        if (level < current_level_) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::cout << std::put_time(std::gmtime(&in_time_t), "%Y-%m-%d %H:%M:%S UTC")
                  << " [" << level_to_string(level) << "] "
                  << message << std::endl;
    }

private:
    Logger() : current_level_(LogLevel::INFO) {}
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    static const char* level_to_string(LogLevel level) noexcept {
        switch (level) {
            case LogLevel::DEBUG:    return "DEBUG";
            case LogLevel::INFO:     return "INFO";
            case LogLevel::WARNING:  return "WARNING";
            case LogLevel::ERR:      return "ERROR";
            case LogLevel::CRITICAL: return "CRITICAL";
            default:                 return "UNKNOWN";
        }
    }

    LogLevel current_level_;
    std::mutex mutex_;
};

#define CLOW_LOG_DEBUG(msg)    clow::core::Logger::instance().log(clow::core::LogLevel::DEBUG, msg)
#define CLOW_LOG_INFO(msg)     clow::core::Logger::instance().log(clow::core::LogLevel::INFO, msg)
#define CLOW_LOG_WARNING(msg)  clow::core::Logger::instance().log(clow::core::LogLevel::WARNING, msg)
#define CLOW_LOG_ERROR(msg)    clow::core::Logger::instance().log(clow::core::LogLevel::ERR, msg)
#define CLOW_LOG_CRITICAL(msg) clow::core::Logger::instance().log(clow::core::LogLevel::CRITICAL, msg)

} // namespace clow::core
