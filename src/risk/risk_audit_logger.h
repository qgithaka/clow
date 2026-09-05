#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace clow::risk {

enum class AuditEventType {
    OrderEvaluated,
    OrderApproved,
    OrderRejected,
    StateTransition,
    DrawdownUpdate,
    TradingHalted,
    TradingResumed,
    PanicKillSwitch,
    DailyReset
};

const char* audit_event_type_to_string(AuditEventType type) noexcept;

struct AuditRecord {
    uint64_t timestamp_ms{0};
    AuditEventType event_type{AuditEventType::OrderEvaluated};
    std::string symbol;
    int64_t order_id{0};
    std::string gate_or_state;
    std::string details;
    double numeric_metric_1{0.0};
    double numeric_metric_2{0.0};
};

/**
 * @brief Thread-safe immutable risk audit event log manager.
 * 
 * Writes structured chronological records for compliance, governance, and post-mortem analysis.
 */
class RiskAuditLogger {
public:
    explicit RiskAuditLogger(const std::string& log_file_path = "logs/risk_audit.log");
    ~RiskAuditLogger();

    /**
     * @brief Appends an immutable audit event record.
     */
    void log_event(const AuditRecord& record);

    /**
     * @brief Quick helper to log gate rejections.
     */
    void log_rejection(const std::string& symbol, const std::string& gate, const std::string& reason);

    /**
     * @brief Quick helper to log state changes.
     */
    void log_state_transition(int64_t order_id, const std::string& symbol, const std::string& from, const std::string& to);

    /**
     * @brief Returns total recorded audit events count in current session.
     */
    [[nodiscard]] size_t records_count() const;

    /**
     * @brief Retrieves all in-memory audit records.
     */
    [[nodiscard]] std::vector<AuditRecord> get_records() const;

    /**
     * @brief Clears in-memory cache (file log remains preserved).
     */
    void clear_in_memory();

private:
    std::string m_file_path;
    std::ofstream m_file_stream;
    mutable std::mutex m_mutex;
    std::vector<AuditRecord> m_records;

    static uint64_t current_timestamp_ms();
};

} // namespace clow::risk
