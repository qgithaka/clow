#include "risk/risk_audit_logger.h"
#include <filesystem>
#include <iostream>

namespace clow::risk {

const char* audit_event_type_to_string(AuditEventType type) noexcept {
    switch (type) {
        case AuditEventType::OrderEvaluated:   return "ORDER_EVALUATED";
        case AuditEventType::OrderApproved:    return "ORDER_APPROVED";
        case AuditEventType::OrderRejected:    return "ORDER_REJECTED";
        case AuditEventType::StateTransition:  return "STATE_TRANSITION";
        case AuditEventType::DrawdownUpdate:   return "DRAWDOWN_UPDATE";
        case AuditEventType::TradingHalted:    return "TRADING_HALTED";
        case AuditEventType::TradingResumed:   return "TRADING_RESUMED";
        case AuditEventType::PanicKillSwitch:  return "PANIC_KILL_SWITCH";
        case AuditEventType::DailyReset:       return "DAILY_RESET";
        default:                               return "UNKNOWN";
    }
}

uint64_t RiskAuditLogger::current_timestamp_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );
}

RiskAuditLogger::RiskAuditLogger(const std::string& log_file_path)
    : m_file_path(log_file_path) {
    try {
        std::filesystem::path p(m_file_path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
        m_file_stream.open(m_file_path, std::ios::app);
    } catch (...) {}
}

RiskAuditLogger::~RiskAuditLogger() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_file_stream.is_open()) {
        m_file_stream.flush();
        m_file_stream.close();
    }
}

void RiskAuditLogger::log_event(const AuditRecord& record) {
    std::lock_guard<std::mutex> lock(m_mutex);

    AuditRecord rec = record;
    if (rec.timestamp_ms == 0) {
        rec.timestamp_ms = current_timestamp_ms();
    }

    m_records.push_back(rec);

    if (m_file_stream.is_open()) {
        m_file_stream << rec.timestamp_ms << " | "
                      << audit_event_type_to_string(rec.event_type) << " | "
                      << rec.symbol << " | Order#"
                      << rec.order_id << " | Gate/State: "
                      << rec.gate_or_state << " | Details: "
                      << rec.details << "\n";
        m_file_stream.flush();
    }
}

void RiskAuditLogger::log_rejection(const std::string& symbol, const std::string& gate, const std::string& reason) {
    AuditRecord rec;
    rec.event_type = AuditEventType::OrderRejected;
    rec.symbol = symbol;
    rec.gate_or_state = gate;
    rec.details = reason;
    log_event(rec);
}

void RiskAuditLogger::log_state_transition(int64_t order_id, const std::string& symbol, const std::string& from, const std::string& to) {
    AuditRecord rec;
    rec.event_type = AuditEventType::StateTransition;
    rec.order_id = order_id;
    rec.symbol = symbol;
    rec.gate_or_state = from + " -> " + to;
    rec.details = "State transition executed";
    log_event(rec);
}

size_t RiskAuditLogger::records_count() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_records.size();
}

std::vector<AuditRecord> RiskAuditLogger::get_records() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_records;
}

void RiskAuditLogger::clear_in_memory() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_records.clear();
}

} // namespace clow::risk
