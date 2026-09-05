#include "ui/positions_table.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace clow::ui {

void PositionsTable::add_or_update(const PositionRow& row) {
    if (m_rows.find(row.ticket) == m_rows.end()) {
        m_ticket_order.push_back(row.ticket);
    }
    m_rows[row.ticket] = row;
}

bool PositionsTable::remove(int64_t ticket) {
    auto it = m_rows.find(ticket);
    if (it == m_rows.end()) {
        return false;
    }
    m_rows.erase(it);
    m_ticket_order.erase(
        std::remove(m_ticket_order.begin(), m_ticket_order.end(), ticket),
        m_ticket_order.end()
    );
    return true;
}

void PositionsTable::recalculate_row_pnl(PositionRow& row, double bid, double ask) {
    double pip_mult = (row.symbol.find("JPY") != std::string::npos) ? 100.0 : 10000.0;

    if (!row.is_position) {
        row.current_price = (row.order_type.rfind("BUY", 0) == 0) ? ask : bid;
        row.floating_pnl = 0.0;
        row.floating_pips = 0.0;
        return;
    }

    if (row.order_type == "BUY") {
        row.current_price = bid;
        row.floating_pips = (row.current_price - row.open_price) * pip_mult;
        row.floating_pnl = (row.current_price - row.open_price) * (row.volume * 100000.0);
    } else if (row.order_type == "SELL") {
        row.current_price = ask;
        row.floating_pips = (row.open_price - row.current_price) * pip_mult;
        row.floating_pnl = (row.open_price - row.current_price) * (row.volume * 100000.0);
    }
}

void PositionsTable::update_quote(const std::string& symbol, double bid, double ask) {
    for (auto& [ticket, row] : m_rows) {
        if (row.symbol == symbol) {
            recalculate_row_pnl(row, bid, ask);
        }
    }
}

const PositionRow* PositionsTable::get(int64_t ticket) const {
    auto it = m_rows.find(ticket);
    if (it != m_rows.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<PositionRow> PositionsTable::get_all_rows() const {
    std::vector<PositionRow> rows;
    for (int64_t t : m_ticket_order) {
        auto it = m_rows.find(t);
        if (it != m_rows.end()) {
            rows.push_back(it->second);
        }
    }
    return rows;
}

std::vector<PositionRow> PositionsTable::get_open_positions() const {
    std::vector<PositionRow> open_pos;
    for (int64_t t : m_ticket_order) {
        auto it = m_rows.find(t);
        if (it != m_rows.end() && it->second.is_position) {
            open_pos.push_back(it->second);
        }
    }
    return open_pos;
}

std::vector<PositionRow> PositionsTable::get_pending_orders() const {
    std::vector<PositionRow> pending;
    for (int64_t t : m_ticket_order) {
        auto it = m_rows.find(t);
        if (it != m_rows.end() && !it->second.is_position) {
            pending.push_back(it->second);
        }
    }
    return pending;
}

PositionsSummary PositionsTable::get_summary() const {
    PositionsSummary summary;
    for (const auto& [_, row] : m_rows) {
        if (row.is_position) {
            ++summary.open_positions_count;
            summary.total_open_lots += row.volume;
            summary.total_floating_pnl += row.floating_pnl;
            if (row.stop_loss > 0.0) {
                double pip_mult = (row.symbol.find("JPY") != std::string::npos) ? 100.0 : 10000.0;
                summary.total_risk_pips += std::abs(row.open_price - row.stop_loss) * pip_mult;
            }
        } else {
            ++summary.pending_orders_count;
        }
    }
    return summary;
}

std::string PositionsTable::format_table_ascii() const {
    std::ostringstream oss;
    oss << "=================================== LIVE TRADING DOCK ===================================\n";
    oss << std::left << std::setw(10) << "Ticket"
        << std::setw(10) << "Symbol"
        << std::setw(12) << "Type"
        << std::setw(8)  << "Lots"
        << std::setw(12) << "Open Price"
        << std::setw(12) << "Current"
        << std::setw(12) << "SL"
        << std::setw(12) << "TP"
        << std::setw(12) << "P/L ($)"
        << std::setw(10) << "Pips"
        << "\n";
    oss << "-----------------------------------------------------------------------------------------\n";

    for (int64_t t : m_ticket_order) {
        const auto& r = m_rows.at(t);
        oss << std::left << std::setw(10) << r.ticket
            << std::setw(10) << r.symbol
            << std::setw(12) << r.order_type
            << std::fixed << std::setprecision(2) << std::setw(8) << r.volume
            << std::setprecision(5) << std::setw(12) << r.open_price
            << std::setw(12) << r.current_price
            << std::setw(12) << r.stop_loss
            << std::setw(12) << r.take_profit
            << std::setprecision(2) << std::setw(12) << r.floating_pnl
            << std::setprecision(1) << std::setw(10) << r.floating_pips
            << "\n";
    }

    auto summary = get_summary();
    oss << "-----------------------------------------------------------------------------------------\n";
    oss << "Open Positions: " << summary.open_positions_count
        << " | Pending Orders: " << summary.pending_orders_count
        << " | Total Volume: " << std::fixed << std::setprecision(2) << summary.total_open_lots << " Lots"
        << " | Floating P/L: $" << std::setprecision(2) << summary.total_floating_pnl << "\n";
    oss << "=========================================================================================\n";

    return oss.str();
}

void PositionsTable::clear() noexcept {
    m_rows.clear();
    m_ticket_order.clear();
}

} // namespace clow::ui
