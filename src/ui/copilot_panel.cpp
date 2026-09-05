#include "ui/copilot_panel.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace clow::ui {

const char* copilot_action_to_string(CopilotAction action) noexcept {
    switch (action) {
        case CopilotAction::PendingReview: return "PENDING_REVIEW";
        case CopilotAction::Approved: return "APPROVED";
        case CopilotAction::Rejected: return "REJECTED";
        case CopilotAction::Expired: return "EXPIRED";
        default: return "UNKNOWN";
    }
}

CopilotPanel::CopilotPanel() = default;

void CopilotPanel::set_submission_handler(CopilotSubmissionHandler handler) {
    m_submission_handler = std::move(handler);
}

void CopilotPanel::set_action_callback(CopilotActionCallback callback) {
    m_action_callback = std::move(callback);
}

int64_t CopilotPanel::queue_proposal(
    const clow::risk::OrderProposal& proposal,
    double lot_size,
    double risk_pct,
    double account_balance
) {
    int64_t pid = m_next_proposal_id++;
    CopilotProposalCard card;
    card.proposal_id = pid;
    card.proposal = proposal;
    card.calculated_lot_size = lot_size;
    card.risk_pct = risk_pct;

    double stop_dist = std::abs(proposal.entry_price - proposal.stop_loss);
    double target_dist = std::abs(proposal.take_profit - proposal.entry_price);
    if (stop_dist > 1e-6) {
        card.reward_risk_ratio = target_dist / stop_dist;
    } else {
        card.reward_risk_ratio = 1.0;
    }

    card.estimated_risk_usd = account_balance * (risk_pct / 100.0);
    card.estimated_reward_usd = card.estimated_risk_usd * card.reward_risk_ratio;
    card.action = CopilotAction::PendingReview;

    m_proposals[pid] = card;
    m_proposal_order.push_back(pid);
    return pid;
}

bool CopilotPanel::approve_proposal(int64_t proposal_id) {
    auto it = m_proposals.find(proposal_id);
    if (it == m_proposals.end()) {
        return false;
    }

    if (it->second.action != CopilotAction::PendingReview) {
        return false;
    }

    it->second.action = CopilotAction::Approved;

    if (m_submission_handler) {
        it->second.client_order_id = m_submission_handler(it->second.proposal, it->second.calculated_lot_size);
    }

    if (m_action_callback) {
        m_action_callback(it->second);
    }

    return true;
}

bool CopilotPanel::reject_proposal(int64_t proposal_id, const std::string& reason) {
    auto it = m_proposals.find(proposal_id);
    if (it == m_proposals.end()) {
        return false;
    }

    if (it->second.action != CopilotAction::PendingReview) {
        return false;
    }

    it->second.action = CopilotAction::Rejected;
    it->second.rejection_reason = reason;

    if (m_action_callback) {
        m_action_callback(it->second);
    }

    return true;
}

const CopilotProposalCard* CopilotPanel::get_proposal(int64_t proposal_id) const {
    auto it = m_proposals.find(proposal_id);
    if (it != m_proposals.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<CopilotProposalCard> CopilotPanel::get_pending_reviews() const {
    std::vector<CopilotProposalCard> pending;
    for (int64_t pid : m_proposal_order) {
        auto it = m_proposals.find(pid);
        if (it != m_proposals.end() && it->second.action == CopilotAction::PendingReview) {
            pending.push_back(it->second);
        }
    }
    return pending;
}

std::vector<CopilotProposalCard> CopilotPanel::get_all_proposals() const {
    std::vector<CopilotProposalCard> all;
    for (int64_t pid : m_proposal_order) {
        auto it = m_proposals.find(pid);
        if (it != m_proposals.end()) {
            all.push_back(it->second);
        }
    }
    return all;
}

size_t CopilotPanel::pending_count() const noexcept {
    size_t count = 0;
    for (const auto& [_, card] : m_proposals) {
        if (card.action == CopilotAction::PendingReview) {
            ++count;
        }
    }
    return count;
}

std::string CopilotPanel::format_card_summary(int64_t proposal_id) const {
    auto it = m_proposals.find(proposal_id);
    if (it == m_proposals.end()) {
        return "Co-Pilot: Proposal ID not found.";
    }

    const auto& c = it->second;
    const auto& p = c.proposal;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "[CO-PILOT #" << c.proposal_id << "] " << p.symbol << " " << p.order_type << " "
        << std::setprecision(2) << c.calculated_lot_size << " Lots | Entry: "
        << std::setprecision(5) << p.entry_price << " | SL: " << p.stop_loss << " | TP: " << p.take_profit
        << " | Risk: $" << std::setprecision(2) << c.estimated_risk_usd << " (" << c.risk_pct << "%)"
        << " | Reward: $" << c.estimated_reward_usd << " (R:R 1:" << c.reward_risk_ratio << ")"
        << " | State: " << copilot_action_to_string(c.action);
    return oss.str();
}

void CopilotPanel::clear_all() noexcept {
    m_proposals.clear();
    m_proposal_order.clear();
}

} // namespace clow::ui
