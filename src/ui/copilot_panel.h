#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include "risk/risk_manager.h"
#include "risk/order_state_machine.h"

namespace clow::ui {

enum class CopilotAction {
    PendingReview,
    Approved,
    Rejected,
    Expired
};

const char* copilot_action_to_string(CopilotAction action) noexcept;

struct CopilotProposalCard {
    int64_t proposal_id{0};
    clow::risk::OrderProposal proposal;
    double calculated_lot_size{0.01};
    double risk_pct{1.0};
    double reward_risk_ratio{1.5};
    double estimated_risk_usd{100.0};
    double estimated_reward_usd{150.0};
    int64_t created_timestamp_ms{0};
    CopilotAction action{CopilotAction::PendingReview};
    std::string rejection_reason;
    int64_t client_order_id{0};
};

using CopilotSubmissionHandler = std::function<int64_t(const clow::risk::OrderProposal& proposal, double volume)>;
using CopilotActionCallback = std::function<void(const CopilotProposalCard& card)>;

/**
 * @brief Co-Pilot Mode workflow coordinator.
 * 
 * Presents AI-generated order proposals for human trader verification with
 * one-click [Approve] / [Reject] actions, risk breakdowns, and execution dispatch.
 */
class CopilotPanel {
public:
    CopilotPanel();
    ~CopilotPanel() = default;

    /**
     * @brief Queues a newly generated tactical order proposal for human review.
     * @return Unique proposal ID.
     */
    int64_t queue_proposal(
        const clow::risk::OrderProposal& proposal,
        double lot_size,
        double risk_pct,
        double account_balance = 10000.0
    );

    /**
     * @brief Approves proposal and dispatches order to the execution engine.
     */
    bool approve_proposal(int64_t proposal_id);

    /**
     * @brief Rejects proposal with human trader justification.
     */
    bool reject_proposal(int64_t proposal_id, const std::string& reason = "Rejected by human trader");

    /**
     * @brief Sets callback handler for approved order execution submission.
     */
    void set_submission_handler(CopilotSubmissionHandler handler);

    /**
     * @brief Sets listener callback for proposal approval/rejection events.
     */
    void set_action_callback(CopilotActionCallback callback);

    [[nodiscard]] const CopilotProposalCard* get_proposal(int64_t proposal_id) const;
    [[nodiscard]] std::vector<CopilotProposalCard> get_pending_reviews() const;
    [[nodiscard]] std::vector<CopilotProposalCard> get_all_proposals() const;
    [[nodiscard]] size_t pending_count() const noexcept;
    [[nodiscard]] size_t total_count() const noexcept { return m_proposals.size(); }

    /**
     * @brief Formats human-readable order review card summary.
     */
    [[nodiscard]] std::string format_card_summary(int64_t proposal_id) const;

    void clear_all() noexcept;

private:
    int64_t m_next_proposal_id{50001};
    std::unordered_map<int64_t, CopilotProposalCard> m_proposals;
    std::vector<int64_t> m_proposal_order;
    CopilotSubmissionHandler m_submission_handler;
    CopilotActionCallback m_action_callback;
};

} // namespace clow::ui
