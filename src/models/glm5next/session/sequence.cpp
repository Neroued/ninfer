#include "models/glm5next/session/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {

std::uint32_t SessionImpl::window_end(const Sequence& s, std::uint32_t begin,
                                      std::uint32_t end) const {
    end = std::min(end, begin + shape.prefill_chunk);
    for (const auto frontier : s.identity.execution_frontiers)
        if (frontier > begin) {
            end = std::min(end, frontier);
            break;
        }
    if (s.capture && s.next_capture < s.captures.size())
        end = std::min(end, s.captures[s.next_capture].frontier);
    if (s.request->vision)
        end = vision_window.limit(member(s), s.request->vision->units, begin, end,
                                  plan.vision_max_features);
    return end;
}

std::optional<vision::Work> SessionImpl::prepare_visual(const Sequence& s, std::uint32_t begin,
                                                        std::uint32_t end) {
    if (!s.request->vision || begin == end) return {};
    if (const auto* active = vision_window.current();
        active && vision_window.available(member(s))) {
        if (models::vision::embedding_slices(*active->plan, begin, end - begin).empty()) return {};
        return vision::prepare_work(vision_window.reserve(member(s), active->plan),
                                    s.request->vision, plan.states.size());
    }
    auto window = models::vision::plan_window(s.request->vision->units, begin, end,
                                              plan.vision_max_features, plan.states.size());
    if (!window) return {};
    return vision::prepare_work(vision_window.reserve(member(s), std::move(window)),
                                s.request->vision, plan.states.size());
}

std::uint32_t SessionImpl::unit_end(const Sequence& s, ExecutionUnitKind kind,
                                    std::uint32_t tokens) const {
    const auto frontier = s.continuation.metadata.main_frontier;
    switch (kind) {
    case ExecutionUnitKind::Prefill:
        return window_end(s, frontier, s.request->summary.prompt_tokens);
    case ExecutionUnitKind::Replay:
        return window_end(s, frontier, s.replay_target);
    case ExecutionUnitKind::Decode:
        return frontier + std::min({shape.draft_tokens + 1, std::max(tokens, 1U),
                                    shape.max_context - frontier});
    case ExecutionUnitKind::Control:
        return std::min(shape.max_context, frontier + tokens);
    }
    throw std::logic_error("invalid execution unit");
}

void SessionImpl::settle(Sequence& s, context::ContextTransaction& transaction) {
    for (std::uint32_t i = 0; i < pool_count(); ++i) {
        const auto frontier = i == 0 ? s.continuation.metadata.main_frontier
                                     : s.continuation.metadata.mtp_frontier.value();
        const auto retained = i == 0 ? s.recovery_peak : mtp_history_frontier(s.recovery_peak);
        directory.pool(i).histories().settle_growth(s.histories[i], frontier, retained,
                                                    transaction);
    }
    s.permit.reset();
}

void SessionImpl::settle(Sequence& s) {
    auto transaction = directory.begin_transaction();
    settle(s, transaction);
    transaction.prepare_actions();
    if (!actions(transaction).empty())
        throw std::logic_error("settlement outside execution changed a live mapping");
    transaction.mark_submitted();
    transaction.mark_completed();
    transaction.publish();
}

WorkRow SessionImpl::work_row(const Sequence& s, std::uint32_t tokens) const {
    WorkRow row;
    row.member     = member(s);
    row.states     = selectors(s);
    row.main_table = directory.pool(0).histories().execution_row(s.histories[0].handle()).handle();
    const auto& m  = s.continuation.metadata;
    row.main_frontier     = m.main_frontier;
    row.attention_rank    = m.placement.value;
    row.sampling          = s.request->sampling;
    row.sampling_position = m.main_frontier;
    const auto start      = m.main_frontier;
    if (start > s.identity.tokens.size() || tokens > s.identity.tokens.size() - start)
        throw std::logic_error("execution input exceeds committed identity");
    row.ids.assign(s.identity.tokens.begin() + start, s.identity.tokens.begin() + start + tokens);
    return row;
}

void SessionImpl::prepare_write(Sequence& s, std::uint32_t end,
                                context::ContextTransaction& transaction) {
    if (s.write_destination.valid()) transaction.retain(directory.states(), s.continuation.image);
    for (std::uint32_t i = 0; i < pool_count(); ++i) {
        const auto begin = i == 0 ? s.continuation.metadata.main_frontier
                                  : s.continuation.metadata.mtp_frontier.value();
        directory.pool(i).histories().prepare_write(
            s.histories[i], begin, i == 0 ? end : mtp_history_frontier(end), transaction);
    }
}

ContinuationMetadata SessionImpl::prepare_commit(Sequence& s, const WorkRow& row,
                                                 std::span<const TokenId> inputs,
                                                 bool retain_recovery,
                                                 context::ContextTransaction& transaction) {
    const auto tokens = inputs.size();
    auto metadata     = s.continuation.metadata;
    if (tokens) {
        metadata.main_frontier = row.main_frontier + tokens;
        if (pool_count() > 1) metadata.mtp_frontier = mtp_history_frontier(metadata.main_frontier);
    }
    const auto peak = retain_recovery ? s.recovery_peak : 0;
    for (std::uint32_t i = 0; i < pool_count(); ++i)
        directory.pool(i).histories().prepare_commit(
            s.histories[i], i ? *metadata.mtp_frontier : metadata.main_frontier,
            i ? mtp_history_frontier(peak) : peak, transaction);
    return metadata;
}

void SessionImpl::adopt_commit(Sequence& s, const ContinuationMetadata& metadata,
                               bool state_written, bool retain_recovery) noexcept {
    if (state_written) complete_write(s);
    s.continuation.metadata = metadata;
    if (!retain_recovery) s.recovery_peak = 0;
    s.permit.reset();
}

} // namespace ninfer::models::glm5next::detail
