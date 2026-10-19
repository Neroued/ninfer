#include "models/qwen4_exp/session/internal.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
namespace {
using Kind = detail::ComputeTransaction::Kind;

void submit(detail::SessionImpl& session, runtime::ExecutionReservation reservation, Work work) {
    auto& tx = *session.compute;
    tx.context.prepare_actions();
    work.actions = tx.context.actions();
    tx.ticket    = session.executor->submit(std::move(reservation), std::move(work));
    tx.context.mark_submitted();
}

std::optional<PlePrefetch> prefetch(const detail::SessionImpl& p, const detail::Sequence& s,
                                    std::uint32_t begin, std::uint32_t limit) {
    const auto end = p.window_end(s, begin, limit);
    if (end == begin) return {};
    const auto eos = p.definition->config().text.ple.eos;
    return PlePrefetch{p.member(s),
                       begin,
                       {begin >= 2 ? s.identity.tokens[begin - 2] : eos,
                        begin ? s.identity.tokens[begin - 1] : eos},
                       {s.identity.tokens.begin() + begin, s.identity.tokens.begin() + end}};
}
} // namespace

std::optional<runtime::ExecutionTicket> Session::start_prefill(SequenceHandle handle,
                                                               bool constrained) {
    auto& p = *impl_;
    auto& s = p.sequence(handle);
    if (p.compute) return {};
    if (s.phase != detail::Phase::Prefill || !s.permit || s.capture_pending)
        throw std::logic_error("prefill has no executable unit");
    const auto frontier = s.continuation.metadata.main_frontier;
    const auto end      = p.window_end(s, frontier, s.request->summary.prompt_tokens);
    const bool capture  = s.capture && s.request->captures[s.next_capture].frontier == end;
    const bool sample   = end == s.request->summary.prompt_tokens && !capture;
    auto reservation    = p.executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    p.compute = std::make_unique<detail::ComputeTransaction>(Kind::Prefill, p.directory);
    auto& tx  = *p.compute;
    tx.members.push_back(handle);
    try {
        if (end > frontier) p.prepare_write(s, end, tx.context);
        auto row              = p.work_row(s, end - frontier);
        row.constrained       = constrained;
        row.sampling_position = end;
        tx.rows.push_back(row);
        ForwardWork forward;
        forward.kind   = ForwardKind::Prefill;
        forward.sample = sample ? SampleKind::FirstToken : SampleKind::None;
        forward.rows.push_back(std::move(row));
        forward.prefetch = prefetch(p, s, end, s.request->summary.prompt_tokens);
        submit(p, std::move(*reservation), {{}, std::move(forward)});
        return tx.ticket;
    } catch (const std::exception& error) {
        p.executor->fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket> Session::start_replay(SequenceHandle handle) {
    auto& p = *impl_;
    auto& s = p.sequence(handle);
    if (p.compute) return {};
    if (s.phase != detail::Phase::Replay || !s.permit || s.capture_pending)
        throw std::logic_error("replay has no executable unit");
    const auto begin = s.continuation.metadata.main_frontier;
    const auto end   = p.window_end(s, begin, s.replay_target);
    if (end <= begin) throw std::logic_error("replay has no missing input window");
    auto reservation = p.executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    p.compute = std::make_unique<detail::ComputeTransaction>(Kind::Replay, p.directory);
    auto& tx  = *p.compute;
    tx.members.push_back(handle);
    try {
        if (end > begin) p.prepare_write(s, end, tx.context);
        tx.rows.push_back(p.work_row(s, end - begin));
        ForwardWork forward;
        forward.kind     = ForwardKind::Replay;
        forward.rows     = tx.rows;
        forward.prefetch = prefetch(p, s, end, s.replay_target);
        submit(p, std::move(*reservation), {{}, std::move(forward)});
        return tx.ticket;
    } catch (const std::exception& error) {
        p.executor->fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket>
Session::start_decode(std::span<const SequenceHandle> members,
                      std::span<const runtime::RoundBudget> budgets,
                      std::span<const bool> constrained) {
    auto& p = *impl_;
    if (p.compute) return {};
    if (members.empty() || members.size() > p.shape.max_concurrency ||
        budgets.size() != members.size() ||
        (!constrained.empty() && constrained.size() != members.size()))
        throw std::logic_error("invalid decode members");
    std::vector<std::uint32_t> extents(members.size());
    bool speculative = false;
    for (std::size_t b = 0; b < members.size(); ++b) {
        const auto& s = p.sequence(members[b]);
        const auto f  = s.continuation.metadata.main_frontier;
        if (s.phase != detail::Phase::Active || !s.permit ||
            !budgets[b].generated_tokens_remaining || s.identity.tokens.size() != f + 1 ||
            std::find(members.begin(), members.begin() + b, members[b]) != members.begin() + b)
            throw std::logic_error("decode row is not executable");
        extents[b] = std::min({p.shape.draft_tokens, budgets[b].generated_tokens_remaining - 1,
                               p.shape.max_context - f - 1});
        speculative |= extents[b] != 0;
    }
    auto reservation = p.executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    p.compute = std::make_unique<detail::ComputeTransaction>(
        speculative ? Kind::Mtp : Kind::Ordinary, p.directory);
    auto& tx = *p.compute;
    tx.members.assign(members.begin(), members.end());
    tx.extents = extents;
    try {
        for (std::size_t b = 0; b < members.size(); ++b) {
            auto& s        = p.sequence(members[b]);
            const auto end = s.continuation.metadata.main_frontier + extents[b] + 1;
            if (end > s.permitted_frontier)
                throw std::logic_error("decode exceeds its reservation");
            p.prepare_write(s, end, tx.context);
            auto row        = p.work_row(s, 1);
            row.constrained = !constrained.empty() && constrained[b];
            tx.rows.push_back(std::move(row));
        }
        Work work;
        if (speculative)
            work.operation = MtpWork{tx.rows, tx.extents};
        else
            work.operation = ForwardWork{ForwardKind::Ordinary, SampleKind::NextToken, tx.rows, {}};
        submit(p, std::move(*reservation), std::move(work));
        return tx.ticket;
    } catch (const std::exception& error) {
        p.executor->fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket>
Session::start_control(std::span<const SequenceHandle> members, std::span<const TokenId> tokens,
                       std::uint32_t stride, std::span<const std::optional<std::uint32_t>> splits) {
    auto& p = *impl_;
    if (p.compute) return {};
    if (members.empty() || members.size() > p.shape.max_concurrency || !stride ||
        tokens.size() != members.size() * stride || splits.size() != members.size())
        throw std::logic_error("invalid forced control window");
    for (std::size_t b = 0; b < members.size(); ++b) {
        const auto& s = p.sequence(members[b]);
        if (s.phase != detail::Phase::Active || !s.permit ||
            s.permit->kind != ExecutionUnitKind::Control ||
            s.identity.tokens.size() != s.continuation.metadata.main_frontier + 1 ||
            s.continuation.metadata.main_frontier + stride > s.permitted_frontier ||
            (splits[b] && (!*splits[b] || *splits[b] > stride)))
            throw std::logic_error("forced control row is not executable");
    }
    for (const auto id : tokens)
        if (id < 0 || std::uint32_t(id) >= p.definition->resources().public_token_count)
            throw std::invalid_argument("forced token is outside vocabulary");
    auto reservation = p.executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    p.compute = std::make_unique<detail::ComputeTransaction>(Kind::Control, p.directory);
    auto& tx  = *p.compute;
    tx.members.assign(members.begin(), members.end());
    tx.splits.assign(splits.begin(), splits.end());
    try {
        for (std::size_t b = 0; b < members.size(); ++b) {
            auto& s = p.sequence(members[b]);
            p.prepare_write(s, s.continuation.metadata.main_frontier + stride, tx.context);
            auto row = p.work_row(s, 1);
            row.count_tokens.assign(tokens.begin() + b * stride, tokens.begin() + (b + 1) * stride);
            row.ids.insert(row.ids.end(), row.count_tokens.begin(), row.count_tokens.end() - 1);
            for (const auto frontier : s.identity.positions.execution_frontiers())
                if (frontier > row.main_frontier && frontier < row.main_frontier + stride)
                    row.window_ends.push_back(frontier - row.main_frontier);
            if (splits[b] && *splits[b] + 1 < stride) row.window_ends.push_back(*splits[b] + 1);
            std::sort(row.window_ends.begin(), row.window_ends.end());
            row.window_ends.erase(std::unique(row.window_ends.begin(), row.window_ends.end()),
                                  row.window_ends.end());
            row.rope_positions.resize(3ULL * stride);
            for (std::uint32_t axis = 0; axis < 3; ++axis)
                for (std::uint32_t j = 0; j < stride; ++j)
                    row.rope_positions[axis * stride + j] = row.main_frontier + j + row.rope_delta;
            tx.rows.push_back(std::move(row));
        }
        submit(p, std::move(*reservation),
               {{}, ForwardWork{ForwardKind::Control, SampleKind::None, tx.rows, {}}});
        return tx.ticket;
    } catch (const std::exception& error) {
        p.executor->fail(error.what());
        throw;
    }
}

} // namespace ninfer::models::qwen4_exp
