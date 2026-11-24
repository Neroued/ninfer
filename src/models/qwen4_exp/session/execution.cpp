#include "models/qwen4_exp/session/internal.h"

#include "models/common/speculative/growth.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
namespace {
using Kind = detail::ComputeTransaction::Kind;

void submit(detail::SessionImpl& session, runtime::ExecutionReservation reservation, Work work) {
    auto& tx = *session.compute;
    plan_work_batch(work, *session.plan.batches);
    tx.context.prepare_actions();
    work.actions = session.actions(tx.context);
    tx.ticket    = session.executor->submit(std::move(reservation), std::move(work));
    for (auto h : tx.members) session.sequence(h).computing = true;
    tx.context.mark_submitted();
    if (tx.vision_id) session.vision_window.submitted(tx.vision_id, tx.ticket);
}

std::optional<PlePrefetch> prefetch(const detail::SessionImpl& p, const detail::Sequence& s,
                                    std::uint32_t begin, std::uint32_t limit) {
    const auto end = p.window_end(s, begin, limit);
    if (end == begin) return {};
    const auto eos = p.definition->config().text.ple.eos;
    std::vector<TokenId> tokens;
    s.ledger.copy(begin, end, tokens);
    return PlePrefetch{p.member(s),
                       begin,
                       {begin >= 2 ? s.ledger[begin - 2] : eos, begin ? s.ledger[begin - 1] : eos},
                       std::move(tokens)};
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
    const auto end =
        p.window_end(s, frontier, std::min(s.permitted_frontier, s.request->summary.prompt_tokens));
    const bool capture = s.capture && s.captures[s.next_capture].frontier == end;
    const bool sample  = end == s.request->summary.prompt_tokens && !capture;
    auto reservation   = p.executor->try_reserve(runtime::FrameKind::Compute);
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
        forward.visual = p.prepare_visual(s, frontier, end);
        if (forward.visual) tx.vision_id = forward.visual->window.id;
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
    const auto end   = p.window_end(s, begin, std::min(s.permitted_frontier, s.replay_target));
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
        forward.kind   = ForwardKind::Replay;
        forward.rows   = tx.rows;
        forward.visual = p.prepare_visual(s, begin, end);
        if (forward.visual) tx.vision_id = forward.visual->window.id;
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
                      std::span<const std::uint8_t> constrained) {
    return start_decode_round(members, {.budgets = budgets}, constrained);
}

std::optional<runtime::ExecutionTicket>
Session::start_decode_round(std::span<const SequenceHandle> members,
                            speculative::DecodeRoundRequest selection,
                            std::span<const std::uint8_t> constrained) {
    auto& p = *impl_;
    if (p.compute) return {};
    if (members.empty() || members.size() > p.shape.max_concurrency ||
        !selection.valid_membership(members.size()) ||
        (!constrained.empty() && constrained.size() != members.size()))
        throw std::logic_error("invalid decode members");
    auto reservation = p.executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    auto pending  = std::make_unique<detail::ComputeTransaction>(Kind::Ordinary, p.directory);
    auto& tx      = *pending;
    bool prepared = false;
    try {
        tx.members.assign(members.begin(), members.end());
        const auto& options = p.definition->options();
        const bool adaptive_length =
            p.shape.draft_tokens && selection.uses_adaptive_length(p.shape.length_policy);
        const bool measure_cost =
            p.shape.draft_tokens && (adaptive_length || selection.budgets.empty());
        std::vector<speculative::RoundMember> limits;
        std::vector<speculative::AdaptiveRowInput> adaptive;
        std::vector<std::uint32_t> frontiers;
        limits.reserve(members.size());
        if (adaptive_length) adaptive.reserve(members.size());
        if (measure_cost) frontiers.reserve(members.size());
        std::vector<speculative::RowGrowth> growth(members.size());
        const auto mark = p.next_membership_mark();
        for (std::size_t b = 0; b < members.size(); ++b) {
            auto& s      = p.sequence(members[b]);
            const auto f = s.continuation.metadata.main_frontier;
            if (s.phase != detail::Phase::Active || !s.permit ||
                s.permit->kind != ExecutionUnitKind::Decode || s.ledger.size() != f + 1 ||
                f >= p.shape.max_context || s.permitted_frontier < f + 1 ||
                s.membership_mark == mark)
                throw std::logic_error("decode row is not executable");
            s.membership_mark = mark;
            const auto hard   = options.mtp() && f
                                    ? std::min({p.shape.draft_tokens, selection.draft_budget(b),
                                                p.shape.max_context - f - 1})
                                    : 0U;
            limits.push_back({0, hard, hard});
            if (measure_cost) frontiers.push_back(f);
            if (adaptive_length) adaptive.push_back({limits.back(), f, &s.speculative_state});
            if (hard) {
                growth[b].text.emplace(p.directory.pool(0).histories(), s.histories[0].handle(), f,
                                       f + 1);
                growth[b].backend.emplace(p.directory.pool(1).histories(), s.histories[1].handle(),
                                          *s.continuation.metadata.mtp_frontier,
                                          mtp_history_frontier(f + 1));
            }
        }
        grant_windows(p.directory, growth, limits);
        for (std::size_t b = 0; b < adaptive.size(); ++b) adaptive[b].member = limits[b];
        const auto n     = static_cast<std::uint32_t>(members.size());
        const auto shape = p.plan.batches->decode(n, n);
        const auto path  = p.plan.decode_path;
        tx.round = selection.select(p.shape.length_policy, shape, path, p.plan.decode_routes.get(),
                                    p.cost_model, limits, adaptive, p.shape.max_context);
        if (!tx.round) return {};
        if (measure_cost)
            tx.cost_query = speculative::CostQuery::from_plan(options.speculative, *tx.round,
                                                              frontiers, p.shape.max_context);
        tx.record_online_cost = selection.record_online_cost;
        prepared              = true;
        const auto extents    = tx.round->verify_drafts();
        for (std::size_t b = 0; b < members.size(); ++b) growth[b].prepare(extents[b]);
        for (auto& row : growth) row.adopt();
        for (std::size_t b = 0; b < members.size(); ++b) {
            auto& s = p.sequence(members[b]);
            p.prepare_write(s, s.continuation.metadata.main_frontier + 1 + extents[b], tx.context);
            auto row        = p.work_row(s, 1);
            row.constrained = !constrained.empty() && constrained[b];
            tx.rows.push_back(std::move(row));
        }
        Work work;
        auto geometry =
            options.mtp()
                ? std::make_shared<const SpeculativeExecutionGeometry>(tx.round->geometry(path))
                : nullptr;
        if (tx.round->route().mtp_steps) {
            tx.kind = Kind::Mtp;
            work.operation =
                MtpWork{tx.rows, {extents.begin(), extents.end()}, shape, std::move(geometry)};
        } else {
            ForwardWork ordinary{ForwardKind::Ordinary, SampleKind::NextToken, tx.rows};
            ordinary.shape       = shape;
            ordinary.speculative = std::move(geometry);
            work.operation       = std::move(ordinary);
        }
        p.compute = std::move(pending);
        submit(p, std::move(*reservation), std::move(work));
        return tx.ticket;
    } catch (const std::exception& error) {
        if (prepared) p.executor->fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket>
Session::start_control(std::span<const SequenceHandle> members, std::span<const TokenId> tokens,
                       std::uint32_t stride) {
    auto& p = *impl_;
    if (p.compute) return {};
    if (members.empty() || members.size() > p.shape.max_concurrency || !stride ||
        stride > p.definition->resources().text.control_tokens->size() ||
        tokens.size() != members.size() * stride)
        throw std::logic_error("invalid forced control window");
    const auto mark = p.next_membership_mark();
    for (std::size_t b = 0; b < members.size(); ++b) {
        auto& s = p.sequence(members[b]);
        if (s.membership_mark == mark) throw std::logic_error("duplicate control member");
        s.membership_mark = mark;
        if (s.phase != detail::Phase::Active || !s.permit ||
            s.permit->kind != ExecutionUnitKind::Control ||
            s.ledger.size() != s.continuation.metadata.main_frontier + 1 ||
            s.continuation.metadata.main_frontier + stride > s.permitted_frontier)
            throw std::logic_error("forced control row is not executable");
    }
    for (const auto id : tokens)
        if (id < 0 || std::uint32_t(id) >= p.definition->resources().text.public_token_count)
            throw std::invalid_argument("forced token is outside vocabulary");
    auto reservation = p.executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    p.compute = std::make_unique<detail::ComputeTransaction>(Kind::Control, p.directory);
    auto& tx  = *p.compute;
    tx.members.assign(members.begin(), members.end());
    try {
        for (std::size_t b = 0; b < members.size(); ++b) {
            auto& s = p.sequence(members[b]);
            p.prepare_write(s, s.continuation.metadata.main_frontier + stride, tx.context);
            auto row               = p.work_row(s, 1);
            const auto forced      = tokens.subspan(b * stride, stride);
            row.control_last_token = forced.back();
            row.ids.insert(row.ids.end(), forced.begin(), forced.end() - 1);
            row.rope_positions.clear();
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
