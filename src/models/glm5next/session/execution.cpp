#include "models/glm5next/session/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next {
namespace {
using Kind = detail::ComputeTransaction::Kind;

void submit(detail::SessionImpl& session, runtime::ExecutionReservation reservation, Work work) {
    auto& tx = *session.compute;
    tx.context.prepare_actions();
    work.actions = session.actions(tx.context);
    work.batch   = plan_batch(work_rows(work), session.plan.states.size());
    tx.ticket    = session.executor->submit(std::move(reservation), std::move(work));
    tx.context.mark_submitted();
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
        forward.kind = ForwardKind::Replay;
        forward.rows = tx.rows;
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
    for (std::size_t b = 0; b < members.size(); ++b) {
        const auto& s = p.sequence(members[b]);
        const auto f  = s.continuation.metadata.main_frontier;
        if (s.phase != detail::Phase::Active || !s.permit ||
            !budgets[b].generated_tokens_remaining || s.identity.tokens.size() != f + 1 ||
            std::find(members.begin(), members.begin() + b, members[b]) != members.begin() + b)
            throw std::logic_error("decode row is not executable");
    }
    auto reservation = p.executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    p.compute = std::make_unique<detail::ComputeTransaction>(Kind::Ordinary, p.directory);
    auto& tx  = *p.compute;
    tx.members.assign(members.begin(), members.end());
    try {
        for (std::size_t b = 0; b < members.size(); ++b) {
            auto& s        = p.sequence(members[b]);
            const auto end = s.continuation.metadata.main_frontier + 1;
            if (end > s.permitted_frontier)
                throw std::logic_error("decode exceeds its reservation");
            p.prepare_write(s, end, tx.context);
            auto row        = p.work_row(s, 1);
            row.constrained = !constrained.empty() && constrained[b];
            tx.rows.push_back(std::move(row));
        }
        Work work;
        work.operation = ForwardWork{ForwardKind::Ordinary, SampleKind::NextToken, tx.rows};
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
        if (id < 0 || std::uint32_t(id) >= p.definition->resources().text.public_token_count)
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
            for (const auto frontier : s.identity.execution_frontiers)
                if (frontier > row.main_frontier && frontier < row.main_frontier + stride)
                    row.window_ends.push_back(frontier - row.main_frontier);
            if (splits[b] && *splits[b] + 1 < stride) row.window_ends.push_back(*splits[b] + 1);
            std::sort(row.window_ends.begin(), row.window_ends.end());
            row.window_ends.erase(std::unique(row.window_ends.begin(), row.window_ends.end()),
                                  row.window_ends.end());
            tx.rows.push_back(std::move(row));
        }
        submit(p, std::move(*reservation),
               {{}, ForwardWork{ForwardKind::Control, SampleKind::None, tx.rows}});
        return tx.ticket;
    } catch (const std::exception& error) {
        p.executor->fail(error.what());
        throw;
    }
}

} // namespace ninfer::models::glm5next

namespace ninfer::models::glm5next {

ResourceReservation detail::SessionImpl::reserve_units(std::span<const ExecutionUnit> units) {
    auto& p = *this;
    if (p.directory.maintenance_pending())
        return {.assessment = {.status = context::ContextAvailability::Waiting}};

    struct Need {
        detail::Sequence* sequence;
        std::uint32_t end;
        std::array<std::uint32_t, 2> additional{};
        std::array<context::ContextDemandId, 2> targets;
        std::array<context::RecordReservation, 2> prepared;
    };

    std::vector<Need> needed;
    ContextDemand demand;
    for (const auto& unit : units) {
        auto& sequence = p.sequence(unit.sequence);
        if ((unit.kind == ExecutionUnitKind::Prefill || unit.kind == ExecutionUnitKind::Replay) &&
            p.unit_end(sequence, unit.kind, unit.tokens) ==
                sequence.continuation.metadata.main_frontier &&
            sequence.continuation.metadata.main_frontier < sequence.request->summary.prompt_tokens)
            return {.status = runtime::UnitReservationStatus::ExecutionUnavailable};
        if (std::any_of(needed.begin(), needed.end(),
                        [&](const auto& n) { return n.sequence == &sequence; }))
            throw std::logic_error("execution reservation repeats a sequence");
        Need need{&sequence,
                  std::max(sequence.recovery_peak, p.unit_end(sequence, unit.kind, unit.tokens))};
        if (need.end > p.shape.max_context) throw std::out_of_range("execution exceeds context");
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool) {
            const auto& histories = p.directory.pool(pool).histories();
            const auto history    = sequence.histories[pool].handle();
            const auto begin      = pool ? sequence.continuation.metadata.mtp_frontier.value()
                                         : sequence.continuation.metadata.main_frontier;
            const auto end        = pool ? mtp_history_frontier(need.end) : need.end;
            const auto pages      = histories.write_pages(history, begin, end);
            const auto held       = histories.reserved_growth_pages(history);
            need.additional[pool] = pages > held ? pages - held : 0;
            need.targets[pool]    = demand.add_growth(histories, history, need.additional[pool]);
        }
        needed.push_back(std::move(need));
    }
    auto result = p.directory.reserve(demand);
    if (!result || !result.reservation->ready()) return {.assessment = result.assessment};
    for (auto& need : needed)
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool) {
            need.prepared[pool] = result.reservation->take(need.targets[pool]);
            if (need.additional[pool]) {
                p.directory.pool(pool).histories().prepare_growth(
                    need.sequence->histories[pool].handle(), need.prepared[pool]);
            }
        }
    for (std::size_t i = 0; i < units.size(); ++i) {
        auto& sequence = *needed[i].sequence;
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool)
            if (needed[i].additional[pool])
                p.directory.pool(pool).histories().adopt_growth(
                    sequence.histories[pool].handle(), std::move(needed[i].prepared[pool]));
        sequence.permit             = units[i];
        sequence.permitted_frontier = needed[i].end;
    }
    return {.status = runtime::UnitReservationStatus::Granted, .assessment = result.assessment};
}

ResourceReservation Session::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

void Session::release_units(std::span<const SequenceHandle> handles) noexcept {
    try {
        for (const auto h : handles)
            if (impl_->valid(h) && !execution_pending(h)) impl_->settle(impl_->sequence(h));
    } catch (...) { std::terminate(); }
}

} // namespace ninfer::models::glm5next
