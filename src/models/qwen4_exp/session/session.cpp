#include "models/qwen4_exp/session/internal.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

SessionImpl::SessionImpl(std::shared_ptr<const ModelDefinition> model, SessionPlan prepared)
    : definition(std::move(model)), plan(std::move(prepared)), shape(plan.shape),
      directory(plan.context),
      checkpoints(Checkpoints::capacity_for_metadata(directory.metadata_budget().limit()),
                  directory.metadata_budget(), &recovery_index) {
    if (!definition) throw std::invalid_argument("invalid Qwen4Exp session definition");
    if (shape.purpose == EnginePurpose::Generation) {
        count_storage.acquire(
            TokenCountBuilder::capacity_bytes(definition->resources().text.public_token_count));
        count_builder.prepare(definition->resources().text.public_token_count);
    }
    releases.reserve(shape.max_concurrency);
    for (std::uint16_t i = 0; i < pool_count(); ++i)
        pools.push_back({{i}, i == 0 ? "main" : "mtp"});
}

SessionImpl::~SessionImpl() { cleanup(); }

std::uint32_t SessionImpl::pool_count() const noexcept { return plan.context.pools.size(); }

bool SessionImpl::valid(SequenceHandle handle) const noexcept {
    return std::any_of(sequences.begin(), sequences.end(), [&](const auto& s) {
        return s && s->id == handle.value && s->epoch == handle.generation;
    });
}

Sequence& SessionImpl::sequence(SequenceHandle handle) const {
    for (const auto& s : sequences)
        if (s && s->id == handle.value && s->epoch == handle.generation) return *s;
    throw std::logic_error("stale Qwen4Exp sequence");
}

SequenceHandle SessionImpl::handle(const Sequence& s) const { return {s.id, s.epoch}; }

runtime::ExecutionMember SessionImpl::member(const Sequence& s) const {
    return {handle(s), {{s.lane}, s.binding_generation}};
}

StateBindings SessionImpl::selectors(const Sequence& s) const {
    StateBindings bindings;
    for (const auto rank : directory.states().ranks())
        bindings.emplace(rank, directory.states().selectors(s.continuation.image,
                                                            s.write_destination.valid()
                                                                ? s.write_destination
                                                                : s.continuation.image,
                                                            rank));
    return bindings;
}

ContextActions SessionImpl::actions(const context::ContextTransaction& transaction) const {
    ContextActions actions;
    for (const auto& rank : plan.context.ranks)
        if (!transaction.actions(rank.rank).empty())
            actions.emplace(rank.rank, transaction.actions(rank.rank));
    return actions;
}

void SessionImpl::complete_write(Sequence& s) {
    const auto source      = s.continuation.image;
    const auto destination = s.write_destination.valid() ? s.write_destination : source;
    directory.states().complete_write(source, destination);
    s.continuation.image = destination;
    s.write_destination  = {};
}

CaptureReservation::~CaptureReservation() {
    if (destination.valid() && !states->release(destination)) std::terminate();
    for (const auto& [record, role] : records) {
        if (checkpoints->valid(record)) {
            if (!checkpoints->erase(record)) std::terminate();
        } else {
            checkpoints->cancel(record);
        }
    }
}

Suspended::~Suspended() {
    if (snapshot && owner->checkpoints.valid(*snapshot) && !owner->checkpoints.erase(*snapshot))
        std::terminate();
}

void SessionImpl::release_residency(runtime::ResidencyId id) {
    const auto owner = vision_window.owner();
    if (owner && owner->residency == id) vision_window.revoke(*owner);
    if (executor->failed() || !id.generation) return;
    if (std::find(releases.begin(), releases.end(), id) == releases.end()) releases.push_back(id);
}

void SessionImpl::submit_releases() {
    if (operation || release_ticket || releases.empty() || executor->failed()) return;
    auto reservation = executor->try_reserve(runtime::FrameKind::Context);
    if (!reservation) return;
    Work work;
    ContextWork release;
    release.releases = std::move(releases);
    work.operation   = std::move(release);
    release_ticket   = executor->submit(std::move(*reservation), std::move(work));
}

void SessionImpl::clear(Sequence& s) noexcept {
    vision_window.revoke(member(s));
    try {
        release_residency(member(s).residency);
        s.capture.reset();
        for (auto& history : s.histories) history.reset();
        auto& states = directory.states();
        if (s.write_destination.valid()) {
            states.abort_write(s.continuation.image, s.write_destination);
            if (!states.release(s.write_destination)) std::terminate();
            s.write_destination = {};
        }
        if (states.valid(s.continuation.image) &&
            states.checkpoint_references(s.continuation.image) == 0 &&
            !states.release(s.continuation.image))
            std::terminate();
        s.continuation.image = {};
    } catch (...) { std::terminate(); }
}

void SessionImpl::cleanup() noexcept {
    if (!executor) return;
    pause_reservation.reset();
    pending_maintenance.reset();
    try {
        executor->drain();
    } catch (...) {
        // Rank teardown has attempted every reader stream before reporting a device failure.
    }
    try {
        if (compute) {
            compute->context.fail_after_drain();
            if (compute->completion) compute->completion->fail_after_drain();
            compute.reset();
        }
        if (operation) operation->migration.fail_after_drain();
        cancel_operation();
        scoring.reset();
        for (auto& s : sequences) {
            if (s) clear(*s);
            s.reset();
        }
        checkpoints.clear();
        events.clear();
        releases.clear();
        release_ticket.reset();
    } catch (...) { std::terminate(); }
}

std::uint32_t SessionImpl::window_end(const Sequence& s, std::uint32_t begin,
                                      std::uint32_t end) const {
    end = std::min(end, begin + shape.prefill_chunk);
    for (const auto frontier : s.identity.positions.execution_frontiers())
        if (frontier > begin) {
            end = std::min(end, frontier);
            break;
        }
    if (s.capture && s.next_capture < s.captures.size())
        end = std::min(end, s.captures[s.next_capture].frontier);
    if (s.request->vision && begin < s.request->summary.prompt_tokens && begin < end)
        end = vision_window.limit(member(s), s.request->vision->units, begin,
                                  std::min(end, s.request->summary.prompt_tokens),
                                  plan.vision_max_features);
    return end;
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
    if (pool_count() > 1)
        row.mtp_table =
            directory.pool(1).histories().execution_row(s.histories[1].handle()).handle();
    const auto& m         = s.continuation.metadata;
    row.main_frontier     = m.main_frontier;
    row.mtp_frontier      = m.mtp_frontier.value_or(0);
    row.rope_delta        = m.rope_delta;
    row.ple_raw_tail      = m.ple_raw_tail;
    row.sampling          = s.request->sampling;
    row.sampling_position = m.main_frontier;
    const auto start      = m.main_frontier;
    if (start > s.identity.tokens.size() || tokens > s.identity.tokens.size() - start)
        throw std::logic_error("execution input exceeds committed identity");
    row.ids.assign(s.identity.tokens.begin() + start, s.identity.tokens.begin() + start + tokens);
    row.rope_positions.resize(3ULL * tokens);
    const auto position = [&](std::uint32_t p, std::uint32_t axis) {
        return p < s.request->prompt.token_ids.size() ? s.request->prompt.position_axis(axis)[p]
                                                      : static_cast<std::int32_t>(p) + m.rope_delta;
    };
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        row.previous_rope[axis] = start ? position(start - 1, axis) : 0;
        for (std::uint32_t j = 0; j < tokens; ++j)
            row.rope_positions[axis * tokens + j] = position(start + j, axis);
    }
    bool linear = true;
    for (std::uint32_t axis = 0; axis < 3; ++axis)
        for (std::uint32_t j = 0; j < tokens; ++j)
            linear &= row.rope_positions[axis * tokens + j] ==
                      static_cast<std::int32_t>(start + j) + m.rope_delta;
    if (linear) row.rope_positions.clear();
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
        metadata.ple_raw_tail = {tokens > 1 ? inputs[tokens - 2] : metadata.ple_raw_tail[1],
                                 inputs[tokens - 1]};
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

PhysicalUsageSnapshot SessionImpl::usage() const noexcept { return directory.usage(); }

std::optional<qwen::VisionWork> SessionImpl::prepare_visual(const Sequence& s, std::uint32_t begin,
                                                            std::uint32_t end) {
    if (!s.request->vision || begin >= end) return {};
    const auto ranks   = static_cast<std::uint32_t>(plan.context.ranks.size());
    const auto* active = vision_window.current();
    if (active && active->owner.sequence == member(s).sequence &&
        active->owner.residency == member(s).residency)
        return qwen::prepare_vision_work(vision_window.reserve(active->owner, {}),
                                         s.request->vision, ranks);
    auto planned =
        vision::plan_window(s.request->vision->units, begin, end, plan.vision_max_features, ranks);
    if (!planned) return {};
    return qwen::prepare_vision_work(vision_window.reserve(member(s), std::move(planned)),
                                     s.request->vision, ranks);
}

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp {

PreparedSession::PreparedSession(std::unique_ptr<detail::SessionImpl> impl)
    : impl_(std::move(impl)) {}

PreparedSession::~PreparedSession()                                     = default;
PreparedSession::PreparedSession(PreparedSession&&) noexcept            = default;
PreparedSession& PreparedSession::operator=(PreparedSession&&) noexcept = default;

PreparedSession prepare_session(std::shared_ptr<const ModelDefinition> definition,
                                SessionPlan plan) {
    return PreparedSession(
        std::make_unique<detail::SessionImpl>(std::move(definition), std::move(plan)));
}

Session::Session(PreparedSession&& prepared, runtime::Executor<ExecutionTypes>& executor)
    : impl_(std::move(prepared.impl_)) {
    if (!impl_ || impl_->executor || !executor.epoch().value)
        throw std::invalid_argument("invalid prepared Qwen4Exp Session");
    impl_->executor            = &executor;
    impl_->identity_generation = executor.epoch().value;
}

Session::~Session() = default;

std::span<const runtime::KvPoolDescriptor> Session::kv_pools() const noexcept {
    return impl_->pools;
}

PhysicalUsageSnapshot Session::physical_usage() const noexcept { return impl_->usage(); }

bool Session::has_context_operation() const noexcept {
    return impl_->operation || impl_->pending_maintenance || impl_->release_ticket ||
           !impl_->releases.empty();
}

bool Session::execution_pending(SequenceHandle handle) const noexcept {
    return impl_->compute &&
           std::find(impl_->compute->members.begin(), impl_->compute->members.end(), handle) !=
               impl_->compute->members.end();
}

bool Session::context_blocks(SequenceHandle handle) const noexcept {
    const auto& p = *impl_;
    if (!p.valid(handle)) return false;
    if (p.directory.maintenance_pending() || (p.operation && p.operation->preparing_resources))
        return true;
    if (!p.operation) return false;
    const auto& s = p.sequence(handle);
    if (const auto* bind = std::get_if<detail::BindOperation>(&p.operation->action)) {
        for (std::uint32_t i = 0; i < p.pool_count(); ++i) {
            const auto* binding = p.operation->migration.binding(i);
            if (binding && binding->blocks_writer(s.histories[i].handle())) return true;
        }
    } else if (const auto* capture = std::get_if<detail::CaptureOperation>(&p.operation->action)) {
        return capture->lane == s.lane;
    } else if (const auto* pause = std::get_if<detail::PauseOperation>(&p.operation->action)) {
        return pause->lane == s.lane;
    }
    return false;
}

bool Session::recovery_pending(SequenceHandle handle) const noexcept {
    return impl_->valid(handle) && impl_->sequence(handle).recovery_peak != 0;
}

void Session::cancel_context() {
    if (impl_->operation) impl_->operation->migration.cancel();
}

void Session::supply(runtime::ExecutionTicket ticket, runtime::DecisionInput&& input) {
    if (!impl_->compute || impl_->compute->ticket != ticket)
        throw std::logic_error("decision input has no active Qwen4Exp execution");
    impl_->executor->supply(ticket, std::move(input));
}

std::optional<SessionEvent> Session::poll() {
    try {
        auto& p = *impl_;
        for (;;) {
            if (!p.events.empty()) {
                auto event = std::move(p.events.front());
                p.events.pop_front();
                return event;
            }
            if (p.pending_maintenance) (void)p.try_pending_maintenance();
            p.submit_releases();
            auto event = p.executor->poll();
            if (!event) return {};
            std::visit(
                [&](auto&& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, runtime::ExecutionFailed>) {
                        throw std::runtime_error(value.message);
                    } else if constexpr (std::is_same_v<T, runtime::DecisionNeeded>) {
                        if (!p.compute || p.compute->ticket != value.ticket)
                            throw std::logic_error("decision request belongs to another execution");
                        p.events.push_back(
                            {value.ticket, p.compute->members, std::move(value.request)});
                    } else if constexpr (std::is_same_v<T, runtime::DecisionConsumed>) {
                        if (!p.compute || p.compute->ticket != value.ticket)
                            throw std::logic_error("decision upload belongs to another execution");
                        p.events.push_back({value.ticket, p.compute->members, std::move(value)});
                    } else if constexpr (std::is_same_v<T, runtime::ResultReady<ExecutionTypes>>) {
                        if (p.release_ticket == value.ticket) {
                            if (!std::holds_alternative<ContextResult>(value.result))
                                throw std::logic_error(
                                    "residency release returned a compute result");
                        } else if (p.operation && p.operation->ticket == value.ticket) {
                            p.operation->result.emplace(
                                std::get<ContextResult>(std::move(value.result)));
                        } else {
                            if (!p.compute || p.compute->ticket != value.ticket)
                                throw std::logic_error("result belongs to another execution");
                            p.compute->observed_result = value.execution;
                            if (auto* score = std::get_if<ScoreResult>(&value.result))
                                p.finish_score(std::move(*score));
                            else
                                p.finish_result(std::get<ComputeResult>(std::move(value.result)));
                        }
                    } else if constexpr (std::is_same_v<T, runtime::DeviceRetired>) {
                        if (p.release_ticket == value.ticket) {
                            p.release_ticket.reset();
                            if (p.scoring && p.scoring->finishing) {
                                p.events.push_back({p.scoring->ticket,
                                                    {},
                                                    ScoreComplete{std::move(p.scoring->logprobs)}});
                                p.scoring.reset();
                            }
                        } else if (p.operation && p.operation->ticket == value.ticket) {
                            if (!p.operation->result)
                                throw std::logic_error("context retired without its result");
                            auto progress = p.complete_context();
                            if (!progress.complete) return;
                            if (p.scoring && !p.scoring->sequence) {
                                if (!progress.published || !progress.sequence)
                                    throw std::logic_error("scoring binding did not publish");
                                p.scoring->sequence = progress.sequence;
                                (void)p.submit_score();
                            } else {
                                p.events.push_back({value.ticket, {}, std::move(progress)});
                            }
                        } else if (p.compute && p.compute->ticket == value.ticket) {
                            p.retire_compute(value);
                        } else {
                            throw std::logic_error("retirement belongs to another execution");
                        }
                    }
                },
                std::move(*event));
        }
    } catch (const std::exception& error) {
        impl_->executor->fail(error.what());
        throw;
    }
}

void Session::fail_all_cleanup() noexcept { impl_->cleanup(); }

ResumeState::ResumeState(std::unique_ptr<detail::Suspended> state) : state_(std::move(state)) {}

ResumeState::~ResumeState()                                 = default;
ResumeState::ResumeState(ResumeState&&) noexcept            = default;
ResumeState& ResumeState::operator=(ResumeState&&) noexcept = default;

bool ResumeState::has_snapshot() const noexcept { return state_ && state_->snapshot.has_value(); }

std::optional<CheckpointHandle> ResumeState::snapshot_handle() const noexcept {
    return state_ ? state_->snapshot : std::nullopt;
}

std::uint32_t ResumeState::frontier() const noexcept { return state_ ? state_->frontier : 0; }

std::optional<SequenceHandle> Session::prefill_continuation() const {
    const auto owner = impl_->vision_window.owner();
    if (owner && impl_->valid(owner->sequence)) return owner->sequence;
    return {};
}

void Session::yield_prefill(SequenceHandle sequence) {
    const auto owner = impl_->vision_window.owner();
    if (owner && owner->sequence == sequence) impl_->vision_window.revoke(*owner);
}

} // namespace ninfer::models::qwen4_exp
