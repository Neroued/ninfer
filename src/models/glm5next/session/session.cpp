#include "models/glm5next/session/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {

SessionImpl::SessionImpl(std::shared_ptr<const ModelDefinition> model, SessionPlan prepared,
                         std::unique_ptr<context::ContextDirectory> prepared_directory)
    : definition(std::move(model)), plan(std::move(prepared)), shape(plan.shape),
      directory_owner(std::move(prepared_directory)), directory(*directory_owner),
      checkpoints(Checkpoints::capacity_for_metadata(directory.metadata_budget().limit()),
                  directory.metadata_budget(), &recovery_index) {
    if (!definition) throw std::invalid_argument("invalid GLM5Next session definition");
    releases.reserve(shape.max_concurrency);
    for (std::uint32_t i = 0; i < plan.states.size(); ++i) placements.push_back({i});
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
    throw std::logic_error("stale GLM5Next sequence");
}

SequenceHandle SessionImpl::handle(const Sequence& s) const { return {s.id, s.epoch}; }

runtime::ExecutionMember SessionImpl::member(const Sequence& s) const {
    return {handle(s), {{s.lane}, s.binding_generation}};
}

StateBindings SessionImpl::selectors(const Sequence& s) const {
    StateBindings bindings;
    for (const auto rank : directory.states().ranks(s.continuation.metadata.placement.value))
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
    try {
        vision_window.revoke(member(s));
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
        vision_window = {};
        checkpoints.clear();
        events.clear();
        releases.clear();
        release_ticket.reset();
    } catch (...) { std::terminate(); }
}

PhysicalUsageSnapshot SessionImpl::usage() const noexcept { return directory.usage(); }

} // namespace ninfer::models::glm5next::detail

namespace ninfer::models::glm5next {

PreparedSession::PreparedSession(std::unique_ptr<detail::SessionImpl> impl)
    : impl_(std::move(impl)) {}

PreparedSession::~PreparedSession()                                     = default;
PreparedSession::PreparedSession(PreparedSession&&) noexcept            = default;
PreparedSession& PreparedSession::operator=(PreparedSession&&) noexcept = default;

PreparedSession prepare_session(std::shared_ptr<const ModelDefinition> definition, SessionPlan plan,
                                std::unique_ptr<context::ContextDirectory> directory) {
    if (!directory) throw std::invalid_argument("GLM Session requires its prepared directory");
    return PreparedSession(std::make_unique<detail::SessionImpl>(
        std::move(definition), std::move(plan), std::move(directory)));
}

Session::Session(PreparedSession&& prepared, runtime::Executor<ExecutionTypes>& executor)
    : impl_(std::move(prepared.impl_)) {
    if (!impl_ || impl_->executor || !executor.epoch().value)
        throw std::invalid_argument("invalid prepared GLM5Next Session");
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
        throw std::logic_error("decision input has no active GLM5Next execution");
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

void Session::yield_prefill(SequenceHandle h) {
    impl_->vision_window.revoke(impl_->member(impl_->sequence(h)));
}

} // namespace ninfer::models::glm5next
