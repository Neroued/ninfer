#include "models/qwen3_5/session/internal.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

RequestBasePlan::RequestBasePlan(std::shared_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept = default;

RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;

RequestBasePlan::~RequestBasePlan() = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const qwen::PreparedContextCache& RequestBasePlan::context_cache() const noexcept {
    static const qwen::PreparedContextCache empty;
    return impl_ != nullptr ? impl_->context_cache : empty;
}

std::optional<PrefixShortlistKey>
RequestBasePlan::prefix_shortlist_key(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr || frontier == 0 || frontier > impl_->prefix_digests.size()) {
        return std::nullopt;
    }
    return PrefixShortlistKey{
        .digests      = impl_->prefix_digests.at(frontier),
        .frontier     = frontier,
        .identity_tag = impl_->prefix_identity_tag,
    };
}

ResumeState::ResumeState(std::unique_ptr<detail::ResumeStateImpl> impl) noexcept
    : impl_(std::move(impl)) {}

ResumeState::ResumeState(ResumeState&&) noexcept            = default;
ResumeState& ResumeState::operator=(ResumeState&&) noexcept = default;
ResumeState::~ResumeState()                                 = default;

bool ResumeState::has_snapshot() const noexcept { return impl_ && impl_->snapshot.has_value(); }

std::optional<CheckpointHandle> ResumeState::snapshot_handle() const noexcept {
    return impl_ ? impl_->snapshot : std::nullopt;
}

std::uint32_t ResumeState::frontier() const noexcept { return impl_ ? impl_->frontier : 0; }

PreparedSession::PreparedSession(std::unique_ptr<detail::SessionImpl> impl)
    : impl_(std::move(impl)) {}

PreparedSession::PreparedSession(PreparedSession&&) noexcept            = default;
PreparedSession& PreparedSession::operator=(PreparedSession&&) noexcept = default;
PreparedSession::~PreparedSession()                                     = default;

PreparedSession prepare_session(std::shared_ptr<const ModelDefinition> definition,
                                SessionPlan plan) {
    return PreparedSession(
        std::make_unique<detail::SessionImpl>(std::move(definition), std::move(plan)));
}

Session::Session(PreparedSession&& prepared, runtime::Executor<ExecutionTypes>& executor)
    : impl_(std::move(prepared.impl_)) {
    if (!impl_ || impl_->executor_) throw std::invalid_argument("prepared session is unavailable");
    impl_->executor_ = &executor;
}

Session::~Session() noexcept = default;

RequestBasePlan Session::plan_request(qwen::PreparedPrompt&& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(qwen::PreparedPromptAccess::take(std::move(prompt)), options);
}

std::optional<SourceCandidate>
Session::inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint,
                        bool consume_source, std::span<const CheckpointHandle> private_points,
                        std::span<const CheckpointHandle> retired_points) const {
    return impl_->inspect_source(base, checkpoint, consume_source, private_points, retired_points);
}

PrefixShortlistKey Session::checkpoint_key(CheckpointHandle h, std::uint32_t f) const {
    return impl_->checkpoint_key(h, f);
}

std::span<const runtime::KvPoolDescriptor> Session::kv_pools() const noexcept {
    return impl_->plan.kv_pools;
}

ContextResources Session::checkpoint_footprint(std::span<const CheckpointHandle> handles) const {
    return impl_->checkpoint_footprint(handles);
}

CheckpointSummary Session::checkpoint_summary(CheckpointHandle h) const {
    return impl_->checkpoint_summary(h);
}

CheckpointMetadata Session::checkpoint_metadata(CheckpointHandle h) const {
    return impl_->checkpoint_metadata(h);
}

bool Session::checkpoint_matches(CheckpointHandle h, const RequestBasePlan& b) const {
    return impl_->checkpoint_matches(h, b);
}

std::uint32_t Session::checkpoint_recovery_frontier(CheckpointHandle retained,
                                                    const RequestBasePlan& base,
                                                    std::uint32_t target) const {
    return impl_->checkpoint_recovery_frontier(retained, base, target);
}

context::RecoveryEvaluation<>
Session::evaluate_recovery(std::span<const CheckpointHandle> advertised) const {
    impl_->recovery_index.synchronize(qwen::RecoveryStoreView{impl_->checkpoints});
    return impl_->recovery_index.evaluate(advertised);
}

std::size_t Session::checkpoint_capacity() const noexcept { return impl_->checkpoints.size(); }

std::vector<runtime::ContextTransferRequirement>
Session::checkpoint_restore_requirements(CheckpointHandle h) const {
    return impl_->checkpoints.at(h).snapshot.restore_requirements(impl_->directory);
}

bool Session::valid_checkpoint(CheckpointHandle h) const noexcept {
    return impl_->valid_checkpoint(h);
}

bool Session::release_checkpoint(CheckpointHandle h) noexcept {
    return impl_->release_checkpoint(h);
}

bool Session::revoke_snapshot(ResumeState& paused) noexcept {
    return impl_->revoke_snapshot(paused);
}

ContextResources Session::snapshot_resources(const ResumeState& paused) const {
    return impl_->snapshot_resources(paused);
}

std::size_t Session::host_bytes_released(std::span<const CheckpointHandle> checkpoints) const {
    return impl_->host_bytes_released(checkpoints);
}

std::optional<std::size_t> Session::pause_host_bytes(SequenceHandle sequence) const {
    return impl_->pause_host_bytes(sequence);
}

std::size_t Session::release_redundant_host(std::span<const CheckpointHandle> excluded,
                                            std::optional<SequenceHandle> pending_backup) {
    return impl_->release_redundant_host(excluded, pending_backup);
}

ResourceReservation Session::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

bool Session::reclaim_capture_reservation(ContextResources shortage) {
    return impl_->reclaim_capture_reservation(shortage);
}

void Session::release_units(std::span<const SequenceHandle> units) noexcept {
    impl_->release_units(units);
}

BindingReservation Session::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& source, ResumeState* resume,
                                          ExecutionUnitKind kind, std::uint32_t tokens) {
    return impl_->start_binding(base, lane, source, resume, kind, tokens);
}

bool Session::start_capture(SequenceHandle h) { return impl_->start_capture(h); }

std::vector<std::uint32_t> Session::capture_frontiers(SequenceHandle h) const {
    std::vector<std::uint32_t> result;
    for (const auto& point : impl_->requests[impl_->lane(h)].capture_groups)
        result.push_back(point.frontier);
    return result;
}

bool Session::capture_is_shared(SequenceHandle h) const {
    const auto& request = impl_->requests[impl_->lane(h)];
    if (request.next_capture >= request.capture_groups.size()) return false;
    const auto& roles = request.capture_groups[request.next_capture].roles;
    return roles.size() == 1 && roles.front() == runtime::CheckpointRole::SharedPrefix;
}

bool Session::capture_is_input(SequenceHandle h) const { return impl_->capture_is_input(h); }

std::optional<CapturePreparation> Session::prepare_capture(SequenceHandle h) {
    return impl_->prepare_capture(h);
}

void Session::skip_capture(SequenceHandle h, runtime::CaptureSkip reason) {
    impl_->skip_capture(h, reason);
}

bool Session::start_demote(const ContextDemotion& plan) { return impl_->start_demote(plan); }

ContextReclaimPlan Session::plan_reclaim(std::span<const CheckpointHandle> allowed,
                                         std::span<const CheckpointHandle> excluded,
                                         ContextResources shortage) const {
    return impl_->plan_reclaim(allowed, excluded, shortage);
}

context::ReleasePlan<ContextRelease, ContextResources>
Session::plan_releases(std::span<const CheckpointHandle> allowed,
                       std::span<const CheckpointHandle> excluded,
                       ContextResources shortage) const {
    return impl_->plan_releases(allowed, excluded, shortage);
}

bool Session::start_pause(SequenceHandle h, bool save) { return impl_->start_pause(h, save); }

bool Session::context_blocks(SequenceHandle sequence) const noexcept {
    return impl_->context_blocks(sequence);
}

bool Session::recovery_pending(SequenceHandle sequence) const noexcept {
    return impl_->recovery_pending(sequence);
}

bool Session::has_context_operation() const noexcept { return impl_->has_context_operation(); }

std::optional<runtime::ExecutionTicket> Session::start_prefill(SequenceHandle h, bool c) {
    return impl_->start_prefill(h, c);
}

std::optional<runtime::ExecutionTicket> Session::start_replay(SequenceHandle h) {
    return impl_->start_replay(h);
}

std::optional<runtime::ExecutionTicket>
Session::start_decode(std::span<const SequenceHandle> h, std::span<const runtime::RoundBudget> b,
                      std::span<const bool> c) {
    return impl_->start_decode(h, b, c);
}

std::optional<runtime::ExecutionTicket>
Session::start_control(std::span<const SequenceHandle> h, std::span<const TokenId> ids,
                       std::uint32_t stride, std::span<const std::optional<std::uint32_t>> splits) {
    return impl_->start_control(h, ids, stride, splits);
}

std::optional<runtime::ExecutionTicket> Session::start_score(qwen::PreparedPrompt&& prompt,
                                                             std::uint32_t first) {
    return impl_->start_score(qwen::PreparedPromptAccess::take(std::move(prompt)), first);
}

std::optional<SessionEvent> Session::poll() { return impl_->poll(); }

void Session::supply(runtime::ExecutionTicket ticket, runtime::DecisionInput&& input) {
    impl_->supply(ticket, std::move(input));
}

CommitResult Session::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> d) {
    return impl_->commit(std::move(pending), d);
}

DiscardResult Session::discard(PendingBatch&& pending) noexcept {
    return impl_->discard(std::move(pending));
}

void Session::cancel_context() noexcept { impl_->cancel_context(); }

bool Session::execution_pending(SequenceHandle h) const noexcept {
    return impl_->execution_pending(h);
}

FinishResult Session::finish(SequenceHandle h) noexcept { return impl_->finish(h); }

AbortResult Session::abort(SequenceHandle h) noexcept { return impl_->abort(h); }

void Session::fail_all_cleanup() noexcept { impl_->fail_all_cleanup(); }

PhysicalUsageSnapshot Session::physical_usage() const noexcept { return impl_->physical_usage(); }

void Session::reset_measurement_peaks() { impl_->reset_measurement_peaks(); }

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {

SessionImpl::SessionImpl(std::shared_ptr<const ModelDefinition> definition_in, SessionPlan plan_in)
    : definition(std::move(definition_in)), plan(std::move(plan_in)), directory(plan.context),
      state_store(&directory.states()), text_kv_histories(&directory.pool(0).histories()),
      backend_kv_histories(directory.pool_count() > 1 ? &directory.pool(1).histories() : nullptr),
      text_kv_pages(&directory.pool(0).pages()),
      backend_kv_pages(directory.pool_count() > 1 ? &directory.pool(1).pages() : nullptr),
      host_kv_extents(plan.context.host.extent_count ? &directory.host() : nullptr),
      host_context_arena(plan.context.host.bytes ? &directory.host_allocation() : nullptr),
      capacity(plan.capacity), max_concurrency(plan.max_concurrency),
      prefill_chunk(plan.prefill_chunk), draft_window(plan.draft_window),
      speculative_backend(plan.speculative_backend), context_cache(plan.context_cache),
      vision_enabled(plan.profile->options.vision), causal_scoring(plan.causal_scoring) {
    if (!definition || !plan.profile || plan.context.pools.empty())
        throw std::invalid_argument("Session preparation is incomplete");
    checkpoints.resize(4ULL * plan.context.state.logical_capacity + 2ULL * max_concurrency);
    releases.reserve(max_concurrency);
    for (std::uint32_t i = 0; i < max_concurrency; ++i) sequences[i].lane = i;
    for (std::size_t i = 0; i < plan.kv_pools.size(); ++i) {
        auto& descriptor    = plan.kv_pools[i];
        descriptor.geometry = &plan.context.pools[i].geometry;
    }
}

SessionImpl::~SessionImpl() {
    if (executor_) fail_all_cleanup();
}

void SessionImpl::fail_all_cleanup() noexcept {
    closing = true;
    if (executor_) {
        try {
            executor().drain();
        } catch (...) {}
    }
    if (context_operation_ && context_operation_->transaction.in_flight())
        context_operation_->transaction.mark_completed();
    abort_context();
    if (execution) {
        try {
            if (execution->preparation && execution->preparation->in_flight())
                execution->preparation->mark_completed();
            if (execution->settlement && execution->settlement->in_flight())
                execution->settlement->mark_completed();
            if (execution->preparation) execution->preparation->discard();
            if (execution->settlement) execution->settlement->discard();
        } catch (...) { std::terminate(); }
        execution.reset();
    }
    events.clear();
    releases.clear();
    release_ticket.reset();
    after_releases.reset();
    scoring.reset();
    for (std::uint32_t l = 0; l < max_concurrency; ++l)
        if (requests[l].lifecycle != Lifecycle::Empty) clear_lane(sequences[l], requests[l]);
}

FinishResult SessionImpl::finish(SequenceHandle h) noexcept {
    if (!valid_sequence(h) || execution_pending(h) || context_blocks(h)) return {};
    try {
        const auto l  = lane(h);
        auto& state   = sequences[l];
        auto& request = requests[l];
        if (request.lifecycle != Lifecycle::Finishable) return {};
        FinishResult out{runtime::ConsumeStatus::Consumed, request.speculative_stats, {}};
        if (request.publish_continuation) out.checkpoint = detach_checkpoint(state);
        clear_lane(state, request);
        return out;
    } catch (...) { std::terminate(); }
}

AbortResult SessionImpl::abort(SequenceHandle h) noexcept {
    if (!valid_sequence(h) || execution_pending(h) || context_blocks(h)) return {};
    const auto l = lane(h);
    AbortResult out{runtime::ConsumeStatus::Consumed, requests[l].speculative_stats};
    clear_lane(sequences[l], requests[l]);
    return out;
}

void SessionImpl::reset_measurement_peaks() {
    executor().reset_measurement_peaks();
    directory.host_allocation().reset_peak();
}

} // namespace ninfer::models::qwen3_5::detail
