#include "models/qwen3_5/session/internal.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

namespace {

// Advances the context epoch when a call starts or ends a context operation. Execution steps that
// leave the retained checkpoints and their placements alone do not, so a cached reclaim ranking
// survives the decode steps between two reclaim attempts.
class OperationWatch {
public:
    explicit OperationWatch(detail::SessionImpl& impl) noexcept
        : impl_(impl), pending_(impl.has_context_operation()) {}

    ~OperationWatch() {
        if (impl_.has_context_operation() != pending_) impl_.note_context_change();
    }

    OperationWatch(const OperationWatch&)            = delete;
    OperationWatch& operator=(const OperationWatch&) = delete;

private:
    detail::SessionImpl& impl_;
    bool pending_;
};

} // namespace

RequestBasePlan::RequestBasePlan(std::shared_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept = default;

RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;

RequestBasePlan::~RequestBasePlan() = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const models::frontend::PreparedContextCache& RequestBasePlan::context_cache() const noexcept {
    static const models::frontend::PreparedContextCache empty;
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
                        std::span<const CheckpointHandle> retired_points,
                        runtime::PlacementId target, const ResumeState* resume) const {
    return impl_->inspect_source(base, checkpoint, consume_source, private_points, retired_points,
                                 target, resume);
}

std::vector<PlacementDescriptor> Session::placement_descriptions() const { return {{0, {0}}}; }

std::span<const runtime::PlacementId> Session::placements(const RequestBasePlan&) const {
    static constexpr std::array<runtime::PlacementId, 1> placements{{{0}}};
    return placements;
}

runtime::PlacementId Session::placement(SequenceHandle sequence) const {
    (void)impl_->lane(sequence);
    return {0};
}

std::uint64_t Session::load_tokens(SequenceHandle sequence) const {
    const auto lane     = impl_->lane(sequence);
    const auto& control = impl_->requests[lane];
    const auto& state   = impl_->sequences[lane];
    return std::max({control.base->summary.prompt_tokens,
                     state.continuation.metadata.execution_frontier,
                     control.recovery ? control.recovery->frontier : 0U});
}

BindingEvaluation Session::assess_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& candidate,
                                          const ResumeState* resume, ExecutionUnitKind kind,
                                          std::uint32_t tokens) const {
    return impl_->plan_binding(base, lane, candidate, resume, kind, tokens).evaluation;
}

PrefixShortlistKey Session::checkpoint_key(CheckpointHandle h, std::uint32_t f) const {
    return impl_->checkpoint_key(h, f);
}

std::span<const runtime::KvPoolDescriptor> Session::kv_pools() const noexcept {
    return impl_->plan.kv_pools;
}

ContextAssessment Session::assess(const ContextDemand& demand,
                                  const ContextReleaseSet& releases) const {
    return impl_->directory.assess(demand, releases);
}

bool Session::releases_metadata(const ContextReleaseSet& released,
                                const ContextAssessment& assessment) const {
    return assessment.metadata_contribution(released);
}

std::uint64_t Session::released_bytes(const ContextReleaseSet& releases,
                                      const ContextAssessment& assessment) const {
    ContextReleaseSet relevant = releases;
    std::erase_if(relevant.records, [&](const auto& record) {
        return std::none_of(assessment.placement.domains.begin(),
                            assessment.placement.domains.end(),
                            [&](const auto& item) { return item.domain == record.domain; });
    });
    return impl_->directory.released_bytes(relevant);
}

std::uint64_t Session::useful_release_bytes(const ContextReleaseSet& released,
                                            const ContextAssessment& assessment) const {
    return impl_->directory.useful_release_bytes(released, assessment);
}

std::vector<std::uint64_t> Session::release_limits(const ContextAssessment& assessment) const {
    return impl_->directory.release_limits(assessment);
}

runtime::ContextReleaseMeasure Session::release_measure(const ContextReleaseSet& released,
                                                        const ContextAssessment& assessment) const {
    return impl_->directory.release_measure(released, assessment);
}

runtime::ContextStart Session::start_maintenance(const ContextAssessment& assessment) {
    impl_->note_context_change();
    return impl_->start_maintenance(assessment);
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

std::size_t Session::checkpoint_extent() const noexcept { return impl_->checkpoints.size(); }

std::vector<runtime::ContextTransferRequirement>
Session::checkpoint_restore_requirements(CheckpointHandle h) const {
    return impl_->checkpoints.at(h).snapshot.restore_requirements(impl_->directory);
}

bool Session::valid_checkpoint(CheckpointHandle h) const noexcept {
    return impl_->valid_checkpoint(h);
}

bool Session::release_checkpoint(CheckpointHandle h) noexcept {
    impl_->note_context_change();
    return impl_->release_checkpoint(h);
}

bool Session::revoke_snapshot(ResumeState& paused) noexcept {
    impl_->note_context_change();
    return impl_->revoke_snapshot(paused);
}

std::size_t Session::host_bytes_released(std::span<const CheckpointHandle> checkpoints) const {
    return impl_->host_bytes_released(checkpoints);
}

std::size_t Session::release_redundant_host(std::span<const CheckpointHandle> excluded,
                                            std::optional<SequenceHandle> pending_backup) {
    impl_->note_context_change();
    return impl_->release_redundant_host(excluded, pending_backup);
}

ResourceReservation Session::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

bool Session::reclaim_capture_reservation(const ContextAssessment& assessment) {
    impl_->note_context_change();
    return impl_->reclaim_capture_reservation(assessment);
}

void Session::release_units(std::span<const SequenceHandle> units) noexcept {
    impl_->note_context_change();
    impl_->release_units(units);
}

BindingReservation Session::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& source, ResumeState* resume,
                                          ExecutionUnitKind kind, std::uint32_t tokens) {
    impl_->note_context_change();
    return impl_->start_binding(base, lane, source, resume, kind, tokens);
}

bool Session::start_capture(SequenceHandle h) {
    impl_->note_context_change();
    return impl_->start_capture(h);
}

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

std::optional<CapturePreparation>
Session::prepare_capture(SequenceHandle h, std::span<const CheckpointHandle> host_victims) {
    impl_->note_context_change();
    return impl_->prepare_capture(h, host_victims);
}

void Session::skip_capture(SequenceHandle h, runtime::CaptureSkip reason) {
    impl_->note_context_change();
    impl_->skip_capture(h, reason);
}

runtime::ContextStart Session::start_demote(const ContextDemotion& plan,
                                            std::span<const CheckpointHandle> host_victims) {
    impl_->note_context_change();
    return impl_->start_demote(plan, host_victims);
}

std::vector<PauseOption> Session::pause_options(SequenceHandle sequence) const {
    return impl_->pause_options(sequence);
}

bool Session::pause_contributes(SequenceHandle sequence, const PauseOption& option,
                                const ContextAssessment& assessment) const {
    return impl_->pause_contributes(sequence, option, assessment);
}

bool Session::prepare_pause(SequenceHandle sequence, runtime::PauseAction action,
                            std::span<const CheckpointHandle> host_victims) {
    impl_->note_context_change();
    return impl_->prepare_pause(sequence, action, host_victims);
}

Session::HostRequirement Session::demotion_host_requirements(const ContextDemotion& plan) const {
    return impl_->demotion_host_requirements(plan);
}

ContextDemotion Session::combine_demotions(std::span<const ContextDemotion> quotes) const {
    return impl_->combine_demotions(quotes);
}

std::size_t Session::host_write_bytes(const HostRequirement& write) const {
    return impl_->directory.host_write_bytes(write);
}

bool Session::can_place_host(const HostRequirement& write) const {
    return impl_->directory.plan_host(write).feasible();
}

bool Session::can_place_host(const HostRequirement& write,
                             const ContextReleaseSet& releases) const {
    return impl_->directory.assess(host_demand(write), releases).feasible();
}

ContextDemand Session::host_demand(const HostRequirement& write) const {
    return impl_->directory.host_demand(write);
}

ContextReclaimPlan Session::plan_reclaim(std::span<const CheckpointHandle> allowed,
                                         std::span<const CheckpointHandle> excluded,
                                         const ContextDemand& demand) const {
    return impl_->plan_reclaim(allowed, excluded, demand);
}

context::ReleasePlan<ContextRelease, ContextReleaseSet>
Session::plan_releases(std::span<const CheckpointHandle> allowed,
                       std::span<const CheckpointHandle> excluded,
                       const ContextDemand& demand) const {
    return impl_->plan_releases(allowed, excluded, demand);
}

bool Session::start_pause(SequenceHandle h, runtime::PauseAction action) {
    impl_->note_context_change();
    return impl_->start_pause(h, action);
}

bool Session::context_blocks(SequenceHandle sequence) const noexcept {
    return impl_->context_blocks(sequence);
}

bool Session::recovery_pending(SequenceHandle sequence) const noexcept {
    return impl_->recovery_pending(sequence);
}

bool Session::has_context_operation() const noexcept { return impl_->has_context_operation(); }

std::uint64_t Session::context_epoch() const noexcept { return impl_->context_epoch(); }

bool Session::context_pins(CheckpointHandle handle) const noexcept {
    return impl_->context_pins(handle);
}

std::uint32_t Session::prefill_pack_limit() const noexcept { return impl_->prefill_pack_limit(); }

std::uint32_t Session::prefill_pack_tokens() const noexcept { return impl_->prefill_pack_tokens(); }

std::optional<PrefillStart> Session::start_prefill(std::span<const SequenceHandle> sequences,
                                                   std::span<const bool> constrained) {
    const OperationWatch watch(*impl_);
    return impl_->start_prefill(sequences, constrained);
}

std::uint32_t Session::mixed_decode_limit() const noexcept { return impl_->mixed_decode_limit(); }

bool Session::prefill_mixable(SequenceHandle h) const { return impl_->prefill_mixable(h); }

std::optional<MixedStart> Session::start_mixed(std::span<const SequenceHandle> prefill,
                                               std::span<const bool> prefill_constrained,
                                               std::span<const SequenceHandle> decode,
                                               std::span<const runtime::RoundBudget> budgets,
                                               std::span<const bool> decode_constrained) {
    const OperationWatch watch(*impl_);
    return impl_->start_mixed(prefill, prefill_constrained, decode, budgets, decode_constrained);
}

std::optional<runtime::ExecutionTicket> Session::start_replay(SequenceHandle h) {
    const OperationWatch watch(*impl_);
    return impl_->start_replay(h);
}

std::optional<runtime::ExecutionTicket>
Session::start_decode(std::span<const SequenceHandle> h, std::span<const runtime::RoundBudget> b,
                      std::span<const bool> c) {
    const OperationWatch watch(*impl_);
    return impl_->start_decode(h, b, c);
}

std::optional<runtime::ExecutionTicket>
Session::start_control(std::span<const SequenceHandle> h, std::span<const TokenId> ids,
                       std::uint32_t stride, std::span<const std::optional<std::uint32_t>> splits) {
    const OperationWatch watch(*impl_);
    return impl_->start_control(h, ids, stride, splits);
}

std::optional<runtime::ExecutionTicket> Session::start_score(qwen::PreparedPrompt&& prompt,
                                                             std::uint32_t first) {
    const OperationWatch watch(*impl_);
    return impl_->start_score(qwen::PreparedPromptAccess::take(std::move(prompt)), first);
}

std::optional<SessionEvent> Session::poll() {
    const OperationWatch watch(*impl_);
    auto event = impl_->poll();
    // A finished context operation has moved checkpoints, pages or placements.
    if (event && std::holds_alternative<ContextProgress>(event->result))
        impl_->note_context_change();
    return event;
}

void Session::supply(runtime::ExecutionTicket ticket, runtime::DecisionInput&& input) {
    const OperationWatch watch(*impl_);
    impl_->supply(ticket, std::move(input));
}

CommitResult Session::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> d) {
    const OperationWatch watch(*impl_);
    return impl_->commit(std::move(pending), d);
}

DiscardResult Session::discard(PendingBatch&& pending) noexcept {
    const OperationWatch watch(*impl_);
    return impl_->discard(std::move(pending));
}

void Session::cancel_context() noexcept {
    impl_->note_context_change();
    impl_->cancel_context();
}

bool Session::execution_pending(SequenceHandle h) const noexcept {
    return impl_->execution_pending(h);
}

bool Session::terminal_blocked(SequenceHandle h) const noexcept {
    return impl_->terminal_blocked(h);
}

FinishResult Session::finish(SequenceHandle h) noexcept {
    impl_->note_context_change();
    return impl_->finish(h);
}

AbortResult Session::abort(SequenceHandle h) noexcept {
    impl_->note_context_change();
    return impl_->abort(h);
}

void Session::fail_all_cleanup() noexcept {
    impl_->note_context_change();
    impl_->fail_all_cleanup();
}

PhysicalUsageSnapshot Session::physical_usage() const noexcept { return impl_->physical_usage(); }

std::optional<SequenceHandle> Session::prefill_continuation() const {
    const auto owner = impl_->vision_window.owner();
    if (owner && impl_->valid_sequence(owner->sequence)) return owner->sequence;
    return {};
}

void Session::yield_prefill(SequenceHandle sequence) {
    const auto owner = impl_->vision_window.owner();
    if (owner && owner->sequence == sequence) impl_->vision_window.revoke(*owner);
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {

SessionImpl::SessionImpl(std::shared_ptr<const ModelDefinition> definition_in, SessionPlan plan_in)
    : definition(std::move(definition_in)), plan(std::move(plan_in)), directory(plan.context),
      state_store(&directory.states()), text_kv_histories(&directory.pool(0).histories()),
      backend_kv_histories(directory.pool_count() > 1 ? &directory.pool(1).histories() : nullptr),
      text_kv_pages(&directory.pool(0).pages()),
      backend_kv_pages(directory.pool_count() > 1 ? &directory.pool(1).pages() : nullptr),
      host_context_enabled(plan.context.ranks.at(0).host_bytes != 0), capacity(plan.capacity),
      max_concurrency(plan.max_concurrency), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), speculative_backend(plan.speculative_backend),
      context_cache(plan.context_cache), vision_enabled(plan.profile->options.vision),
      causal_scoring(plan.causal_scoring) {
    if (!definition || !plan.profile || plan.context.pools.empty() ||
        plan.pack_tokens < plan.prefill_chunk || plan.pack_segments == 0 ||
        plan.pack_segments > plan.max_concurrency)
        throw std::invalid_argument("Session preparation is incomplete");
    checkpoints.configure(Checkpoints::capacity_for_metadata(directory.metadata_budget().limit()),
                          directory.metadata_budget(), &recovery_index);
    if (!causal_scoring) {
        count_storage.acquire(TokenCountBuilder::capacity_bytes(plan.profile->public_tokens));
        count_builder.prepare(plan.profile->public_tokens);
    }
    releases.reserve(max_concurrency);
    for (std::uint32_t i = 0; i < max_concurrency; ++i) sequences[i].lane = i;
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
    pending_maintenance_.reset();
    if (context_operation_) context_operation_->transaction.fail_after_drain();
    abort_context();
    if (execution) {
        try {
            if (execution->preparation) execution->preparation->fail_after_drain();
            if (execution->settlement) execution->settlement->fail_after_drain();
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
    if (terminal_blocked(h)) return {};
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
    if (terminal_blocked(h)) return {};
    const auto l = lane(h);
    AbortResult out{runtime::ConsumeStatus::Consumed, requests[l].speculative_stats};
    clear_lane(sequences[l], requests[l]);
    return out;
}


} // namespace ninfer::models::qwen3_5::detail
