#include "models/qwen3_5/session/internal.h"
#include "core/paged_kv_geometry.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

RequestBasePlan::RequestBasePlan(std::shared_ptr<const detail::RequestBasePlanImpl> impl) noexcept
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

std::optional<PrefixKey> RequestBasePlan::prefix_key(std::uint32_t frontier) const {
    if (!impl_ || !frontier || frontier > impl_->summary.prompt_tokens) return {};
    return impl_->query.key(frontier);
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
    if (!definition) throw std::invalid_argument("Session preparation is incomplete");
    return PreparedSession(std::make_unique<detail::SessionImpl>(std::move(plan)));
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

PrefixKey Session::checkpoint_key(CheckpointHandle h) const {
    return impl_->checkpoint(h).snapshot.input_prefix().key();
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
    impl_->recovery_index.synchronize(
        context::RecoveryStoreView{impl_->checkpoints, impl_->prefixes});
    return impl_->recovery_index.evaluate(advertised);
}

std::size_t Session::checkpoint_extent() const noexcept { return impl_->checkpoints.size(); }

std::uint64_t Session::checkpoint_catalog_revision() const noexcept {
    return impl_->checkpoints.revision();
}

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

std::size_t Session::host_bytes_released(std::span<const CheckpointHandle> checkpoints) const {
    return impl_->host_bytes_released(checkpoints);
}

std::size_t Session::release_redundant_host(std::span<const CheckpointHandle> excluded,
                                            std::optional<SequenceHandle> pending_backup) {
    return impl_->release_redundant_host(excluded, pending_backup);
}

ResourceReservation Session::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

bool Session::reclaim_capture_reservation(const ContextAssessment& assessment) {
    return impl_->reclaim_capture_reservation(assessment);
}

void Session::release_units(std::span<const SequenceHandle> units) noexcept {
    impl_->release_units(units);
}

BindingReservation Session::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& source, ResumeState* resume,
                                          ExecutionUnitKind kind, std::uint32_t tokens) {
    return impl_->start_binding(base, lane, source, resume, kind, tokens);
}

runtime::ContextStart Session::start_capture(SequenceHandle h) { return impl_->start_capture(h); }

bool Session::retry_capture(SequenceHandle h) { return impl_->retry_capture(h); }

std::vector<CheckpointHandle> Session::share_capture(SequenceHandle h, CheckpointHandle source) {
    return impl_->share_capture(h, source);
}

std::vector<std::uint32_t> Session::capture_frontiers(SequenceHandle h, std::uint32_t begin,
                                                      std::uint32_t end) const {
    std::vector<std::uint32_t> result;
    const auto& points = impl_->requests[impl_->lane(h)].capture_groups;
    for (auto it = std::upper_bound(points.begin(), points.end(), begin,
                                    [](auto f, const auto& point) { return f < point.frontier; });
         it != points.end() && it->frontier <= end; ++it)
        result.push_back(it->frontier);
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
Session::prepare_capture(SequenceHandle h, std::span<const CheckpointHandle> victims,
                         runtime::CaptureStorage storage) {
    return impl_->prepare_capture(h, victims, storage);
}

void Session::skip_capture(SequenceHandle h, runtime::CaptureSkip reason) {
    impl_->skip_capture(h, reason);
}

runtime::ContextStart Session::start_demote(const ContextDemotion& plan,
                                            std::span<const CheckpointHandle> host_victims) {
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
    return impl_->start_pause(h, action);
}

bool Session::context_blocks(SequenceHandle sequence) const noexcept {
    return impl_->context_blocks(sequence);
}

bool Session::recovery_pending(SequenceHandle sequence) const noexcept {
    return impl_->recovery_pending(sequence);
}

bool Session::has_context_operation() const noexcept { return impl_->has_context_operation(); }

runtime::PrefillBudgetPolicy Session::prefill_budget_policy() const noexcept {
    return {.max_tokens_global           = impl_->prefill_chunk,
            .max_tokens_per_placement    = impl_->prefill_chunk,
            .max_sequences_per_step      = impl_->max_concurrency,
            .max_sequences_per_placement = impl_->max_concurrency};
}

runtime::PrefillCandidateInfo Session::prefill_candidate(SequenceHandle handle) const {
    auto info      = impl_->prefill_candidate(handle);
    info.placement = placement(handle);
    return info;
}

std::optional<runtime::ExecutionTicket> Session::start_prefill(SequenceHandle h, bool constrained) {
    return impl_->start_prefill(h, constrained);
}

std::optional<runtime::ExecutionTicket>
Session::start_prefill(std::span<const SequenceHandle> members,
                       std::span<const runtime::PrefillGrant> grants) {
    return impl_->start_prefill(members, grants);
}

std::optional<runtime::ExecutionTicket> Session::start_replay(SequenceHandle h) {
    return impl_->start_replay(h);
}

std::optional<runtime::ExecutionTicket>
Session::start_decode(std::span<const SequenceHandle> h, std::span<const runtime::RoundBudget> b,
                      std::span<const std::uint8_t> c) {
    return start_decode_round(h, {.budgets = b}, c);
}

std::optional<runtime::ExecutionTicket>
Session::start_decode_round(std::span<const SequenceHandle> members,
                            speculative::DecodeRoundRequest selection,
                            std::span<const std::uint8_t> constrained) {
    return impl_->start_decode_round(members, selection, constrained);
}

std::optional<runtime::ExecutionTicket> Session::start_control(std::span<const SequenceHandle> h,
                                                               std::span<const TokenId> ids,
                                                               std::uint32_t stride) {
    return impl_->start_control(h, ids, stride);
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

std::optional<runtime::ExecutionTicket>
Session::start_decode_probe(std::span<const SequenceHandle> members,
                            const speculative::RoundRoute& route,
                            std::span<const std::uint32_t> extents,
                            std::span<const std::uint8_t> constrained, bool record_online_cost) {
    return start_decode_round(
        members, {.extents = extents, .route = &route, .record_online_cost = record_online_cost},
        constrained);
}

std::optional<runtime::ExecutionTicket>
Session::start_decode_extents(std::span<const SequenceHandle> members,
                              std::span<const std::uint32_t> extents,
                              std::span<const std::uint8_t> constrained) {
    return start_decode_round(members, {.extents = extents}, constrained);
}

speculative::CostModelSummary Session::speculative_cost_summary() const noexcept {
    return impl_->cost_model.summary();
}

std::vector<std::uint32_t> Session::decode_target_widths() const {
    const auto& routes = impl_->plan.decode_routes;
    return routes ? routes->target_widths() : std::vector<std::uint32_t>{};
}

speculative::CalibrationConstraints Session::calibration_constraints() const {
    const auto& p = *impl_;
    return {.model           = p.cost_model.config(),
            .batches         = p.plan.batch_execution.get(),
            .path            = p.plan.decode_path,
            .routes          = p.plan.decode_routes.get(),
            .max_concurrency = p.max_concurrency,
            .max_context     = p.capacity,
            .kv_page_tokens  = kPagedKVPageSize,
            .available_kv_pages =
                speculative::available_calibration_pages(p.directory, placement_descriptions())};
}

std::optional<speculative::CostQuery>
Session::decode_cost_query(runtime::ExecutionTicket ticket) const noexcept {
    const auto& active = impl_->execution;
    return active && active->ticket == ticket ? active->cost_query : std::nullopt;
}

void Session::fit_speculative_cost_model(std::span<const speculative::TimedCostAnchor> anchors,
                                         double elapsed_ms) {
    impl_->cost_model.fit(anchors, elapsed_ms);
}

std::uint32_t Session::decode_frontier(SequenceHandle handle) const {
    return impl_->sequences[impl_->lane(handle)].continuation.metadata.execution_frontier;
}

std::optional<SequenceHandle> Session::prefill_continuation() const {
    const auto owner = impl_->vision_window.owner();
    if (owner && impl_->valid_sequence(owner->sequence)) return owner->sequence;
    return {};
}

void Session::yield_prefill(SequenceHandle sequence) {
    const auto owner = impl_->vision_window.owner();
    if (owner && owner->sequence == sequence) impl_->vision_window.revoke(*owner);
}

runtime::PrefixId Session::checkpoint_prefix(CheckpointHandle h) const {
    return impl_->checkpoints.at(h).snapshot.input_prefix().id();
}

std::vector<runtime::PrefixId> Session::matching_prefixes(const RequestBasePlan& base) const {
    const auto& plan = *base.impl_;
    return plan.query.ancestors();
}

std::vector<runtime::PrefixRelation>
Session::recovery_relations(std::span<const runtime::PrefixId> ids) const {
    return impl_->prefixes.relations(ids);
}

bool Session::recovery_precedes(runtime::PrefixId a, std::uint32_t af, runtime::PrefixId b,
                                std::uint32_t bf) const {
    return af <= bf && impl_->prefixes.valid(a) && impl_->prefixes.valid(b) &&
           impl_->prefixes.is_prefix(a, b);
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {
namespace {
speculative::CostModelConfig cost_config(const SessionPlan& plan) {
    const auto* moe =
        plan.profile ? std::get_if<MoeConfig>(&plan.profile->config.text.ffn) : nullptr;
    return {
        plan.speculative_backend,           plan.draft_window,     1, moe ? moe->num_experts : 0,
        moe ? moe->num_experts_per_tok : 0, plan.hybrid_max_tokens};
}
} // namespace

SessionImpl::SessionImpl(SessionPlan plan_in)
    : plan(std::move(plan_in)), directory(plan.context), state_store(&directory.states()),
      text_kv_histories(&directory.pool(0).histories()),
      backend_kv_histories(directory.pool_count() > 1 ? &directory.pool(1).histories() : nullptr),
      text_kv_pages(&directory.pool(0).pages()),
      backend_kv_pages(directory.pool_count() > 1 ? &directory.pool(1).pages() : nullptr),
      host_context_enabled(plan.context.ranks.at(0).host_bytes != 0), capacity(plan.capacity),
      max_concurrency(plan.max_concurrency), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), speculative_backend(plan.speculative_backend),
      cost_model(cost_config(plan)), context_cache(plan.context_cache),
      vision_enabled(plan.profile->options.vision), causal_scoring(plan.causal_scoring),
      sequences(max_concurrency), requests(max_concurrency), lane_epochs(max_concurrency),
      lane_marks(max_concurrency),
      recovery_metadata(directory.metadata_budget(), plan.causal_scoring ? 0 : plan.max_concurrency,
                        1, 1) {
    if (!plan.profile || !plan.batch_execution || plan.context.pools.empty())
        throw std::invalid_argument("Session preparation is incomplete");
    checkpoints.configure(Checkpoints::capacity_for_metadata(directory.metadata_budget().limit()),
                          directory.metadata_budget(), &recovery_index);
    if (!causal_scoring) {
        count_storage.acquire(TokenCountBuilder::capacity_bytes(plan.profile->public_tokens));
        count_builder.prepare(plan.profile->public_tokens);
    }
    releases.reserve(max_concurrency);
    resident_sequences.reserve(max_concurrency);
    for (std::uint32_t i = 0; i < max_concurrency; ++i) sequences[i].lane = i;
}

std::uint64_t SessionImpl::next_lane_mark() {
    if (++lane_mark_generation == 0) {
        std::fill(lane_marks.begin(), lane_marks.end(), 0);
        ++lane_mark_generation;
    }
    return lane_mark_generation;
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


} // namespace ninfer::models::qwen3_5::detail
