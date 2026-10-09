#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/program_impl.h"
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {


SequencePlan::SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlan::SequencePlan(SequencePlan&&) noexcept = default;

SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;

SequencePlan::~SequencePlan() = default;

std::uint32_t SequencePlan::capacity() const noexcept {
    return impl_ != nullptr ? impl_->capacity : 0;
}

std::uint32_t SequencePlan::kv_capacity() const noexcept {
    return impl_ != nullptr ? impl_->kv_capacity : 0;
}

std::uint32_t SequencePlan::max_concurrency() const noexcept {
    return impl_ != nullptr ? impl_->max_concurrency : 0;
}

std::size_t SequencePlan::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->device_reservation_bytes : 0;
}

std::size_t SequencePlan::workspace_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->workspace.capacity : 0;
}

std::size_t SequencePlan::host_capacity_bytes() const noexcept {
    return impl_ ? impl_->context_cache.host_capacity_bytes.value_or(0) : 0;
}

SequencePlanner::SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlanner::SequencePlanner(SequencePlanner&&) noexcept = default;

SequencePlanner& SequencePlanner::operator=(SequencePlanner&&) noexcept = default;

SequencePlanner::~SequencePlanner() = default;

const runtime::SequenceCapacityCurve& SequencePlanner::capacity_curve() const noexcept {
    static const runtime::SequenceCapacityCurve empty;
    return impl_ != nullptr ? impl_->curve : empty;
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return SequencePlan(detail::finalize_sequence_plan_impl(std::move(impl_), main_page_groups));
}

RequestBasePlan::RequestBasePlan(std::shared_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept = default;

RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;

RequestBasePlan::~RequestBasePlan() = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PreparedContextCache& RequestBasePlan::context_cache() const noexcept {
    static const PreparedContextCache empty;
    return impl_ != nullptr ? impl_->context_cache : empty;
}

std::vector<std::uint32_t> RequestBasePlan::capture_frontiers() const {
    std::vector<std::uint32_t> frontiers;
    if (impl_) {
        frontiers.reserve(impl_->capture_groups.size());
        for (const auto& group : impl_->capture_groups) { frontiers.push_back(group.frontier); }
    }
    return frontiers;
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

Program::Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}

Program::~Program() noexcept = default;

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(PreparedPromptAccess::take(std::move(prompt)), options);
}

std::vector<float> Program::causal_score(PreparedPrompt&& prompt, std::uint32_t target) {
    return impl_->causal_score(PreparedPromptAccess::take(std::move(prompt)), target);
}

std::optional<SourceCandidate>
Program::inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint,
                        bool consume_source, std::span<const CheckpointHandle> private_points,
                        std::span<const CheckpointHandle> retired_points) const {
    return impl_->inspect_source(base, checkpoint, consume_source, private_points, retired_points);
}

PrefixShortlistKey Program::checkpoint_key(CheckpointHandle h, std::uint32_t f) const {
    return impl_->checkpoint_key(h, f);
}

runtime::ContextResourceUsage
Program::checkpoint_footprint(std::span<const CheckpointHandle> handles) const {
    return impl_->checkpoint_footprint(handles);
}

CheckpointSummary Program::checkpoint_summary(CheckpointHandle h) const {
    return impl_->checkpoint_summary(h);
}

CheckpointMetadata Program::checkpoint_metadata(CheckpointHandle h) const {
    return impl_->checkpoint_metadata(h);
}

bool Program::checkpoint_matches(CheckpointHandle h, const RequestBasePlan& b) const {
    return impl_->checkpoint_matches(h, b);
}

std::uint32_t Program::checkpoint_recovery_frontier(CheckpointHandle retained,
                                                    const RequestBasePlan& base,
                                                    std::uint32_t target) const {
    return impl_->checkpoint_recovery_frontier(retained, base, target);
}

std::uint64_t Program::checkpoint_recovery_loss(std::span<const CheckpointHandle> removed,
                                                std::span<const CheckpointHandle> surviving) const {
    return impl_->checkpoint_recovery_loss(removed, surviving);
}

bool Program::valid_checkpoint(CheckpointHandle h) const noexcept {
    return impl_->valid_checkpoint(h);
}

bool Program::release_checkpoint(CheckpointHandle h) noexcept {
    return impl_->release_checkpoint(h);
}

bool Program::revoke_snapshot(ResumeState& paused) noexcept {
    return impl_->revoke_snapshot(paused);
}

runtime::ContextResourceUsage Program::snapshot_resources(const ResumeState& paused) const {
    return impl_->snapshot_resources(paused);
}

std::size_t Program::host_bytes_released(std::span<const CheckpointHandle> checkpoints) const {
    return impl_->host_bytes_released(checkpoints);
}

std::optional<std::size_t> Program::pause_host_bytes(SequenceHandle sequence) const {
    return impl_->pause_host_bytes(sequence);
}

std::size_t Program::release_redundant_host(std::span<const CheckpointHandle> excluded,
                                            std::optional<SequenceHandle> pending_backup) {
    return impl_->release_redundant_host(excluded, pending_backup);
}

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

bool Program::reclaim_capture_reservation(runtime::ContextResourceUsage shortage) {
    return impl_->reclaim_capture_reservation(shortage);
}

void Program::release_units(std::span<const SequenceHandle> units) noexcept {
    impl_->release_units(units);
}

BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& source, ResumeState* resume,
                                          ExecutionUnitKind kind, std::uint32_t tokens) {
    return impl_->start_binding(base, lane, source, resume, kind, tokens);
}

bool Program::start_capture(SequenceHandle h) { return impl_->start_capture(h); }

bool Program::capture_is_input(SequenceHandle h) const { return impl_->capture_is_input(h); }

std::optional<CapturePreparation> Program::prepare_capture(SequenceHandle h) {
    return impl_->prepare_capture(h);
}

void Program::skip_capture(SequenceHandle h) { impl_->skip_capture(h); }

bool Program::start_demote(const ContextDemotion& plan) { return impl_->start_demote(plan); }

ContextReclaimPlan Program::plan_reclaim(std::span<const CheckpointHandle> allowed,
                                         std::span<const CheckpointHandle> excluded,
                                         runtime::ContextResourceUsage shortage) const {
    return impl_->plan_reclaim(allowed, excluded, shortage);
}

std::vector<ContextRelease> Program::plan_releases(std::span<const CheckpointHandle> allowed,
                                                   std::span<const CheckpointHandle> excluded,
                                                   runtime::ContextResourceUsage shortage) const {
    return impl_->plan_releases(allowed, excluded, shortage);
}

bool Program::start_pause(SequenceHandle h, bool save, runtime::ExecutionTiming* timing) {
    return impl_->start_pause(h, save, timing);
}

ContextProgress Program::poll_context(runtime::CancellationFlagView c) {
    return impl_->poll_context(c);
}

bool Program::context_blocks(SequenceHandle sequence) const noexcept {
    return impl_->context_blocks(sequence);
}

bool Program::recovery_pending(SequenceHandle sequence) const noexcept {
    return impl_->recovery_pending(sequence);
}

bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }

PrefillProgress Program::advance_prefill(SequenceHandle h, runtime::ExecutionTiming* t,
                                         runtime::TokenMaskProvider* m) {
    return impl_->advance_prefill(h, t, m);
}

ReplayProgress Program::advance_replay(SequenceHandle h, runtime::ExecutionTiming* t) {
    return impl_->advance_replay(h, t);
}

BranchScoreResult Program::score_branches(std::span<const BranchScoreRow> rows,
                                          const PreparationControl& control) {
    return impl_->score_branches(rows, control);
}

bool Program::release_branch_cache() { return impl_->release_branch_cache(); }

PendingBatch Program::decode(std::span<const SequenceHandle> s,
                             std::span<const runtime::RoundBudget> b, runtime::ExecutionTiming* t,
                             runtime::TokenMaskProvider* m) {
    return impl_->decode(s, b, t, m);
}

runtime::ExecutionTiming Program::append_forced_tokens(
    std::span<const SequenceHandle> s, std::span<const TokenId> ids, std::uint32_t stride,
    std::span<const std::optional<std::uint32_t>> splits, runtime::ExecutionTiming* t) {
    return impl_->append_forced_tokens(s, ids, stride, splits, t);
}

CommitResult Program::commit(PendingBatch&& p, std::span<const runtime::CommitDecision> d,
                             runtime::CommitObservation o, runtime::ExecutionTiming* t) {
    return impl_->commit(std::move(p), d, o, t);
}

DiscardResult Program::abort_pending(PendingBatch&& p) noexcept {
    return impl_->abort_pending(std::move(p));
}

FinishResult Program::finish(SequenceHandle s) noexcept { return impl_->finish(s); }

AbortResult Program::abort(SequenceHandle s) noexcept { return impl_->abort(s); }

void Program::fail_all_cleanup() noexcept { impl_->fail_all_cleanup(); }

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->physical_usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }

void Program::reset_memory_peaks() noexcept { impl_->reset_memory_peaks(); }

SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                      DeviceContext& device, const EngineOptions& options) {
    return SequencePlanner(detail::make_sequence_planner_impl(parameters, device, options));
}

std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                        SequencePlan&& plan, DeviceContext& device,
                                        const StartupObserver& startup_observer) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.impl_->parameters != &parameters) {
        throw std::invalid_argument("sequence plan belongs to another model instance");
    }
    if (plan.impl_->multiprocessor_count != device.multiprocessor_count()) {
        throw std::invalid_argument("sequence plan device capacity does not match execution");
    }
    auto impl =
        std::make_unique<detail::ProgramImpl>(parameters, *plan.impl_, device, startup_observer);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

} // namespace ninfer::models::qwen3_5
