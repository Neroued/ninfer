#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/program_impl.h"
#include <algorithm>
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

SequencePlanner::SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlanner::SequencePlanner(SequencePlanner&&) noexcept = default;

SequencePlanner& SequencePlanner::operator=(SequencePlanner&&) noexcept = default;

SequencePlanner::~SequencePlanner() = default;

std::size_t SequencePlanner::unallocated_reservation_bytes(int rank) const noexcept {
    return impl_ != nullptr && impl_->minimum != nullptr
               ? impl_->minimum->unallocated_reservation_bytes(rank)
               : 0;
}

std::size_t SequencePlanner::kv_pool_bytes(std::uint32_t main_page_groups) const noexcept {
    if (impl_ == nullptr || impl_->minimum == nullptr) { return 0; }
    const auto& d = impl_->minimum->persistent.decoder;
    // Page-major planes: a pool's device bytes are its bytes per page times its pages.
    const auto per_page = [](const qwen3_5::PagedKVCacheLayout& pool) {
        return pool.pages.spec.page_group_count == 0
                   ? std::size_t{0}
                   : pool.pages.payload_bytes() / pool.pages.spec.page_group_count;
    };
    const std::uint32_t text_pages = d.text_kv.pages.spec.page_group_count;
    std::size_t bytes              = per_page(d.text_kv) * main_page_groups;
    if (d.mtp_kv) {
        bytes += per_page(*d.mtp_kv) *
                 (main_page_groups + (d.mtp_kv->pages.spec.page_group_count - text_pages));
    }
    return bytes;
}

std::uint32_t SequencePlanner::kv_pool_pages() const noexcept {
    return impl_ != nullptr && impl_->minimum != nullptr
               ? impl_->minimum->persistent.decoder.text_kv.pages.spec.page_group_count
               : 0;
}

std::size_t SequencePlanner::kv_plane_count() const noexcept {
    if (impl_ == nullptr || impl_->minimum == nullptr) { return 0; }
    const auto& d = impl_->minimum->persistent.decoder;
    // Device planes only: active KV in RAM keeps the others in Host memory (no elastic range).
    const auto devices = [](const qwen3_5::PagedKVCacheLayout& pool) {
        return static_cast<std::size_t>(
            std::count_if(pool.pages.planes.begin(), pool.pages.planes.end(),
                          [](const DeviceKVPlaneLayout& plane) { return !plane.geometry.host; }));
    };
    return devices(d.text_kv) + (d.mtp_kv ? devices(*d.mtp_kv) : 0);
}

const runtime::SequenceCapacityCurve& SequencePlanner::capacity_curve() const noexcept {
    static const runtime::SequenceCapacityCurve empty;
    return impl_ != nullptr ? impl_->curve : empty;
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return SequencePlan(detail::finalize_sequence_plan_impl(std::move(impl_), main_page_groups));
}

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept
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

std::optional<runtime::PrefillWork>
RequestBasePlan::shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr) { return std::nullopt; }
    const auto found = std::find_if(impl_->shared_candidates.begin(),
                                    impl_->shared_candidates.end(), [&](const auto& candidate) {
                                        return candidate.frontier == frontier && candidate.identity;
                                    });
    return found == impl_->shared_candidates.end()
               ? std::nullopt
               : std::optional<runtime::PrefillWork>(found->identity->rebuild_work);
}

PressurePlanningSession::PressurePlanningSession(
    std::unique_ptr<detail::PressurePlanningSessionImpl> impl) noexcept
    : impl_(std::move(impl)) {}

PressurePlanningSession::PressurePlanningSession(PressurePlanningSession&&) noexcept = default;

PressurePlanningSession&
PressurePlanningSession::operator=(PressurePlanningSession&&) noexcept = default;

PressurePlanningSession::~PressurePlanningSession() = default;

CapturePressurePlanningSession::CapturePressurePlanningSession(
    CapturePressurePlanningSession&&) noexcept = default;

CapturePressurePlanningSession&
CapturePressurePlanningSession::operator=(CapturePressurePlanningSession&&) noexcept = default;

CapturePressurePlanningSession::~CapturePressurePlanningSession() = default;

PressureTargetHandle
PressurePlanningSession::identity_target(runtime::PlanningCandidateId candidate) const {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->identity_target(candidate);
}

PressureTargetHandle
PressurePlanningSession::root_maximal_target(runtime::PlanningCandidateId root_candidate) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->root_maximal_target(root_candidate);
}

PressureTargetHandle
PressurePlanningSession::maximal_target(runtime::PlanningCandidateId candidate) {
    return impl_->maximal_target(candidate);
}

PressureConstructionCursor PressurePlanningSession::begin_construction(PressureTargetHandle target,
                                                                       bool restore) {
    return impl_->begin_construction(target, restore);
}

runtime::PressureConstructionStep
PressurePlanningSession::next_construction_option(PressureConstructionCursor& cursor) {
    return impl_->next_construction_option(cursor);
}

void PressurePlanningSession::choose_construction(PressureConstructionCursor& cursor,
                                                  runtime::PressureConstructionOptionId option) {
    impl_->choose_construction(cursor, option);
}

std::optional<PressureTargetHandle>
PressurePlanningSession::construction_target(const PressureConstructionCursor& cursor) {
    return impl_->construction_target(cursor);
}

runtime::PressureTargetGuidance PressurePlanningSession::guidance(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->guidance(target);
}

AssessedPressureTarget PressurePlanningSession::assess(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->assess(target);
}

PreparedPressureExpansion PressurePlanningSession::prepare_expansion(PressureTargetHandle parent,
                                                                     std::uint32_t maximum_owners) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->prepare_expansion(parent, maximum_owners);
}

PressureExpansionView
PressurePlanningSession::commit_expansion(PreparedPressureExpansion&& prepared) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->commit_expansion(std::move(prepared));
}

void PressurePlanningSession::discard_expansion(PreparedPressureExpansion&& prepared) noexcept {
    if (impl_ != nullptr) { impl_->discard_expansion(std::move(prepared)); }
}

runtime::PrefillWork PressurePlanningSession::shared_capture_split_prefill_work(
    const AssessedPressureTarget& assessed, const PreparedPrompt& prompt,
    std::span<const std::uint32_t> frontiers) const {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->shared_capture_split_prefill_work(assessed, PreparedPromptAccess::view(prompt),
                                                    frontiers);
}

std::optional<ResourcePlan> PressurePlanningSession::seal(AssessedPressureTarget&& assessed,
                                                          const PreparedPrompt& prompt,
                                                          runtime::FinalScheduleIntent intent) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    std::optional<AdmissionCandidate> sealed =
        impl_->seal(std::move(assessed), PreparedPromptAccess::view(prompt), intent);
    if (!sealed) { return std::nullopt; }
    const bool needs_transfer = sealed->impl_->needs_transfer;
    return ResourcePlan(std::move(*sealed), impl_->resource_revision, needs_transfer);
}

std::optional<CapturePressurePlan>
PressurePlanningSession::seal_capture(AssessedPressureTarget&& assessed) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    std::optional<CapturePressureCandidate> sealed = impl_->seal_capture(std::move(assessed));
    if (!sealed) { return std::nullopt; }
    return CapturePressurePlan(std::move(*sealed), impl_->resource_revision);
}

PressureTargetHandle CapturePressurePlanningSession::identity_target() const {
    if (!candidate_.impl_) { throw std::logic_error("capture pressure candidate is empty"); }
    return session_.identity_target(candidate_id());
}

runtime::PressureTargetGuidance
CapturePressurePlanningSession::guidance(PressureTargetHandle target) {
    return session_.guidance(target);
}

AssessedPressureTarget CapturePressurePlanningSession::assess(PressureTargetHandle target) {
    return session_.assess(target);
}

PreparedPressureExpansion
CapturePressurePlanningSession::prepare_expansion(PressureTargetHandle parent) {
    return session_.prepare_expansion(parent);
}

PressureExpansionView
CapturePressurePlanningSession::commit_expansion(PreparedPressureExpansion&& prepared) {
    return session_.commit_expansion(std::move(prepared));
}

void CapturePressurePlanningSession::discard_expansion(
    PreparedPressureExpansion&& prepared) noexcept {
    session_.discard_expansion(std::move(prepared));
}

std::optional<CapturePressurePlan>
CapturePressurePlanningSession::seal(AssessedPressureTarget&& assessed) {
    return session_.seal_capture(std::move(assessed));
}

Program::Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}

Program::~Program() noexcept = default;

RequestBasePlan Program::plan_request(const PreparedPrompt& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(PreparedPromptAccess::view(prompt), options);
}

std::vector<float> Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target) {
    return impl_->causal_score(PreparedPromptAccess::take(std::move(prompt)), first_target);
}

std::uint32_t Program::disk_resume_tokens(const PreparedPrompt& prompt) {
    return impl_->disk_resume_tokens(PreparedPromptAccess::view(prompt));
}

std::optional<AdmissionCandidate> Program::inspect_admission(
    const PreparedPrompt& prompt, const RequestBasePlan& base, runtime::LaneId destination,
    const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
    std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source) {
    return impl_->inspect_admission(PreparedPromptAccess::view(prompt), base, destination, source,
                                    shared_source, checkpoint, must_retain_private_source);
}

std::optional<ResourcePlan> Program::seal_identity(const AdmissionCandidate& admission,
                                                   const PreparedPrompt& prompt,
                                                   runtime::FinalScheduleIntent intent) {
    std::optional<AdmissionCandidate> sealed = impl_->seal_materialization(
        admission, PreparedPromptAccess::view(prompt), {}, {}, {}, {}, {}, {});
    if (!sealed) { return std::nullopt; }
    impl_->select_shared_captures(*sealed, PreparedPromptAccess::view(prompt),
                                  intent.shared_capture_frontiers);
    if (impl_->revalidate_materialization(*sealed, PreparedPromptAccess::view(prompt)) !=
        runtime::PreflightStatus::Ready) {
        return std::nullopt;
    }
    const bool needs_transfer = sealed->impl_->needs_transfer;
    return ResourcePlan(std::move(*sealed), impl_->resource_revision(), needs_transfer);
}

PressurePlanningSession
Program::begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                                 std::span<const runtime::PlanningCandidateId> candidate_ids,
                                 std::span<const ContinuationHandle* const> private_owners,
                                 std::span<const runtime::PlanningOwnerId> private_owner_ids,
                                 std::span<const SharedPrefixHandle* const> shared_owners,
                                 std::span<const runtime::PlanningOwnerId> shared_owner_ids) {
    using SessionImpl = detail::PressurePlanningSessionImpl;
    std::vector<SessionImpl::PhysicalCandidateBinding> physical_candidates;
    physical_candidates.reserve(candidates.size());
    for (const AdmissionCandidate* candidate : candidates) {
        if (candidate == nullptr || candidate->impl_ == nullptr) {
            throw std::invalid_argument("pressure planning candidate is empty");
        }
        physical_candidates.push_back(SessionImpl::PhysicalCandidateBinding{
            .state     = candidate->impl_.get(),
            .admission = candidate->impl_.get(),
        });
    }
    return PressurePlanningSession(std::make_unique<detail::PressurePlanningSessionImpl>(
        *impl_, physical_candidates, candidate_ids, private_owners, private_owner_ids,
        shared_owners, shared_owner_ids));
}

runtime::PrefillWork
Program::shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                           const PreparedPrompt& prompt,
                                           std::span<const std::uint32_t> frontiers) {
    if (impl_ == nullptr) { throw std::logic_error("Program is empty"); }
    return impl_->shared_capture_split_prefill_work(candidate, PreparedPromptAccess::view(prompt),
                                                    frontiers);
}

runtime::ContextTransactionReserveStatus
Program::start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                                    runtime::CancellationFlagView cancellation) {
    if (plan.revision_.value == 0 || plan.revision_ != impl_->resource_revision()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    return impl_->reserve_materialization(
        std::move(plan.admission_), PreparedPromptAccess::take(std::move(prompt)), cancellation);
}

std::optional<PersistentBackfillProof>
Program::prove_persistent_backfill(const RequestBasePlan& blocked_head,
                                   const ResourcePlan& candidate,
                                   std::span<const SequenceHandle> persistent_borrowers) const {
    if (candidate.revision_.value == 0 || candidate.revision_ != impl_->resource_revision() ||
        !impl_->persistent_backfill_safe(blocked_head, candidate.admission_,
                                         persistent_borrowers)) {
        return std::nullopt;
    }
    return PersistentBackfillProof(candidate.revision_);
}

ContextTransactionProgress
Program::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    return impl_->progress_context_transaction(cancellation);
}

void Program::finalize_context_transaction() noexcept { impl_->finalize_context_transaction(); }

bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }

PrefillProgress Program::advance_prefill(SequenceHandle sequence,
                                         runtime::ExecutionTiming* failed_timing,
                                         runtime::TokenMaskProvider* masks) {
    return impl_->advance_prefill(sequence, failed_timing, masks);
}

CaptureAssessment
Program::inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                         const SharedPrefixHandle* replacement,
                         std::optional<runtime::CheckpointRef> private_replacement,
                         bool permit_shared_publication) const {
    return impl_->inspect_capture(offer, exact_shared, replacement, private_replacement,
                                  permit_shared_publication);
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program::checkpoint_recovery_work(const ContinuationHandle& owner,
                                  runtime::CheckpointRef checkpoint) const {
    return impl_->checkpoint_recovery_work(owner, checkpoint);
}

CapturePressurePlanningSession Program::begin_capture_pressure_planning(
    const CaptureAssessment& assessment, std::span<const ContinuationHandle* const> private_owners,
    std::span<const runtime::PlanningOwnerId> private_owner_ids,
    std::span<const SharedPrefixHandle* const> shared_owners,
    std::span<const runtime::PlanningOwnerId> shared_owner_ids) {
    CapturePressureCandidate candidate(impl_->make_capture_physical_candidate(assessment));
    using SessionImpl = detail::PressurePlanningSessionImpl;
    const std::array physical_candidates{SessionImpl::PhysicalCandidateBinding{
        .state   = candidate.impl_.get(),
        .capture = candidate.impl_.get(),
    }};
    const std::array candidate_ids{CapturePressurePlanningSession::candidate_id()};
    PressurePlanningSession session(std::make_unique<detail::PressurePlanningSessionImpl>(
        *impl_, physical_candidates, candidate_ids, private_owners, private_owner_ids,
        shared_owners, shared_owner_ids));
    return CapturePressurePlanningSession(std::move(candidate), std::move(session));
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program::checkpoint_recovery_work(const SharedPrefixHandle& owner,
                                  runtime::CheckpointRef checkpoint) const {
    return impl_->checkpoint_recovery_work(owner, checkpoint);
}

bool Program::shared_capture_matches(const CaptureOffer& offer,
                                     const SharedPrefixHandle& shared) const {
    return impl_->shared_capture_matches(offer, shared);
}

void Program::skip_capture(CaptureOffer&& offer) { impl_->skip_capture(std::move(offer)); }

runtime::ContextTransactionReserveStatus
Program::reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                                const SharedPrefixHandle* replacement,
                                std::optional<runtime::CheckpointRef> private_replacement,
                                bool permit_shared_publication,
                                runtime::CancellationFlagView cancellation) {
    return impl_->reserve_active_capture(std::move(offer), exact_shared, replacement,
                                         private_replacement, permit_shared_publication,
                                         cancellation);
}

runtime::ContextTransactionReserveStatus Program::reserve_active_capture_with_pressure(
    CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
    const SharedPrefixHandle* replacement,
    std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
    CapturePressurePlan&& pressure, runtime::CancellationFlagView cancellation) {
    if (pressure.revision_.value == 0 || pressure.revision_ != impl_->resource_revision()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    return impl_->reserve_active_capture_with_pressure(
        std::move(offer), exact_shared, replacement, private_replacement, permit_shared_publication,
        std::move(pressure.pressure_), cancellation);
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed_timing,
                             runtime::TokenMaskProvider* masks) {
    return impl_->decode(sequences, budgets, failed_timing, masks);
}

bool Program::sampling_bypass_hooks() const noexcept { return impl_->sampling_bypass_hooks(); }

runtime::ExecutionTiming
Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                              std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                              std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                              runtime::ExecutionTiming* failed_timing) {
    return impl_->append_forced_tokens(sequences, row_major_tokens, row_stride,
                                       prefix_execution_splits, failed_timing);
}

CommitResult Program::commit(PendingBatch&& pending,
                             std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation observation,
                             runtime::ExecutionTiming* failed_timing) {
    return impl_->commit(std::move(pending), decisions, observation, failed_timing);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
    return impl_->abort_pending(std::move(pending));
}

FinishResult Program::finish(SequenceHandle sequence) noexcept { return impl_->finish(sequence); }

FinishResult Program::finish_cancelled(SequenceHandle sequence) noexcept {
    return impl_->finish_cancelled(sequence);
}

AbortResult Program::abort(SequenceHandle sequence) noexcept { return impl_->abort(sequence); }

std::optional<std::uint32_t>
Program::device_kv_lease_settlement_tokens(SequenceHandle sequence,
                                           std::uint32_t forced_span_tokens) const noexcept {
    return impl_->device_kv_lease_settlement_tokens(sequence, forced_span_tokens);
}

std::optional<DeviceKVLeaseShortfall>
Program::device_kv_lease_shortfall(SequenceHandle sequence) const noexcept {
    return impl_->device_kv_lease_shortfall(sequence);
}

bool Program::resume_device_kv_lease(SequenceHandle sequence) noexcept {
    return impl_->resume_device_kv_lease(sequence);
}

DeviceKVPages
Program::retained_device_kv_pages(const ContinuationHandle& continuation) const noexcept {
    return impl_->retained_device_kv_pages(continuation);
}

DeviceKVPages Program::retained_device_kv_pages(const SharedPrefixHandle& shared) const noexcept {
    return impl_->retained_device_kv_pages(shared);
}

DeviceKVPages Program::idle_device_kv_pages(const ContinuationHandle& continuation) const {
    return impl_->idle_device_kv_pages(continuation);
}

DeviceKVPages Program::idle_device_kv_pages(const SharedPrefixHandle& shared) const {
    return impl_->idle_device_kv_pages(shared);
}

std::vector<DeviceKVPages>
Program::releasable_device_kv_pages(std::span<const ContinuationHandle* const> continuations,
                                    std::span<const SharedPrefixHandle* const> shared) const {
    return impl_->releasable_device_kv_pages(continuations, shared);
}

DeviceKVPages Program::device_kv_available_pages() const noexcept {
    return impl_->device_kv_available_pages();
}

std::optional<ContinuationSummary>
Program::demote_retained_kv_for_lease(const ContinuationHandle& continuation,
                                      DeviceKVPages needed) noexcept {
    return impl_->demote_retained_kv_for_lease(continuation, needed);
}

std::optional<SharedPrefixSummary>
Program::demote_retained_shared_kv_for_lease(const SharedPrefixHandle& shared,
                                             DeviceKVPages needed) noexcept {
    return impl_->demote_retained_shared_kv_for_lease(shared, needed);
}

bool Program::idle_device_kv_offload() const noexcept { return impl_->idle_device_kv_offload(); }

std::optional<std::size_t>
Program::idle_device_kv_host_bytes(const ContinuationHandle& continuation) const noexcept {
    return impl_->idle_device_kv_host_bytes(continuation);
}

std::size_t Program::host_kv_free_bytes() const noexcept { return impl_->host_kv_free_bytes(); }

DeviceKVCompaction Program::compact_device_kv() { return impl_->compact_device_kv(); }

bool Program::disk_cache_enabled() const noexcept { return impl_->disk_kv != nullptr; }

std::uint32_t Program::disk_cache_idle_ms() const noexcept {
    return impl_->disk_kv != nullptr ? impl_->context_cache.disk_idle_ms : 0U;
}

std::uint32_t Program::disk_cache_flush_seconds() const noexcept {
    return impl_->disk_kv != nullptr ? impl_->context_cache.disk_flush_seconds : 0U;
}

bool Program::disk_cache_has_unsaved() const noexcept { return impl_->disk_cache_has_unsaved(); }

DiskCacheSave Program::disk_cache_save_one(const std::function<bool()>& abort) {
    return impl_->disk_cache_save_one(abort);
}

bool Program::disk_cache_drain(std::chrono::steady_clock::time_point deadline) {
    return impl_->disk_kv == nullptr || impl_->disk_kv->drain(deadline);
}

std::string Program::disk_cache_status() const { return impl_->disk_cache_status(); }

ReleaseResult Program::release_continuation(ContinuationHandle&& continuation) noexcept {
    return impl_->release_continuation(std::move(continuation));
}

std::optional<ContinuationSummary>
Program::drop_continuation_anchor(const ContinuationHandle& continuation,
                                  runtime::CheckpointRef anchor) noexcept {
    return impl_->drop_continuation_anchor(continuation, anchor);
}

ReleaseResult Program::release_shared_prefix(SharedPrefixHandle&& shared) noexcept {
    return impl_->release_shared_prefix(std::move(shared));
}

void Program::fail_all_cleanup() noexcept { impl_->fail_all_cleanup(); }

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    return impl_->isolated_request_feasible(base);
}

runtime::ProgramResourceRevision Program::resource_revision() const noexcept {
    return impl_->resource_revision();
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->physical_usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }

void Program::reset_memory_peaks() noexcept { impl_->reset_memory_peaks(); }

const TpTransportStatus& Program::tp_transport() const noexcept {
    return impl_->tp_transport_status;
}

bool Program::peer_exchange_timed_out() const noexcept {
    return impl_->peer_mailbox && impl_->peer_mailbox->hang_reported();
}

std::string Program::recover_peer_transport() { return impl_->recover_peer_transport(); }

bool gguf_routed_experts(const execution::Parameters& parameters) {
    return parameters.qwen4 && parameters.model.config().text.qwen4_routed_experts() &&
           !parameters.qwen4->layers.empty() && parameters.qwen4->layers.front().moe().experts.gguf;
}

std::int64_t offloaded_expert_slot_bytes(const execution::Parameters& parameters,
                                         const EngineOptions& options) {
    // A dense Qwen4Exp FFN has no routed experts, so no slot.
    if (!parameters.qwen4 || !parameters.model.config().text.qwen4_routed_experts() ||
        options.routed_experts.backend != RoutedExpertBackend::Offload) {
        return 0;
    }
    // Under MTP the MTP layer is one more cache layer (planning: Qwen4PersistentLayout).
    const bool mtp = options.speculative.backend == SpeculativeBackend::Mtp && parameters.qwen4_mtp;
    const auto layers = static_cast<std::int32_t>(parameters.qwen4->layers.size()) + (mtp ? 1 : 0);
    if (gguf_routed_experts(parameters)) {
        // ggml block banks: one slot of every cache layer, each its own size; the per-layer base
        // slot profile moves slots between layers of different sizes, so the larger of the
        // profile-weighted and the uniform bytes (an elastic column is uniform).
        std::vector<std::int64_t> bytes;
        for (const auto& layer : parameters.qwen4->layers) {
            bytes.push_back(execution::Qwen4OffloadExperts::gguf_slot_bytes(layer.moe().experts));
        }
        if (mtp) {
            bytes.push_back(execution::Qwen4OffloadExperts::gguf_slot_bytes(
                parameters.qwen4_mtp->layer.moe().experts));
        }
        constexpr std::int32_t kProbe = 100;
        const auto base               = execution::Qwen4OffloadExperts::layer_slot_profile(
            layers, static_cast<std::int32_t>(parameters.qwen4->layers.size()), kProbe, 0,
            static_cast<std::int32_t>(options.tp));
        std::int64_t uniform = 0, weighted = 0;
        for (std::size_t l = 0; l < bytes.size(); ++l) {
            uniform += bytes[l];
            weighted += bytes[l] * base[l];
        }
        return std::max(uniform, (weighted + kProbe - 1) / kProbe);
    }
    return execution::Qwen4OffloadExperts::slot_bytes(layers);
}

std::int64_t offloaded_expert_staging_bytes(const execution::Parameters& parameters,
                                            const EngineOptions& options) {
    if (offloaded_expert_slot_bytes(parameters, options) == 0) { return 0; }
    return static_cast<std::int64_t>(2U * options.routed_experts.prefetch_slots) *
           execution::Qwen4OffloadExperts::slot_bytes(1);
}

std::int64_t offloaded_expert_split_bytes(const execution::Parameters& parameters,
                                          const EngineOptions& options) {
    if (offloaded_expert_slot_bytes(parameters, options) == 0 ||
        !execution::staged_miss_split(options.routed_experts)) {
        return 0;
    }
    // Planning: four scratch planes, the I32 [512] table and the pick list, each arena-aligned.
    return execution::Qwen4OffloadExperts::slot_bytes(1) * execution::kMissSplitSlots + 6 * 4096;
}

SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                      DeviceContext& device, const EngineOptions& options) {
    return SequencePlanner(
        detail::make_sequence_planner_impl(parameters, nullptr, device, options));
}

SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                      const execution::Parameters& peer_parameters,
                                      DeviceContext& device, const EngineOptions& options) {
    return SequencePlanner(
        detail::make_sequence_planner_impl(parameters, &peer_parameters, device, options));
}

std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                        SequencePlan&& plan, DeviceContext& device,
                                        const StartupObserver& startup_observer) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.impl_->parameters != &parameters) {
        throw std::invalid_argument("sequence plan belongs to another model instance");
    }
    auto impl =
        std::make_unique<detail::ProgramImpl>(parameters, *plan.impl_, device, startup_observer);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                        const execution::Parameters& peer_parameters,
                                        SequencePlan&& plan, ExecutionContext& execution,
                                        const StartupObserver& startup_observer) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.impl_->parameters != &parameters) {
        throw std::invalid_argument("sequence plan belongs to another model instance");
    }
    if (execution.tp != 2 || plan.impl_->tp != 2 || !execution.dev[0] || !execution.dev[1]) {
        throw std::invalid_argument("tensor-parallel Program requires a two-device plan and "
                                    "ExecutionContext");
    }
    auto impl = std::make_unique<detail::ProgramImpl>(
        parameters, *plan.impl_, *execution.dev[0], startup_observer, &execution, &peer_parameters);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5 {

CaptureAssessment::CaptureAssessment()
    : implementation(std::make_shared<detail::CaptureAssessmentImpl>()) {}

AdmissionCandidate::AdmissionCandidate(
    std::unique_ptr<detail::AdmissionCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}

AdmissionCandidate::AdmissionCandidate(AdmissionCandidate&&) noexcept = default;

AdmissionCandidate& AdmissionCandidate::operator=(AdmissionCandidate&&) noexcept = default;

AdmissionCandidate::~AdmissionCandidate() = default;

CapturePressureCandidate::CapturePressureCandidate(
    std::unique_ptr<detail::CapturePressureCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}

CapturePressureCandidate::CapturePressureCandidate(CapturePressureCandidate&&) noexcept = default;

CapturePressureCandidate&
CapturePressureCandidate::operator=(CapturePressureCandidate&&) noexcept = default;

CapturePressureCandidate::~CapturePressureCandidate() = default;

const runtime::RequestPlanSummary& AdmissionCandidate::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const runtime::IdentityMaterializationAssessment&
AdmissionCandidate::identity_assessment() const noexcept {
    static const runtime::IdentityMaterializationAssessment empty;
    return impl_ != nullptr ? impl_->identity_assessment : empty;
}

} // namespace ninfer::models::qwen3_5
