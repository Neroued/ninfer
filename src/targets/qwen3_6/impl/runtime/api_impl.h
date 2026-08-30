#include "targets/qwen3_6/impl/runtime/instance.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6 {

using detail::NINFER_QWEN36_RUNTIME_NS::Variant;

template <>
SequencePlan<Variant>::SequencePlan(
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
SequencePlan<Variant>::SequencePlan(SequencePlan&& other) noexcept
    : impl_(std::move(other.impl_)) {}
template <>
SequencePlan<Variant>& SequencePlan<Variant>::operator=(SequencePlan&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}
template <>
SequencePlan<Variant>::~SequencePlan() { impl_.reset(); }

template <>
std::uint32_t SequencePlan<Variant>::capacity() const noexcept {
    return impl_ != nullptr ? impl_->capacity : 0;
}

template <>
std::uint32_t SequencePlan<Variant>::kv_capacity() const noexcept {
    return impl_ != nullptr ? impl_->kv_capacity : 0;
}

template <>
std::uint32_t SequencePlan<Variant>::max_concurrency() const noexcept {
    return impl_ != nullptr ? impl_->max_concurrency : 0;
}

template <>
std::size_t SequencePlan<Variant>::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->device_reservation_bytes : 0;
}

template <>
std::size_t SequencePlan<Variant>::workspace_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->workspace.capacity : 0;
}

template <>
SequencePlanner<Variant>::SequencePlanner(
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
SequencePlanner<Variant>::SequencePlanner(SequencePlanner&& other) noexcept
    : impl_(std::move(other.impl_)) {}
template <>
SequencePlanner<Variant>& SequencePlanner<Variant>::operator=(SequencePlanner&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}
template <>
SequencePlanner<Variant>::~SequencePlanner() { impl_.reset(); }

template <>
const runtime::SequenceCapacityCurve& SequencePlanner<Variant>::capacity_curve() const noexcept {
    static const runtime::SequenceCapacityCurve empty;
    return impl_ != nullptr ? impl_->curve : empty;
}

template <>
SequencePlan<Variant> SequencePlanner<Variant>::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return SequencePlan<Variant>(detail::NINFER_QWEN36_RUNTIME_NS::finalize_sequence_plan_impl(
        std::move(impl_), main_page_groups));
}

template <>
RequestBasePlan<Variant>::RequestBasePlan(
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
RequestBasePlan<Variant>::RequestBasePlan(RequestBasePlan&& other) noexcept
    : impl_(std::move(other.impl_)) {}
template <>
RequestBasePlan<Variant>& RequestBasePlan<Variant>::operator=(RequestBasePlan&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}
template <>
RequestBasePlan<Variant>::~RequestBasePlan() { impl_.reset(); }

template <>
const runtime::RequestPlanSummary& RequestBasePlan<Variant>::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

template <>
const PreparedContextCache& RequestBasePlan<Variant>::context_cache() const noexcept {
    static const PreparedContextCache empty;
    return impl_ != nullptr ? impl_->context_cache : empty;
}

template <>
std::optional<PrefixShortlistKey>
RequestBasePlan<Variant>::prefix_shortlist_key(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr || frontier == 0 || frontier > impl_->prefix_digests.size()) {
        return std::nullopt;
    }
    return PrefixShortlistKey{
        .digests      = impl_->prefix_digests.at(frontier),
        .frontier     = frontier,
        .identity_tag = impl_->prefix_identity_tag,
    };
}

template <>
std::optional<runtime::PrefillWork>
RequestBasePlan<Variant>::shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr) { return std::nullopt; }
    const auto found = std::find_if(impl_->shared_candidates.begin(),
                                    impl_->shared_candidates.end(), [&](const auto& candidate) {
                                        return candidate.frontier == frontier && candidate.identity;
                                    });
    return found == impl_->shared_candidates.end()
               ? std::nullopt
               : std::optional<runtime::PrefillWork>(found->identity->rebuild_work);
}

template <>
PressurePlanningSession<Variant>::PressurePlanningSession(
    std::unique_ptr<detail::PressurePlanningSessionImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
PressurePlanningSession<Variant>::PressurePlanningSession(PressurePlanningSession&&) noexcept =
    default;

template <>
PressurePlanningSession<Variant>&
PressurePlanningSession<Variant>::operator=(PressurePlanningSession&&) noexcept = default;

template <>
PressurePlanningSession<Variant>::~PressurePlanningSession() = default;

template <>
PressureTargetHandle PressurePlanningSession<Variant>::identity_target(
    const AdmissionCandidate<Variant>& candidate) const {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->identity_target(candidate);
}

template <>
PressureTargetHandle PressurePlanningSession<Variant>::root_maximal_target(
    const AdmissionCandidate<Variant>& root_candidate) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->root_maximal_target(root_candidate);
}

template <>
runtime::PressureTargetAssessment
PressurePlanningSession<Variant>::assess(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->assess(target);
}

template <>
PreparedPressureExpansion<Variant>
PressurePlanningSession<Variant>::prepare_expansion(PressureTargetHandle parent) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->prepare_expansion(parent);
}

template <>
PressureExpansionView
PressurePlanningSession<Variant>::commit_expansion(PreparedPressureExpansion<Variant>&& prepared) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->commit_expansion(std::move(prepared));
}

template <>
void PressurePlanningSession<Variant>::discard_expansion(
    PreparedPressureExpansion<Variant>&& prepared) noexcept {
    if (impl_ != nullptr) { impl_->discard_expansion(std::move(prepared)); }
}

template <>
std::optional<ResourcePlan<Variant>>
PressurePlanningSession<Variant>::seal(PressureTargetHandle target, const PreparedPrompt& prompt) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    std::optional<AdmissionCandidate<Variant>> sealed =
        impl_->seal(target, PreparedPromptAccess::view(prompt));
    if (!sealed) { return std::nullopt; }
    const bool needs_transfer = sealed->impl_->needs_transfer;
    return ResourcePlan<Variant>(std::move(*sealed), impl_->resource_revision, needs_transfer);
}

template <>
std::optional<CapturePressurePlan<Variant>>
PressurePlanningSession<Variant>::seal_capture(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    std::optional<AdmissionCandidate<Variant>> sealed = impl_->seal_capture(target);
    if (!sealed) { return std::nullopt; }
    return CapturePressurePlan<Variant>(std::move(*sealed), impl_->resource_revision);
}

template <>
Program<Variant>::Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
Program<Variant>::~Program() noexcept { impl_.reset(); }

template <>
RequestBasePlan<Variant>
Program<Variant>::plan_request(const PreparedPrompt& prompt,
                               const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(PreparedPromptAccess::view(prompt), options);
}

template <>
std::vector<float> Program<Variant>::causal_score(PreparedPrompt&& prompt,
                                                  std::uint32_t first_target) {
    return impl_->causal_score(PreparedPromptAccess::take(std::move(prompt)), first_target);
}

template <>
std::optional<AdmissionCandidate<Variant>> Program<Variant>::inspect_admission(
    const PreparedPrompt& prompt, const RequestBasePlan<Variant>& base, runtime::LaneId destination,
    const ContinuationHandle<Variant>* source, const SharedPrefixHandle<Variant>* shared_source,
    std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source,
    const runtime::ContextMachineCostModel& machine_cost) {
    return impl_->inspect_admission(PreparedPromptAccess::view(prompt), base, destination, source,
                                    shared_source, checkpoint, must_retain_private_source,
                                    machine_cost);
}

template <>
std::optional<ResourcePlan<Variant>>
Program<Variant>::seal_identity(const AdmissionCandidate<Variant>& admission,
                                const PreparedPrompt& prompt) {
    std::optional<AdmissionCandidate<Variant>> sealed =
        impl_->seal_materialization(admission, PreparedPromptAccess::view(prompt), {}, {}, {}, {});
    if (!sealed) { return std::nullopt; }
    const bool needs_transfer = sealed->impl_->needs_transfer;
    return ResourcePlan<Variant>(std::move(*sealed), impl_->resource_revision(), needs_transfer);
}

template <>
PressurePlanningSession<Variant> Program<Variant>::begin_pressure_planning(
    const runtime::ContextMachineCostModel& machine_cost,
    std::span<const AdmissionCandidate<Variant>* const> candidates,
    std::span<const ContinuationHandle<Variant>* const> private_owners,
    std::span<const std::uint32_t> private_owner_ordinals,
    std::span<const SharedPrefixHandle<Variant>* const> shared_owners,
    std::span<const std::uint32_t> shared_owner_ordinals) {
    return PressurePlanningSession<Variant>(
        std::make_unique<detail::PressurePlanningSessionImpl<Variant>>(
            *impl_, machine_cost, candidates, private_owners, private_owner_ordinals, shared_owners,
            shared_owner_ordinals));
}

template <>
void Program<Variant>::select_shared_captures(ResourcePlan<Variant>& plan,
                                              const PreparedPrompt& prompt,
                                              std::span<const std::uint32_t> frontiers) {
    if (plan.revision_ == 0 || plan.revision_ != impl_->resource_revision()) {
        throw std::logic_error("shared capture selection observes a stale resource plan");
    }
    impl_->select_shared_captures(plan.admission_, PreparedPromptAccess::view(prompt), frontiers);
}

template <>
std::uint64_t Program<Variant>::shared_capture_split_cost_ns(
    const ResourcePlan<Variant>& plan, const PreparedPrompt& prompt,
    std::span<const std::uint32_t> frontiers,
    const runtime::ContextMachineCostModel& machine_cost) {
    if (impl_ == nullptr) { throw std::logic_error("Program is empty"); }
    return impl_->shared_capture_split_cost_ns(plan.admission_, PreparedPromptAccess::view(prompt),
                                               frontiers, machine_cost);
}

template <>
runtime::ContextTransactionReserveStatus
Program<Variant>::start_resource_transaction(ResourcePlan<Variant>&& plan, PreparedPrompt&& prompt,
                                             runtime::CancellationFlagView cancellation) {
    if (plan.revision_ == 0 || plan.revision_ != impl_->resource_revision()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    return impl_->reserve_materialization(
        std::move(plan.admission_), PreparedPromptAccess::take(std::move(prompt)), cancellation);
}

template <>
std::optional<PersistentBackfillProof<Variant>> Program<Variant>::prove_persistent_backfill(
    const RequestBasePlan<Variant>& blocked_head, const ResourcePlan<Variant>& candidate,
    std::span<const SequenceHandle<Variant>> persistent_borrowers) const {
    if (candidate.revision_ == 0 || candidate.revision_ != impl_->resource_revision() ||
        !impl_->persistent_backfill_safe(blocked_head, candidate.admission_,
                                         persistent_borrowers)) {
        return std::nullopt;
    }
    return PersistentBackfillProof<Variant>(candidate.revision_);
}

template <>
ContextTransactionProgress<Variant>
Program<Variant>::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    return impl_->progress_context_transaction(cancellation);
}

template <>
void Program<Variant>::finalize_context_transaction() noexcept {
    impl_->finalize_context_transaction();
}

template <>
bool Program<Variant>::has_context_transaction() const noexcept {
    return impl_->has_context_transaction();
}

template <>
PrefillProgress<Variant>
Program<Variant>::advance_prefill(SequenceHandle<Variant> sequence,
                                  runtime::ExecutionTiming* failed_timing) {
    return impl_->advance_prefill(sequence, failed_timing);
}

template <>
CaptureAssessment Program<Variant>::inspect_capture(
    const CaptureOffer<Variant>& offer, const SharedPrefixHandle<Variant>* exact_shared,
    const SharedPrefixHandle<Variant>* replacement,
    std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
    const runtime::ContextMachineCostModel& machine_cost) const {
    return impl_->inspect_capture(offer, exact_shared, replacement, private_replacement,
                                  permit_shared_publication, machine_cost);
}

template <>
std::uint64_t Program<Variant>::checkpoint_recovery_ns(
    const ContinuationHandle<Variant>& owner, runtime::CheckpointRef checkpoint,
    const runtime::ContextMachineCostModel& machine_cost) const {
    return impl_->checkpoint_recovery_ns(owner, checkpoint, machine_cost);
}

template <>
AdmissionCandidate<Variant> Program<Variant>::make_capture_pressure_candidate(
    const CaptureAssessment& assessment,
    const runtime::ContextMachineCostModel& machine_cost) const {
    return impl_->make_capture_pressure_candidate(assessment, machine_cost);
}

template <>
std::uint64_t Program<Variant>::checkpoint_recovery_ns(
    const SharedPrefixHandle<Variant>& owner, runtime::CheckpointRef checkpoint,
    const runtime::ContextMachineCostModel& machine_cost) const {
    return impl_->checkpoint_recovery_ns(owner, checkpoint, machine_cost);
}

template <>
bool Program<Variant>::shared_capture_matches(const CaptureOffer<Variant>& offer,
                                              const SharedPrefixHandle<Variant>& shared) const {
    return impl_->shared_capture_matches(offer, shared);
}

template <>
void Program<Variant>::skip_capture(CaptureOffer<Variant>&& offer) {
    impl_->skip_capture(std::move(offer));
}

template <>
runtime::ContextTransactionReserveStatus Program<Variant>::reserve_active_capture(
    CaptureOffer<Variant>&& offer, const SharedPrefixHandle<Variant>* exact_shared,
    const SharedPrefixHandle<Variant>* replacement,
    std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
    const runtime::ContextMachineCostModel& machine_cost,
    runtime::CancellationFlagView cancellation) {
    return impl_->reserve_active_capture(std::move(offer), exact_shared, replacement,
                                         private_replacement, permit_shared_publication,
                                         machine_cost, cancellation);
}

template <>
runtime::ContextTransactionReserveStatus Program<Variant>::reserve_active_capture_with_pressure(
    CaptureOffer<Variant>&& offer, const SharedPrefixHandle<Variant>* exact_shared,
    const SharedPrefixHandle<Variant>* replacement,
    std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
    CapturePressurePlan<Variant>&& pressure, const runtime::ContextMachineCostModel& machine_cost,
    runtime::CancellationFlagView cancellation) {
    if (pressure.revision_ == 0 || pressure.revision_ != impl_->resource_revision()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    return impl_->reserve_active_capture_with_pressure(
        std::move(offer), exact_shared, replacement, private_replacement, permit_shared_publication,
        std::move(pressure.pressure_), machine_cost, cancellation);
}

template <>
PendingBatch<Variant> Program<Variant>::decode(std::span<const SequenceHandle<Variant>> sequences,
                                               std::span<const runtime::RoundBudget> budgets,
                                               runtime::ExecutionTiming* failed_timing) {
    return impl_->decode(sequences, budgets, failed_timing);
}

template <>
runtime::ExecutionTiming Program<Variant>::append_forced_tokens(
    std::span<const SequenceHandle<Variant>> sequences, std::span<const TokenId> row_major_tokens,
    std::uint32_t row_stride, runtime::ExecutionTiming* failed_timing) {
    return impl_->append_forced_tokens(sequences, row_major_tokens, row_stride, failed_timing);
}

template <>
CommitResult<Variant> Program<Variant>::commit(PendingBatch<Variant>&& pending,
                                               std::span<const runtime::CommitDecision> decisions,
                                               runtime::CommitObservation observation,
                                               runtime::ExecutionTiming* failed_timing) {
    return impl_->commit(std::move(pending), decisions, observation, failed_timing);
}

template <>
DiscardResult<Variant> Program<Variant>::abort_pending(PendingBatch<Variant>&& pending) noexcept {
    return impl_->abort_pending(std::move(pending));
}

template <>
FinishResult<Variant> Program<Variant>::finish(SequenceHandle<Variant> sequence) noexcept {
    return impl_->finish(sequence);
}

template <>
AbortResult<Variant> Program<Variant>::abort(SequenceHandle<Variant> sequence) noexcept {
    return impl_->abort(sequence);
}

template <>
ReleaseResult<Variant>
Program<Variant>::release_continuation(ContinuationHandle<Variant>&& continuation) noexcept {
    return impl_->release_continuation(std::move(continuation));
}

template <>
ReleaseResult<Variant>
Program<Variant>::release_shared_prefix(SharedPrefixHandle<Variant>&& shared) noexcept {
    return impl_->release_shared_prefix(std::move(shared));
}

template <>
void Program<Variant>::fail_all_cleanup() noexcept {
    impl_->fail_all_cleanup();
}

template <>
bool Program<Variant>::isolated_request_feasible(
    const RequestBasePlan<Variant>& base) const noexcept {
    return impl_->isolated_request_feasible(base);
}

template <>
std::uint64_t Program<Variant>::resource_revision() const noexcept {
    return impl_->resource_revision();
}

template <>
PhysicalUsageSnapshot Program<Variant>::physical_usage() const noexcept {
    return impl_->physical_usage();
}

template <>
MemorySummary Program<Variant>::memory_summary() const noexcept {
    return impl_->memory_summary();
}

template <>
void Program<Variant>::reset_memory_peaks() noexcept {
    impl_->reset_memory_peaks();
}

template <>
SequencePlanner<Variant> make_sequence_planner<Variant>(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        Variant::WeightsProfile weights_profile) {
    return SequencePlanner<Variant>(detail::NINFER_QWEN36_RUNTIME_NS::make_sequence_planner_impl(
        device, options, weights_profile));
}

template <>
std::unique_ptr<Program<Variant>>
create_program<Variant>(const Variant::ModelView& model, Variant::WeightsProfile weights_profile,
                        SequencePlan<Variant>&& plan, DeviceContext& device) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.impl_->weights_profile != weights_profile) {
        throw std::invalid_argument(
            "loaded model weights profile does not match the sequence plan");
    }
    auto impl = std::make_unique<detail::ProgramImpl<Variant>>(model, *plan.impl_, device);
    plan.impl_.reset();
    return std::unique_ptr<Program<Variant>>(new Program<Variant>(std::move(impl)));
}

} // namespace ninfer::targets::qwen3_6
