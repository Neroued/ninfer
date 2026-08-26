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
        .digest       = impl_->prefix_digests.at(frontier),
        .frontier     = frontier,
        .identity_tag = impl_->prefix_identity_tag,
    };
}

template <>
AdmissionPlan<Variant>::AdmissionPlan(
    std::unique_ptr<detail::AdmissionPlanImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
AdmissionPlan<Variant>::AdmissionPlan(AdmissionPlan&& other) noexcept
    : impl_(std::move(other.impl_)) {}
template <>
AdmissionPlan<Variant>& AdmissionPlan<Variant>::operator=(AdmissionPlan&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}
template <>
AdmissionPlan<Variant>::~AdmissionPlan() { impl_.reset(); }

template <>
const runtime::RequestPlanSummary& AdmissionPlan<Variant>::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

template <>
runtime::ClaimDisposition AdmissionPlan<Variant>::source_disposition() const noexcept {
    return impl_ != nullptr ? impl_->source_disposition
                            : runtime::ClaimDisposition::ConsumedToActive;
}

template <>
bool AdmissionPlan<Variant>::needs_transfer() const noexcept {
    return impl_ != nullptr && impl_->needs_transfer;
}

template <>
runtime::PrefillWork AdmissionPlan<Variant>::remaining_prefill_work() const noexcept {
    return impl_ != nullptr ? impl_->remaining_prefill_work : runtime::PrefillWork{};
}

template <>
std::span<const runtime::ContextTransferRequirement>
AdmissionPlan<Variant>::transfer_requirements() const noexcept {
    return impl_ != nullptr
               ? std::span<const runtime::ContextTransferRequirement>(impl_->transfer_requirements)
               : std::span<const runtime::ContextTransferRequirement>{};
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
std::optional<AdmissionPlan<Variant>> Program<Variant>::inspect_admission(
    const PreparedPrompt& prompt, const RequestBasePlan<Variant>& base, runtime::LaneId destination,
    const ContinuationHandle<Variant>* source, const SharedPrefixHandle<Variant>* shared_source,
    std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source) {
    return impl_->inspect_admission(PreparedPromptAccess::view(prompt), base, destination, source,
                                    shared_source, checkpoint, must_retain_private_source);
}

template <>
std::vector<PressureOption>
Program<Variant>::inspect_pressure_options(const AdmissionPlan<Variant>& admission,
                                           const ContinuationHandle<Variant>& continuation) const {
    return impl_->inspect_pressure_options(admission, continuation);
}

template <>
PressureOption
Program<Variant>::inspect_eviction_option(const ContinuationHandle<Variant>& continuation) const {
    return impl_->inspect_eviction_option(continuation);
}

template <>
std::vector<PressureOption>
Program<Variant>::inspect_shared_pressure_options(const AdmissionPlan<Variant>& admission,
                                                  const SharedPrefixHandle<Variant>& shared) const {
    return impl_->inspect_shared_pressure_options(admission, shared);
}

template <>
PressureOption
Program<Variant>::inspect_shared_eviction_option(const SharedPrefixHandle<Variant>& shared) const {
    return impl_->inspect_shared_eviction_option(shared);
}

template <>
std::optional<ResourcePlan<Variant>> Program<Variant>::seal_resource_plan(
    const AdmissionPlan<Variant>& admission, const PreparedPrompt& prompt,
    std::span<const ContinuationHandle<Variant>* const> pressure_owners,
    std::span<const PressureOption> pressure_options,
    std::span<const SharedPrefixHandle<Variant>* const> shared_pressure_owners,
    std::span<const PressureOption> shared_pressure_options) {
    std::optional<AdmissionPlan<Variant>> sealed = impl_->seal_materialization(
        admission, PreparedPromptAccess::view(prompt), pressure_owners, pressure_options,
        shared_pressure_owners, shared_pressure_options);
    if (!sealed) { return std::nullopt; }
    return ResourcePlan<Variant>(std::move(*sealed), impl_->resource_revision());
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
CaptureAssessment
Program<Variant>::inspect_capture(const CaptureOffer<Variant>& offer,
                                  const SharedPrefixHandle<Variant>* exact_shared,
                                  const SharedPrefixHandle<Variant>* replacement,
                                  std::optional<runtime::CheckpointRef> private_replacement) const {
    return impl_->inspect_capture(offer, exact_shared, replacement, private_replacement);
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
runtime::ContextTransactionReserveStatus
Program<Variant>::reserve_active_capture(CaptureOffer<Variant>&& offer,
                                         const SharedPrefixHandle<Variant>* exact_shared,
                                         const SharedPrefixHandle<Variant>* replacement,
                                         std::optional<runtime::CheckpointRef> private_replacement,
                                         runtime::CancellationFlagView cancellation) {
    return impl_->reserve_active_capture(std::move(offer), exact_shared, replacement,
                                         private_replacement, cancellation);
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
