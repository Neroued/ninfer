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

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept = default;

RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;

RequestBasePlan::~RequestBasePlan() = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
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

ContextTransactionProgress
Program::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    return impl_->progress_context_transaction(cancellation);
}

void Program::finalize_context_transaction() noexcept { impl_->finalize_context_transaction(); }

bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }

PrefillProgress Program::advance_prefill(SequenceHandle sequence,
                                         runtime::ExecutionTiming* failed_timing) {
    return impl_->advance_prefill(sequence, failed_timing);
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed_timing) {
    return impl_->decode(sequences, budgets, failed_timing);
}

runtime::ExecutionTiming
Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                              std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                              runtime::ExecutionTiming* failed_timing) {
    return impl_->append_forced_tokens(sequences, row_major_tokens, row_stride, failed_timing);
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

AbortResult Program::abort(SequenceHandle sequence) noexcept { return impl_->abort(sequence); }

void Program::fail_all_cleanup() noexcept {
    impl_->fail_all_cleanup(detail::ProgramCleanup::Failure);
}

void Program::shutdown_cleanup() noexcept {
    impl_->fail_all_cleanup(detail::ProgramCleanup::Shutdown);
}

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    return impl_->isolated_request_feasible(base);
}

HybridAdmissionQuote Program::hybrid_quote(const PreparedPrompt& prompt,
                                           const RequestBasePlan& base,
                                           runtime::LaneId destination) {
    return impl_->hybrid_quote(PreparedPromptAccess::view(prompt), base, destination);
}

runtime::ContextTransactionReserveStatus
Program::hybrid_reserve_materialization(HybridAdmissionQuote&& quote, PreparedPrompt&& prompt,
                                        runtime::CancellationFlagView cancellation) {
    // The prompt is taken only once the reservation will succeed: a rejected quote leaves the
    // waiting request intact for a later admission attempt.
    if (!impl_->hybrid_reservable(quote, cancellation)) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    return impl_->hybrid_reserve_materialization(
        std::move(quote), PreparedPromptAccess::view(prompt),
        [&prompt]() { return PreparedPromptAccess::take(std::move(prompt)); }, cancellation);
}

std::optional<std::uint32_t> Program::hybrid_prefetch(const PreparedPrompt& prompt,
                                                      const RequestBasePlan& base) {
    return impl_->hybrid_prefetch(PreparedPromptAccess::view(prompt), base);
}

std::optional<PersistentBackfillProof>
Program::prove_persistent_backfill(const RequestBasePlan& blocked_head,
                                   const HybridAdmissionQuote& candidate,
                                   std::span<const SequenceHandle> persistent_borrowers) const {
    if (!impl_->persistent_backfill_safe(blocked_head, candidate, persistent_borrowers)) {
        return std::nullopt;
    }
    return PersistentBackfillProof(impl_->resource_revision());
}

std::uint32_t Program::hybrid_prefetch_room() const noexcept {
    return impl_->hybrid_prefetch_room();
}

HybridPrefixCacheStats Program::hybrid_stats() const noexcept { return impl_->hybrid_stats(); }

void Program::set_hybrid_cost(const runtime::prefix_cache::CacheCostModel& cost) {
    impl_->set_hybrid_cost(cost);
}

HybridCachePersistence Program::attach_hybrid_cache_file(const std::filesystem::path& path,
                                                         std::string fingerprint,
                                                         const StartupObserver& observer) {
    return impl_->attach_hybrid_cache_file(path, std::move(fingerprint), observer);
}

std::optional<HybridCachePersistence> Program::hybrid_shutdown_save() const {
    return impl_->hybrid_shutdown_save();
}

runtime::ProgramResourceRevision Program::resource_revision() const noexcept {
    return impl_->resource_revision();
}

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
    auto impl =
        std::make_unique<detail::ProgramImpl>(parameters, *plan.impl_, device, startup_observer);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

} // namespace ninfer::models::qwen3_5
