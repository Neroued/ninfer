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
SequencePlan<Variant>::~SequencePlan() = default;

template <>
std::uint32_t SequencePlan<Variant>::capacity() const noexcept {
    return impl_ != nullptr ? impl_->capacity : 0;
}

template <>
std::size_t SequencePlan<Variant>::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->device_reservation_bytes : 0;
}

template <>
RequestPlan<Variant>::RequestPlan(std::unique_ptr<detail::RequestPlanImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
RequestPlan<Variant>::RequestPlan(RequestPlan&& other) noexcept
    : impl_(std::move(other.impl_)) {}
template <>
RequestPlan<Variant>& RequestPlan<Variant>::operator=(RequestPlan&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}
template <>
RequestPlan<Variant>::~RequestPlan() = default;

template <>
const runtime::RequestPlanSummary& RequestPlan<Variant>::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

template <>
Program<Variant>::Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
Program<Variant>::~Program() noexcept = default;

template <>
RequestPlan<Variant> Program<Variant>::plan_request(const PreparedPrompt& prompt,
                                                    const ExecutionOptions& options) const {
    return impl_->plan_request(PreparedPromptAccess::view(prompt), options);
}

template <>
runtime::BeginResult Program<Variant>::begin(PreparedPrompt&& prompt, RequestPlan<Variant>&& plan,
                                             runtime::TransientRegion transient) {
    return impl_->begin(PreparedPromptAccess::take(std::move(prompt)), std::move(plan), transient);
}

template <>
runtime::GeneratedRound Program<Variant>::decode_round(runtime::RoundBudget budget) {
    return impl_->decode_round(budget);
}

template <>
void Program<Variant>::resolve_pending(std::uint32_t accepted_tokens, bool terminal) {
    impl_->resolve_pending(accepted_tokens, terminal);
}

template <>
void Program<Variant>::finish_active() {
    impl_->finish_active();
}

template <>
void Program<Variant>::abort_request() noexcept {
    impl_->abort_request();
}

template <>
std::uint32_t Program<Variant>::materialized_tokens() const noexcept {
    return impl_->materialized_tokens();
}

template <>
MemorySummary Program<Variant>::memory_summary() const noexcept {
    return impl_->memory_summary();
}

template <>
SpeculativeStats Program<Variant>::speculative_stats() const {
    return impl_->speculative_stats();
}

template <>
GenerationTimings Program<Variant>::generation_timings() const noexcept {
    return impl_->generation_timings();
}

template <>
void Program<Variant>::reset_memory_peaks() noexcept {
    impl_->reset_memory_peaks();
}

template <>
SequencePlan<Variant> plan_sequence<Variant>(DeviceContext& device, const EngineOptions& options) {
    return SequencePlan<Variant>(
        detail::NINFER_QWEN36_RUNTIME_NS::plan_sequence_impl(device, options));
}

template <>
std::unique_ptr<Program<Variant>> create_program<Variant>(const Variant::ModelView& model,
                                                          SequencePlan<Variant>&& plan,
                                                          DeviceContext& device) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    auto impl = std::make_unique<detail::ProgramImpl<Variant>>(model, *plan.impl_, device);
    plan.impl_.reset();
    return std::unique_ptr<Program<Variant>>(new Program<Variant>(std::move(impl)));
}

} // namespace ninfer::targets::qwen3_6
