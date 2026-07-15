#include <ninfer/targets/qwen3_6_27b_rtx5090/package.h>

#include "targets/qwen3_6_27b_rtx5090/impl/frontend/frontend.h"
#include "targets/qwen3_6_27b_rtx5090/impl/load/bindings.h"
#include "targets/qwen3_6_27b_rtx5090/impl/program/layouts.h"
#include "targets/qwen3_6_27b_rtx5090/impl/program/program.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {

class LoadPlan::Impl {
public:
    explicit Impl(ArtifactLoadPlan target_plan) : plan(std::move(target_plan)) {}

    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

Program::Program(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

Program::~Program() noexcept = default;

RequestPlan Program::plan_request(const PreparedPrompt& prompt,
                                  const ExecutionOptions& options) const {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    return impl_->plan_request(*prompt.data_, options);
}

runtime::BeginResult Program::begin(PreparedPrompt&& prompt, RequestPlan&& plan,
                                    runtime::TransientRegion transient) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    auto data = std::move(prompt.data_);
    return impl_->begin(std::move(*data), std::move(plan), transient);
}

runtime::GeneratedRound Program::decode_round(runtime::RoundBudget budget) {
    return impl_->decode_round(budget);
}

void Program::resolve_pending(std::uint32_t accepted_tokens, bool terminal) {
    impl_->resolve_pending(accepted_tokens, terminal);
}

void Program::finish_active() { impl_->finish_active(); }

void Program::abort_request() noexcept { impl_->abort_request(); }

std::uint32_t Program::materialized_tokens() const noexcept { return impl_->materialized_tokens(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }

SpeculativeStats Program::speculative_stats() const { return impl_->speculative_stats(); }

GenerationTimings Program::generation_timings() const noexcept {
    return impl_->generation_timings();
}

void Program::reset_memory_peaks() noexcept { impl_->reset_memory_peaks(); }

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail

namespace ninfer::targets::qwen3_6_27b_rtx5090 {

void Package::preflight(DeviceContext& device, const EngineOptions& options) {
    detail::validate_target_options(device, options);
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder) {
    return LoadPlan(std::make_unique<LoadPlan::Impl>(detail::bind_artifact(binder)));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(std::move(plan.impl_->plan.bindings),
                                                    std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return detail::FrontendFactory::create_registered(model.impl_->data.frontend);
}

Package::SequencePlan Package::plan_sequence(const LoadedModel& model, DeviceContext& device,
                                             const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return SequencePlan(detail::plan_sequence_impl(device, options));
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    auto impl = std::make_unique<Program::Impl>(model.impl_->data, *plan.impl_, device);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

} // namespace ninfer::targets::qwen3_6_27b_rtx5090
