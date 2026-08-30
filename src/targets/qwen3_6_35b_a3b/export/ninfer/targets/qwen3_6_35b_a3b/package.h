#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::qwen3_6_35b_a3b {

struct Package;

namespace detail {

struct Variant;

enum class WeightsProfile : std::uint8_t {
    GroupwiseInt,
};

using Frontend        = qwen3_6::Frontend;
using PreparedPrompt  = qwen3_6::PreparedPrompt;
using OutputSession   = qwen3_6::OutputSession;
using PublishedOutput = qwen3_6::PublishedOutput;

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_6_35b_a3b::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_6_35b_a3b::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "qwen3.6-35b-a3b";
    static constexpr std::string_view target_key = "qwen3_6_35b_a3b";

    using WeightsProfile             = detail::WeightsProfile;
    using LoadPlan                   = detail::LoadPlan;
    using LoadedModel                = detail::LoadedModel;
    using Frontend                   = detail::Frontend;
    using PreparedPrompt             = detail::PreparedPrompt;
    using OutputSession              = detail::OutputSession;
    using PublishedOutput            = detail::PublishedOutput;
    using SequencePlanner            = qwen3_6::SequencePlanner<detail::Variant>;
    using SequencePlan               = qwen3_6::SequencePlan<detail::Variant>;
    using RequestBasePlan            = qwen3_6::RequestBasePlan<detail::Variant>;
    using AdmissionCandidate         = qwen3_6::AdmissionCandidate<detail::Variant>;
    using ResourcePlan               = qwen3_6::ResourcePlan<detail::Variant>;
    using PersistentBackfillProof    = qwen3_6::PersistentBackfillProof<detail::Variant>;
    using SequenceHandle             = qwen3_6::SequenceHandle<detail::Variant>;
    using ContinuationHandle         = qwen3_6::ContinuationHandle<detail::Variant>;
    using SharedPrefixHandle         = qwen3_6::SharedPrefixHandle<detail::Variant>;
    using CaptureOffer               = qwen3_6::CaptureOffer<detail::Variant>;
    using CacheSessionKey            = qwen3_6::PreparedSessionKey;
    using ContinuationSummary        = qwen3_6::ContinuationSummary;
    using SharedPrefixSummary        = qwen3_6::SharedPrefixSummary;
    using PressurePlanningSession    = qwen3_6::PressurePlanningSession<detail::Variant>;
    using PressureTargetHandle       = qwen3_6::PressureTargetHandle;
    using CapturePressurePlan        = qwen3_6::CapturePressurePlan<detail::Variant>;
    using MaterializationResult      = qwen3_6::MaterializationResult<detail::Variant>;
    using ContextTransactionProgress = qwen3_6::ContextTransactionProgress<detail::Variant>;
    using CaptureAssessment          = qwen3_6::CaptureAssessment;
    using ActiveCaptureResult        = qwen3_6::ActiveCaptureResult<detail::Variant>;
    using PendingBatch               = qwen3_6::PendingBatch<detail::Variant>;
    using StartResult                = qwen3_6::StartResult<detail::Variant>;
    using PrefillProgress            = qwen3_6::PrefillProgress<detail::Variant>;
    using CommitResult               = qwen3_6::CommitResult<detail::Variant>;
    using DiscardResult              = qwen3_6::DiscardResult<detail::Variant>;
    using FinishResult               = qwen3_6::FinishResult<detail::Variant>;
    using AbortResult                = qwen3_6::AbortResult<detail::Variant>;
    using ReleaseResult              = qwen3_6::ReleaseResult<detail::Variant>;
    using Program                    = qwen3_6::Program<detail::Variant>;

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model,
                                                const EngineOptions& options);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::qwen3_6_35b_a3b
} // namespace ninfer
