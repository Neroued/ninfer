#pragma once

#include "models/common/graph/patch_site.h"

#include <optional>
#include <variant>

namespace ninfer::models {

using GraphBody             = std::function<void()>;
using StreamSectionRecorder = std::function<void(cudaStream_t, const GraphBody&)>;

struct ReadoutCapture {
    std::vector<ContextBucketBinding> readouts;
    ReadoutBody body;
};

struct ProposalStepCapture {
    ReadoutCapture forward;
    SlotPatchSite::Capture handoff;
};

struct ProposalCapture {
    GraphBody prefix;
    std::vector<ProposalStepCapture> steps;
    cudaStream_t handoff_stream = nullptr;
    GraphBody before_readback, readback, drafts_ready, verify_ids;
};

struct ProposalReplay {
    std::span<const ops::ContextBucket> buckets;
    std::optional<std::uint32_t> slot;
    bool readback = false;
};

// The same proposal sequence can be recorded alone or before the resident target.
class ProposalBody {
public:
    ProposalBody(DeviceContext&, const ProposalCapture&);

    void record(CudaExecutionView, ParallelExecution*, const ProposalCapture&, bool gate_steps);
    void finish(const CommunicationGraphWork&);
    void add_calibration(DecodeGraphExecutable&, GraphPatchCalibration&);

    [[nodiscard]] const CommunicationGraphWork*
    select(DecodeGraphExecutable&, const ProposalReplay&, std::uint32_t active_steps);

private:
    struct Step {
        std::vector<ContextBucketSite> sites;
        std::vector<std::size_t> bucket_offsets;
        std::optional<SlotPatchSite> handoff;
    };

    std::vector<Step> steps_;
    std::optional<CudaCompletionEvent> fork_, join_;
    DecodeGraphSection draft_readback_;
    MtpGraphSteps gates_;
};

struct ResidentTargetCapture {
    std::vector<ContextBucketBinding> readouts;
    SlotPatchSite::Capture handoff;
    // The readback stream must already participate in this capture.
    std::function<void(const ReadoutRecorder&, const GraphBody& handoff, const GraphBody& head,
                       const StreamSectionRecorder& readback)>
        body;
    GraphBody head;
};

struct OffloadedLayerCapture {
    ExpertBankId bank;
    moe::TokenGeometry geometry;
    ReadoutCapture mixer;
    moe::GpuBody gpu;
    GraphBody feature;
};

struct InterLayerHookCapture {
    std::size_t layer; // The hook precedes this layer, or precedes the suffix at layers.size().
    SlotPatchSite::Capture body;
};

struct OffloadedTargetCapture {
    Tensor output;
    std::int32_t physical_tokens;
    GraphBody prefix;
    std::vector<std::variant<ReadoutCapture, OffloadedLayerCapture>> layers;
    std::optional<InterLayerHookCapture> hook;
    GraphBody suffix, head;
    GraphBody after_head;
};

using TargetCapture = std::variant<ResidentTargetCapture, OffloadedTargetCapture>;

struct TargetReplay {
    ops::ContextBucket bucket = ops::ContextBucket::K2;
    std::optional<std::uint32_t> slot;
    bool head                  = true;
    bool readback              = false;
    std::int32_t active_tokens = 0;
    ProposalReplay proposal;
};

enum class TargetProgress { Complete, WaitingLayer, AwaitingHook };

class ResidentBody {
public:
    void prepare(DeviceContext&, ParallelExecution*, GraphPreparation&,
                 const ResidentTargetCapture&, const ProposalCapture*, bool gate_steps);
    void start(CudaExecutionView, const TargetReplay&, std::uint32_t active_steps);

private:
    std::optional<ProposalBody> proposal_;
    std::vector<ContextBucketSite> main_sites_;
    std::optional<SlotPatchSite> handoff_;
    DecodeGraphSection head_, readback_;
    GraphSegment graph_;
    ParallelExecution* parallel_ = nullptr;
};

// Keep borrowed operands alive through Device retirement. Cancel an active traversal
// before destroying its graph or schedule.
class OffloadedLayeredBody {
public:
    void prepare(CudaExecutionView, ParallelExecution*, GraphPreparation&, moe::Schedule&,
                 const OffloadedTargetCapture&);
    void begin_prefix(CudaExecutionView, const TargetReplay&);
    [[nodiscard]] TargetProgress start(CudaExecutionView, const TargetReplay&);
    // Complete means the requested layer boundary was reached. poll() submits the suffix.
    [[nodiscard]] TargetProgress advance_until(std::size_t end);
    void launch_hook(std::uint32_t slot);
    [[nodiscard]] TargetProgress poll();
    void cancel();

    [[nodiscard]] std::optional<std::size_t> hook_layer() const noexcept { return hook_layer_; }

private:
    struct OffloadedLayer {
        std::shared_ptr<const moe::LayerGraphDefinition> definition;
        std::unique_ptr<moe::LayerGraphExecutable> executable;
    };

    struct Layer {
        std::vector<ContextBucketSite> sites;
        std::variant<GraphSegment, OffloadedLayer> body;
    };

    GraphSegment prefix_, suffix_;
    DecodeGraphSection head_;
    std::vector<Layer> layers_;
    std::optional<std::size_t> hook_layer_;
    std::optional<SlotPatchSite> hook_slot_;
    GraphSegment hook_;
    ParallelExecution* parallel_ = nullptr;
    moe::Schedule* schedule_     = nullptr;
    Tensor output_;
    std::int32_t physical_tokens_ = 0;

    std::optional<moe::Schedule::Activity> activity_;
    std::optional<CudaExecutionView> cuda_;
    ops::ContextBucket bucket_{};
    std::size_t next_layer_ = 0;
    bool waiting_layer_ = false, hook_submitted_ = false;
};

class TargetGraphBody {
public:
    void prepare(DeviceContext&, ParallelExecution*, GraphPreparation&, moe::Schedule*,
                 const TargetCapture&, const ProposalCapture* fused, bool gate_steps);
    [[nodiscard]] TargetProgress start(CudaExecutionView, const TargetReplay&,
                                       std::uint32_t active_steps = 0);
    void begin_prefix(CudaExecutionView, const TargetReplay&);
    [[nodiscard]] TargetProgress advance_until(std::size_t end);
    void launch_hook(std::uint32_t slot);
    [[nodiscard]] TargetProgress poll();
    void cancel();

    [[nodiscard]] std::optional<std::size_t> hook_layer() const noexcept;

    std::size_t workspace_bytes = 0;

private:
    std::variant<ResidentBody, OffloadedLayeredBody> body_;
};

} // namespace ninfer::models
