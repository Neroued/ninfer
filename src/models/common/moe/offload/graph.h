#pragma once

#include "core/decode_graph.h"
#include "models/common/moe/offload/resources.h"

#include <array>
#include <functional>
#include <optional>

namespace ninfer::models::moe::offload {

[[nodiscard]] GraphPlan graph_plan(const Plan&, ExpertBankId, TokenGeometry);

struct LayerGraphDefinition {
    GraphPlan plan;
    ops::MoeTokenValidity validity;
    DecodeGraphDefinition mixer, steady, feature;
    std::vector<DecodeGraphKernelSlice> mixer_slices, steady_slices;
    std::array<DecodeGraphDefinition, 3> subsets;
};

void capture_layer(LayerGraphDefinition&, Resources&, ExpertBankId, CudaExecutionView,
                   TokenGeometry, const MixerBody& mixer,
                   const std::function<void(const PreparedExecution&)>& gpu_moe,
                   const std::function<void()>& feature = {});

// Owns the GPU segments of one layer. Host cache/CPU handoffs stay outside capture.
class LayerGraphExecutable {
public:
    explicit LayerGraphExecutable(const LayerGraphDefinition&);
    void update_mixer_slice(std::size_t, const DecodeGraphKernelPatch&);
    void upload_mixer(cudaStream_t);
    void upload(cudaStream_t);
    [[nodiscard]] bool start(Resources&, ExpertBankId, Tensor& output, CudaExecutionView,
                             std::function<void()> after_gpu_submit = {});
    [[nodiscard]] bool poll();
    void cancel();

private:
    struct Invocation {
        enum class Stage { BeforeLayer, Prepare, Complete };
        Resources& resources;
        ExpertBankId bank;
        Tensor output;
        CudaExecutionView cuda;
        std::function<void()> after_gpu_submit;
        Stage stage                     = Stage::BeforeLayer;
        bool combined                   = false;
        std::size_t calibration_variant = 0;
        std::optional<PreparedExecution> call;
    };

    [[nodiscard]] bool advance();
    std::optional<Invocation> pending_;
    std::span<const DecodeGraphKernelSlice> mixer_slices_, steady_slices_;
    GraphPlan plan_;
    ops::MoeTokenValidity validity_;
    DecodeGraphExecutable mixer_, steady_, feature_;
    std::array<DecodeGraphExecutable, 3> subsets_;
};

} // namespace ninfer::models::moe::offload
