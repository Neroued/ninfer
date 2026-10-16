#pragma once

#include "core/decode_graph.h"
#include "models/common/moe/offload/resources.h"

#include <array>
#include <functional>
#include <memory>
#include <optional>

namespace ninfer::models::moe::offload {

[[nodiscard]] GraphPlan graph_plan(const Plan&, ExpertBankId, std::int32_t tokens);

// Subset bodies share exact-T bindings across attention context profiles.
struct SubsetGraphDefinitions {
    std::array<DecodeGraphDefinition, 3> subsets;
};

struct LayerGraphDefinition {
    GraphPlan plan;
    DecodeGraphDefinition mixer, steady, feature;
    std::shared_ptr<SubsetGraphDefinitions> moe;
};

void capture_layer(LayerGraphDefinition&, Resources&, ExpertBankId, CudaExecutionView,
                   std::int32_t tokens, const std::function<void()>& mixer,
                   const std::function<void(const PreparedExecution&)>& gpu_moe,
                   const std::function<void()>& feature = {});

// Owns the GPU segments of one layer. Host cache/CPU handoffs stay outside capture.
class LayerGraphExecutable {
public:
    explicit LayerGraphExecutable(const LayerGraphDefinition&);
    void update(const LayerGraphDefinition&);
    void upload(cudaStream_t);
    void replay(Resources&, ExpertBankId, Tensor& output, CudaExecutionView,
                const std::function<void()>& after_gpu_submit = {});

private:
    GraphPlan plan_;
    DecodeGraphExecutable mixer_, steady_, feature_;
    std::array<DecodeGraphExecutable, 3> subsets_;
};

} // namespace ninfer::models::moe::offload
