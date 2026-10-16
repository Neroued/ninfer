#pragma once

#include "core/decode_graph.h"

#include "models/qwen4_exp/execution/text.h"
#include "models/common/moe/schedule.h"

namespace ninfer::models::qwen4_exp {

struct GraphSegment {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;

    void prepare(CudaExecutionView, const std::function<void()>&);
};

// One exact T/B traversal. Context profiles share all GDN and routed-expert definitions.
// The activation frame and PLE slots belong to Resources and outlive every graph.
class TextGraph {
public:
    void prepare(execution::TextRun&, const execution::Parameters&, moe::Schedule*,
                 std::span<const std::uint32_t> bounds, std::array<Tensor, 2> ple,
                 execution::TextOutputs, std::int32_t columns, MemoryArena&, DeviceContext&);
    void select(std::uint32_t visible_tokens);
    void prefix(CudaExecutionView);
    void consume_ple(std::uint32_t slot, CudaExecutionView);
    void suffix(CudaExecutionView);

    [[nodiscard]] std::size_t workspace_bytes() const {
        return profiles_[installed_].workspace_bytes;
    }

private:
    struct Profile {
        std::uint32_t bound;
        std::vector<std::shared_ptr<const moe::LayerGraphDefinition>> layers;
        std::size_t workspace_bytes = 0;
    };

    void traverse(std::uint32_t begin, std::uint32_t end, CudaExecutionView);
    GraphSegment stem_, ple_[2], finish_;
    std::vector<Profile> profiles_;
    std::vector<std::unique_ptr<moe::LayerGraphExecutable>> layers_;
    Tensor block_output_;
    std::vector<ExpertBankId> banks_;
    std::uint32_t ple_layer_ = 0;
    std::size_t installed_   = 0;
};

} // namespace ninfer::models::qwen4_exp
