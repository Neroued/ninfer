#pragma once

#include "core/decode_graph.h"
#include "models/qwen4_exp/execution/text.h"
#include "models/qwen4_exp/ple/upload.h"

namespace ninfer::models::qwen4_exp {

struct GraphSegment {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;

    void prepare(CudaExecutionView, const std::function<void()>&);
};

// One exact T/B traversal. Resident execution submits one graph, with replaceable QSA
// mixers and PLE input. Expert offload retains its Host handoffs between layer segments.
class TextGraph {
public:
    void prepare(execution::TextRun&, const execution::Parameters&, moe::Schedule*,
                 std::span<const std::uint32_t> bounds, std::array<ple::DeviceEmbedding, 2>,
                 execution::TextOutputs, Tensor logits, std::int32_t columns, MemoryArena&,
                 DeviceContext&);
    void select(std::uint32_t visible_tokens);
    [[nodiscard]] bool resident() const;
    void replay(std::uint32_t ple_slot, CudaExecutionView);
    void prefix(CudaExecutionView);
    void consume_ple(std::uint32_t slot, CudaExecutionView);
    void suffix(CudaExecutionView);

    [[nodiscard]] std::size_t workspace_bytes() const { return workspace_bytes_[installed_]; }

private:
    struct Resident {
        GraphSegment main;
        std::array<DecodeGraphDefinition, 2> ple;
        DecodeGraphChild ple_node;
        std::vector<DecodeGraphChild> mixers;
        std::vector<std::vector<DecodeGraphDefinition>> profiles;
        std::uint32_t slot = 0;
    };

    struct Offloaded {
        GraphSegment stem, ple[2], finish;
        std::vector<std::vector<std::shared_ptr<const moe::LayerGraphDefinition>>> profiles;
        std::vector<std::unique_ptr<moe::LayerGraphExecutable>> layers;
        Tensor block_output;
        std::uint32_t ple_layer = 0;

        void traverse(std::uint32_t begin, std::uint32_t end, CudaExecutionView);
    };

    void prepare_resident(execution::TextRun&, const execution::Parameters&,
                          std::array<ple::DeviceEmbedding, 2>, execution::TextOutputs,
                          Tensor logits, MemoryArena&, CudaExecutionView);
    void prepare_offloaded(execution::TextRun&, const execution::Parameters&, moe::Schedule&,
                           std::array<ple::DeviceEmbedding, 2>, execution::TextOutputs,
                           Tensor logits, std::int32_t columns, MemoryArena&, CudaExecutionView);
    std::variant<std::monostate, Resident, Offloaded> implementation_;
    std::vector<std::uint32_t> bounds_;
    std::vector<std::size_t> workspace_bytes_;
    std::size_t installed_ = 0;
};

} // namespace ninfer::models::qwen4_exp
