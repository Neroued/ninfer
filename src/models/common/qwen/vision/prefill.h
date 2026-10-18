#pragma once

#include "models/common/qwen/vision/vision.h"

#include <memory>

namespace ninfer::models::qwen {

// Select contiguous encoder columns whose prompt positions overlap an embedding window.
struct VisualEmbeddingSlice {
    std::size_t source_begin = 0;
    std::vector<std::int32_t> destination_columns;

    [[nodiscard]] std::size_t size() const noexcept { return destination_columns.size(); }

    [[nodiscard]] bool empty() const noexcept { return destination_columns.empty(); }
};

[[nodiscard]] VisualEmbeddingSlice
visual_embedding_slice(std::span<const std::int32_t> scatter_indices, std::uint32_t prompt_tokens,
                       std::uint32_t embedding_begin, std::uint32_t columns);

void scatter_visual_embeddings(Tensor& input, const Tensor& visual,
                               const VisualEmbeddingSlice& slice, Tensor& destination_indices,
                               CudaExecutionView execution);

class VisionPrefillExecution;

// One state per Program workspace. Only Vision encoding can replace the handoff;
// Text and decode use the disjoint general workspace and preserve its owner.
class VisionHandoffState {
    friend class VisionPrefillExecution;

    const VisionPrefillExecution* owner_ = nullptr;
    std::uint64_t generation_            = 0;
};

struct VisionChunk {
    std::int32_t length              = 0;
    const VisionItemControl* control = nullptr;
    Tensor embeddings;
    std::size_t encode_workspace_bytes = 0;
};

class VisionPrefillExecution {
public:
    VisionPrefillExecution(DeviceContext& device, const VisionConfig& config,
                           const VisionParameters& parameters, MemorySpan workspace,
                           const VisionWorkspacePlan& workspace_plan,
                           std::shared_ptr<const VisionExecutionInput> input,
                           VisionHandoffState& handoff, std::size_t& handoff_peak_bytes);
    ~VisionPrefillExecution();

    VisionPrefillExecution(const VisionPrefillExecution&)            = delete;
    VisionPrefillExecution& operator=(const VisionPrefillExecution&) = delete;

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    void retire_handoff() noexcept;
    [[nodiscard]] double elapsed_seconds() const;

    [[nodiscard]] std::size_t active_handoff_bytes() const noexcept {
        return owns_handoff() ? active_handoff_bytes_ : 0;
    }

private:
    [[nodiscard]] bool owns_handoff() const noexcept {
        return handoff_.owner_ == this && handoff_.generation_ == active_generation_;
    }

    DeviceContext& device_;
    MemorySpan workspace_;
    const VisionWorkspacePlan& workspace_plan_;
    std::shared_ptr<const VisionExecutionInput> input_;
    VisionHandoffState& handoff_;
    std::size_t& handoff_peak_bytes_;
    VisionContext context_;
    std::optional<std::uint32_t> active_item_;
    std::uint64_t active_generation_  = 0;
    std::size_t active_handoff_bytes_ = 0;
    std::vector<CudaEventTimer> timers_;
};

} // namespace ninfer::models::qwen
