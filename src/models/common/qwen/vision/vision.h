#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/common/qwen/vision/control.h"
#include "models/common/qwen/vision/parameters.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen {

struct VisionWorkspacePlan {
    std::int32_t output_hidden         = 0;
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

struct VisionItemView {
    std::span<const std::uint16_t> patches;
    const VisionItemControl* control = nullptr;
};

class VisionContext {
public:
    VisionContext(CudaExecutionView execution, const VisionConfig& config,
                  const VisionParameters& parameters);

    [[nodiscard]] static std::size_t workspace_bytes(const VisionConfig& config,
                                                     const VisionParameters& parameters,
                                                     std::size_t patches, std::size_t merged_tokens,
                                                     const VisionWorkspacePlan& plan);
    [[nodiscard]] static VisionWorkspacePlan plan_workspace(const VisionConfig& config,
                                                            const VisionParameters& parameters,
                                                            std::uint32_t max_merged_tokens,
                                                            std::size_t general_capacity_bytes);

    [[nodiscard]] static VisionWorkspacePlan plan_workspace(const VisionConfig& config,
                                                            const VisionProfile& profile,
                                                            std::uint32_t max_tokens,
                                                            std::size_t general_bytes);

    [[nodiscard]] const VisionConfig& config() const noexcept { return config_; }

    [[nodiscard]] static Tensor bind_output(MemorySpan backing, const VisionWorkspacePlan& plan,
                                            std::size_t merged_tokens);
    // Output names bind_output() or disjoint Device BF16 storage. Enqueue encoding and
    // return its workspace extent; an external output is owned and budgeted by the caller.
    [[nodiscard]] std::size_t encode(const VisionItemView& item, Tensor& output, MemorySpan backing,
                                     const VisionWorkspacePlan& plan) const;

private:
    CudaExecutionView execution_;
    const VisionConfig& config_;
    const VisionParameters& parameters_;
};

} // namespace ninfer::models::qwen
