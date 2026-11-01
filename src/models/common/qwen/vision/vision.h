#pragma once

#include "core/device.h"
#include "models/common/qwen/vision/control.h"
#include "models/common/vision/resources.h"

namespace ninfer::models::qwen {

class VisionContext {
public:
    VisionContext(CudaExecutionView, const VisionConfig&, const VisionParameters&);

    [[nodiscard]] static vision::Capacity plan_resources(const VisionConfig&, const VisionProfile&,
                                                         std::uint32_t max_features,
                                                         std::size_t memory_budget);
    // Inputs and pinned staging outlive the submitted execution. Output is disjoint from scratch.
    [[nodiscard]] std::size_t encode(const VisionEncoderBatch&, Tensor& output,
                                     MemorySpan workspace, MemorySpan staging) const;

private:
    CudaExecutionView execution_;
    const VisionConfig& config_;
    const VisionParameters& parameters_;
};

} // namespace ninfer::models::qwen
