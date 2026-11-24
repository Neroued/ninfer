#pragma once

#include "models/common/vision/resources.h"
#include "models/glm5next/execution/parameters.h"
#include "models/glm5next/vision/control.h"

namespace ninfer::models::glm5next::vision {

[[nodiscard]] models::vision::Capacity plan_resources(const VisionConfig&,
                                                      const execution::VisionProfile&,
                                                      std::uint32_t maximum_features,
                                                      std::size_t memory_budget);
[[nodiscard]] std::size_t encode(const VisionConfig&, const execution::VisionParameters&,
                                 const EncoderBatch&, Tensor& output, MemorySpan workspace,
                                 MemorySpan staging, CudaExecutionView);

} // namespace ninfer::models::glm5next::vision
