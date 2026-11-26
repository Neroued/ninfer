#pragma once

#include "models/common/vision/resources.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/vision/control.h"

namespace ninfer::models::inkling2::vision {

[[nodiscard]] models::vision::Capacity plan_resources(const VisionConfig&,
                                                      const execution::VisionProfile&,
                                                      std::uint32_t maximum_features,
                                                      std::size_t memory_budget);
[[nodiscard]] std::size_t encode(const VisionConfig&, const execution::VisionParameters&,
                                 const EncoderBatch&, Tensor& output, MemorySpan workspace,
                                 MemorySpan staging, CudaExecutionView);

} // namespace ninfer::models::inkling2::vision
