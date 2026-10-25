#pragma once

#include "models/common/vision/resources.h"
#include "models/glm5next/execution/parameters.h"

namespace ninfer::models::glm5next::vision {

struct EncoderBatch {
    std::vector<std::span<const std::uint16_t>> patches;
    std::vector<std::int32_t> positions, cu_seqlens;
    std::uint32_t features = 0;
};

[[nodiscard]] std::vector<models::vision::Unit> units(std::span<const models::frontend::VisionItem>,
                                                      const VisionConfig&);
[[nodiscard]] EncoderBatch
prepare_batch(std::span<const models::frontend::VisionItem>,
              std::span<const std::shared_ptr<const models::frontend::PreparedMediaPayload>>,
              const models::vision::WindowPlan&, std::uint32_t rank, const VisionConfig&);
[[nodiscard]] models::vision::Capacity plan_resources(const VisionConfig&,
                                                      const execution::VisionProfile&,
                                                      std::uint32_t maximum_features,
                                                      std::size_t memory_budget);
[[nodiscard]] std::size_t encode(const VisionConfig&, const execution::VisionParameters&,
                                 const EncoderBatch&, Tensor& output, MemorySpan workspace,
                                 MemorySpan staging, CudaExecutionView);

} // namespace ninfer::models::glm5next::vision
