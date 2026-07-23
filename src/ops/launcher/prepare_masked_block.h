#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class PrepareMaskedBlockRoute {
    Warp32,
    Block64,
    Block128,
    Block256,
};

struct PrepareMaskedBlockPlan {
    PrepareMaskedBlockRoute route;
    std::int32_t block_size;
};

[[nodiscard]] PrepareMaskedBlockPlan prepare_masked_block_resolve_plan(std::int32_t block_size);
[[nodiscard]] const char* prepare_masked_block_route_name(PrepareMaskedBlockRoute route);
[[nodiscard]] std::int32_t prepare_masked_block_route_threads(PrepareMaskedBlockRoute route);

void prepare_masked_block_launch(const Tensor& anchor, const Tensor& length, std::int32_t mask_id,
                                 Tensor& ids, Tensor& positions, const PrepareMaskedBlockPlan& plan,
                                 cudaStream_t stream);

} // namespace ninfer::ops::detail
