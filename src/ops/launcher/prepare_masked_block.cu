#include "ops/launcher/prepare_masked_block.h"

#include "core/device.h"
#include "ops/kernel/prepare_masked_block.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {

PrepareMaskedBlockPlan prepare_masked_block_resolve_plan(std::int32_t block_size) {
    if (block_size < 2 || block_size > 16) {
        throw std::invalid_argument("prepare_masked_block plan: B must be 2..16");
    }
    return {
        .route      = PrepareMaskedBlockRoute::Warp32,
        .block_size = block_size,
    };
}

const char* prepare_masked_block_route_name(PrepareMaskedBlockRoute route) {
    switch (route) {
    case PrepareMaskedBlockRoute::Warp32:
        return "warp32";
    case PrepareMaskedBlockRoute::Block64:
        return "block64";
    case PrepareMaskedBlockRoute::Block128:
        return "block128";
    case PrepareMaskedBlockRoute::Block256:
        return "block256";
    }
    return "unknown";
}

std::int32_t prepare_masked_block_route_threads(PrepareMaskedBlockRoute route) {
    switch (route) {
    case PrepareMaskedBlockRoute::Warp32:
        return 32;
    case PrepareMaskedBlockRoute::Block64:
        return 64;
    case PrepareMaskedBlockRoute::Block128:
        return 128;
    case PrepareMaskedBlockRoute::Block256:
        return 256;
    }
    throw std::invalid_argument("prepare_masked_block: invalid route");
}

void prepare_masked_block_launch(const Tensor& anchor, const Tensor& length, std::int32_t mask_id,
                                 Tensor& ids, Tensor& positions, const PrepareMaskedBlockPlan& plan,
                                 cudaStream_t stream) {
    if (plan.block_size != ids.ne[0]) {
        throw std::invalid_argument("prepare_masked_block: inconsistent plan");
    }
    const std::int32_t threads = prepare_masked_block_route_threads(plan.route);
    prepare_masked_block_kernel<<<1, threads, 0, stream>>>(
        static_cast<const std::int32_t*>(anchor.data),
        static_cast<const std::int32_t*>(length.data), mask_id,
        static_cast<std::int32_t*>(ids.data), static_cast<std::int32_t*>(positions.data),
        plan.block_size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
