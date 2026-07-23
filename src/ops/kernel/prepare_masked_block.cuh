#pragma once

#include <cstdint>

namespace ninfer::ops {

__global__ void prepare_masked_block_kernel(const std::int32_t* anchor, const std::int32_t* length,
                                            std::int32_t mask_id, std::int32_t* ids,
                                            std::int32_t* positions, std::int32_t block_size) {
    const int i = static_cast<int>(threadIdx.x);
    if (i >= block_size) return;
    ids[i]       = i == 0 ? anchor[0] : mask_id;
    positions[i] = length[0] + i;
}

} // namespace ninfer::ops
