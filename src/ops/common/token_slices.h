#pragma once

// Private host utility for completing one logical Op call through as many CUDA launches as its
// selected kernel needs. This is not prefill chunking: the callback runs once unless the selected
// kernel's grid.y would exceed CUDA's launch limit. CUDA grid limits never narrow an Op's T domain.

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kCudaGridYLimit = 65'535;

template <class Launch>
void for_each_token_slice(std::int32_t tokens, std::int32_t columns_per_block, Launch&& launch) {
    if (tokens <= 0 || columns_per_block <= 0) {
        throw std::invalid_argument("token slicing requires positive dimensions");
    }
    const std::int64_t capacity = static_cast<std::int64_t>(columns_per_block) * kCudaGridYLimit;
    std::int32_t offset         = 0;
    while (offset < tokens) {
        const std::int32_t count = static_cast<std::int32_t>(
            std::min<std::int64_t>(capacity, static_cast<std::int64_t>(tokens) - offset));
        launch(offset, count);
        offset += count;
    }
}

} // namespace ninfer::ops::detail
