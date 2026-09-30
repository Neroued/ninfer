#pragma once

#include "core/paged_kv_cache.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

// A lane must retain its working set while a prefill chunk grows, including the
// rolling placement's sink/slack margin. The pool shares storage, not this guarantee.
inline std::uint32_t kvmem_lane_page_budget(std::uint32_t context_tokens,
                                           std::uint32_t prefill_chunk,
                                           std::uint32_t window_pages) {
    if (context_tokens == 0 || prefill_chunk == 0 || window_pages == 0) {
        throw std::invalid_argument("KVMem capacity requires nonzero context, chunk and window");
    }
    const auto pages = [](std::uint32_t tokens) {
        return (static_cast<std::uint64_t>(tokens) + kPagedKVPageSize - 1U) / kPagedKVPageSize;
    };
    return static_cast<std::uint32_t>(std::min(
        pages(context_tokens), static_cast<std::uint64_t>(window_pages) +
                                   pages(std::min(prefill_chunk, context_tokens)) + 16U));
}

inline std::uint32_t kvmem_pool_page_budget(std::uint32_t context_tokens,
                                           std::uint32_t prefill_chunk,
                                           std::uint32_t window_pages, std::uint32_t lanes) {
    const auto total = static_cast<std::uint64_t>(
                           kvmem_lane_page_budget(context_tokens, prefill_chunk, window_pages)) * lanes;
    if (lanes == 0 || total > std::numeric_limits<std::uint32_t>::max() / kPagedKVPageSize) {
        throw std::invalid_argument("KVMem pool capacity is outside the token capacity range");
    }
    return static_cast<std::uint32_t>(total);
}

} // namespace ninfer::models::qwen3_5::detail
