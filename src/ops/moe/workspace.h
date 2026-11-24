#pragma once

#include "core/layout.h"
#include "ninfer/ops/moe.h"

#include <algorithm>
#include <span>

namespace ninfer::ops::detail {

template <class Function>
std::size_t interval_capacity(int first, int last, std::span<const int> boundaries,
                              int slice_tokens, Function&& capacity) {
    first             = std::min(first, slice_tokens);
    last              = std::min(last, slice_tokens);
    std::size_t bytes = std::max(capacity(first), capacity(last));
    for (const int boundary : boundaries) {
        for (const int tokens : {boundary - 1, boundary}) {
            if (tokens >= first && tokens <= last) { bytes = std::max(bytes, capacity(tokens)); }
        }
    }
    return bytes;
}

template <class Arena>
ExpertRoutes allocate_routes(Arena& arena, int top_k, int tokens) {
    return {arena.alloc(DType::I32, {top_k, tokens}), arena.alloc(DType::FP32, {top_k, tokens})};
}

} // namespace ninfer::ops::detail
