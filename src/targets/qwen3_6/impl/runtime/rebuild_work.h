#pragma once

#include "runtime/contract/types.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::runtime_support {

inline void include_rebuild_boundary(std::uint32_t& tail_begin, std::uint32_t boundary,
                                     std::uint32_t frontier) noexcept {
    if (boundary <= frontier) { tail_begin = std::max(tail_begin, boundary); }
}

inline void advance_segmented_rebuild_work(runtime::PrefillWork& work, std::uint32_t tail_begin,
                                           std::uint32_t previous_frontier, std::uint32_t frontier,
                                           std::uint32_t prefill_chunk) {
    if (frontier < previous_frontier || work.tokens != previous_frontier ||
        tail_begin > previous_frontier || prefill_chunk == 0) {
        throw std::logic_error("sequence rebuild work is not aligned with its frontier");
    }
    if (frontier == previous_frontier) { return; }

    const std::uint64_t old_tail = previous_frontier - tail_begin;
    const std::uint64_t new_tail = frontier - tail_begin;
    const auto chunks_for        = [prefill_chunk](std::uint64_t tokens) {
        return tokens == 0 ? 0ULL : 1ULL + (tokens - 1ULL) / prefill_chunk;
    };
    const std::uint64_t old_tail_chunks = chunks_for(old_tail);
    const std::uint64_t new_tail_chunks = chunks_for(new_tail);
    if (work.chunks < old_tail_chunks) {
        throw std::logic_error("sequence rebuild chunk accounting is invalid");
    }

    const runtime::PrefillWork total = runtime::make_prefill_work(
        0, frontier, work.vision_items, work.vision_patches, prefill_chunk);
    work.chunks          = work.chunks - old_tail_chunks + new_tail_chunks;
    work.tokens          = total.tokens;
    work.attention_pairs = total.attention_pairs;
}

} // namespace ninfer::targets::qwen3_6::runtime_support
