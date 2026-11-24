#pragma once

#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/softmax_attention/kernels/causal/ragged.h"

namespace ninfer::ops::detail {

template <class Entry, class Allocator>
MemorySpan allocate_causal_segment_map(Allocator& allocator, int segments) {
    if (segments <= kInlineSeqs) return {};
    // Entry counts are I32 and entries have at most three I32 fields.
    return allocator.alloc_bytes(std::size_t(segments) * sizeof(Entry));
}

template <class Allocator>
CausalPartialStorage allocate_causal_short_partials(Allocator& allocator, int heads, int width,
                                                    int segments, int splits) {
    if (splits == 1) return {};
    return allocate_causal_partials(allocator, heads, width, splits, segments);
}

template <class Allocator>
CausalPartialStorage allocate_causal_tiled_partials(Allocator& allocator, int heads, int tokens,
                                                    int splits) {
    if (splits <= 1) return {};
    return allocate_causal_partials(allocator, heads, tokens, splits, 1);
}

} // namespace ninfer::ops::detail
