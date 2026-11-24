#pragma once

#include "ops/gated_residual/mix/operands.h"
#include "ops/common/host/operands.h"
#include "core/layout.h"

namespace ninfer::ops::detail::hc::mix {
struct Scratch {
    MemorySpan storage;
    void* activation;
    MemorySpan partials;
};

// One contiguous reservation gives execution and sizing identical offsets, including alignment.
struct Layout {
    LayoutRegion activation, partials;
    std::size_t bytes;
};

inline Layout layout(Kind kind, std::int32_t columns, int splits) {
    LayoutBuilder b;
    const auto u = b.add(common::operand::bytes(std::uint64_t(kRank) * columns, 2, "hc_mix"), 256,
                         "HC low-rank activation");
    const auto temporary_bytes =
        splits == 1 ? 0 : std::uint64_t(down_rows(kind)) * columns * splits * 4;
    LayoutRegion temporary;
    if (temporary_bytes)
        temporary =
            b.add(common::operand::bytes(temporary_bytes, 1, "hc_mix"), 256, "HC Down partials");
    return {u, temporary, b.finish(1)};
}

inline Scratch allocate_workspace(MemoryArena& arena, Layout l) {
    const auto storage = arena.alloc_bytes(l.bytes, 256);
    return {storage, l.activation.bind(storage).data,
            l.partials.bytes ? l.partials.bind(storage) : MemorySpan{}};
}
} // namespace ninfer::ops::detail::hc::mix
