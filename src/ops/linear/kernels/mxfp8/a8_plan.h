#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "ops/linear/kernels/mxfp8/operands.h"

#include <stdexcept>

namespace ninfer::ops::detail {

struct Mxfp8A8Workspace {
    std::uint8_t* codes;
    std::uint8_t* scales;
};

template <class Arena>
Mxfp8A8Workspace allocate_mxfp8_a8_workspace(Arena& arena, int tokens, int k) {
    if (tokens <= 0 || k <= 0 || k % 32)
        throw std::invalid_argument("MXFP8 workspace requires positive T and K32");
    const auto bytes = static_cast<std::size_t>(tokens) * k;
    return {static_cast<std::uint8_t*>(arena.alloc_bytes(bytes, 256).data),
            static_cast<std::uint8_t*>(arena.alloc_bytes(bytes / 32, 256).data)};
}

inline std::size_t mxfp8_a8_workspace_capacity_bytes(int tokens, int k) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_mxfp8_a8_workspace(layout, tokens, k);
    return layout.peak_bytes(1);
}

inline Mxfp8A8Operands mxfp8_a8_operands(const Weight& weight, Mxfp8A8Workspace x, int tokens) {
    return {x.codes,
            x.scales,
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            weight.n,
            weight.k,
            tokens};
}

} // namespace ninfer::ops::detail
