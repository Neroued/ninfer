#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "ops/linear/kernels/mxfp8/operands.h"

#include <stdexcept>
#include <limits>

namespace ninfer::ops::detail {

struct Mxfp8A8Workspace {
    std::uint8_t* codes;
    std::uint8_t* scales;
};

inline int mxfp8_a8_padded_tokens(int tokens, int alignment) {
    const auto padded = (static_cast<std::int64_t>(tokens) + alignment - 1) / alignment * alignment;
    if (padded > std::numeric_limits<int>::max())
        throw std::invalid_argument("MXFP8 padded token count exceeds int range");
    return static_cast<int>(padded);
}

template <class Arena>
Mxfp8A8Workspace allocate_mxfp8_a8_workspace(Arena& arena, int tokens, int k,
                                             int code_tokens = 0) {
    if (tokens <= 0 || k <= 0 || k % 32)
        throw std::invalid_argument("MXFP8 workspace requires positive T and K32");
    if (!code_tokens) code_tokens = tokens;
    if (code_tokens < tokens)
        throw std::invalid_argument("MXFP8 code storage is smaller than the activation");
    const auto bytes = static_cast<std::size_t>(tokens) * k;
    return {static_cast<std::uint8_t*>(arena.alloc_bytes(static_cast<std::size_t>(code_tokens) * k,
                                                       256).data),
            static_cast<std::uint8_t*>(arena.alloc_bytes(bytes / 32, 256).data)};
}

inline std::size_t mxfp8_a8_workspace_capacity_bytes(int tokens, int k, int code_tokens = 0) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_mxfp8_a8_workspace(layout, tokens, k, code_tokens);
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
