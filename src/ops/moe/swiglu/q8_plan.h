#pragma once

#include "core/layout.h"
#include <algorithm>
#include <utility>

namespace ninfer::ops::detail::swiglu_ffn {
inline int q8_product_columns(int tokens) {
    if (tokens == 1) return 1;
    if (tokens <= 8) return 8;
    if (tokens <= 16) return 16;
    if (tokens <= 32) return 64;
    return tokens;
}

template <class Arena>
auto allocate_q8_product(Arena& arena, int tokens) {
    const int columns = q8_product_columns(tokens);
    return std::pair{columns, arena.alloc_bytes(std::size_t(1024) * columns * sizeof(float))};
}

inline std::size_t q8_workspace_bytes(int tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_q8_product(layout, tokens);
    return layout.peak_bytes(1);
}

inline std::size_t q8_workspace_capacity_bytes(int first, int last) {
    auto result = std::max(q8_workspace_bytes(first), q8_workspace_bytes(last));
    // The padded 17..32 route has a larger product allocation than 33..63.
    if (first <= 32 && last >= 17) result = std::max(result, q8_workspace_bytes(32));
    return result;
}
} // namespace ninfer::ops::detail::swiglu_ffn
