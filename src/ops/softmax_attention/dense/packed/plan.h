#pragma once

#include <cstdint>
#include <array>

namespace ninfer::ops::detail {

// Descriptor granularity depends only on host-visible metadata. Device segment
// boundaries remain the sole source of attention membership.
inline std::array<std::int64_t, 3> packed_region_ends(int segments) {
    return {16LL * segments, 32LL * segments, 4096LL * segments - 1};
}

inline int packed_query_rows(int dimension, int tokens, int segments) {
    if (dimension == 72) return 64;
    const auto ends = packed_region_ends(segments);
    if (tokens <= ends[0]) return 16;
    if (tokens <= ends[1]) return 32;
    return tokens <= ends[2] ? 64 : 128;
}

} // namespace ninfer::ops::detail
