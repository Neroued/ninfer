#pragma once

#include "core/memory.h"
#include <cstddef>
#include <cstdint>
#include <numeric>

namespace ninfer::ops::detail {

// The shape schedule supplies the wave and tile geometry. Only the final
// partial wave needs K128-aligned transient FP32 accumulators.
template <int Rows, int Tokens, int Ctas, int Parts>
struct Fp8BlockA16TailPlan {
    static constexpr int kRows = Rows, kTokens = Tokens, kCtas = Ctas, kParts = Parts;
    static_assert(Rows > 0 && Tokens > 0 && Ctas > 0 && Parts > 0 && Ctas % Parts == 0);
    static constexpr int kTailTiles            = kCtas / kParts;
    static constexpr int kTileElements         = kRows * kTokens;
    static constexpr std::size_t kAlignment    = 256;
    static constexpr std::size_t kPartialBytes = std::size_t(kCtas) * kTileElements * sizeof(float);
    // Accept an arbitrary caller arena cursor without hidden alignment headroom.
    static constexpr std::size_t kWorkspaceBytes = kPartialBytes + kAlignment - 1;

    std::int64_t tiles;

    bool split_tail() const { return tiles % kCtas == kTailTiles; }

    std::int64_t full_tiles() const { return tiles - kTailTiles; }
};

template <class P>
inline P fp8_block_a16_tail_plan(int rows, int tokens) {
    return {((std::int64_t(rows) + P::kRows - 1) / P::kRows) *
            ((std::int64_t(tokens) + P::kTokens - 1) / P::kTokens)};
}

template <class P>
inline std::size_t fp8_block_a16_tail_capacity(int rows, int first, int last) {
    if (first > last) return 0;
    const int row_tiles = (rows + P::kRows - 1) / P::kRows;
    const int period    = P::kCtas / std::gcd(row_tiles, P::kCtas);
    const auto bucket   = (std::int64_t(first) - 1) / P::kTokens;
    for (int step = 0; step < period; ++step) {
        const auto candidate = step ? (bucket + step) * P::kTokens + 1 : first;
        if (candidate > last) break;
        if (fp8_block_a16_tail_plan<P>(rows, int(candidate)).split_tail())
            return P::kWorkspaceBytes;
    }
    return 0;
}

template <class P>
inline float* bind_fp8_block_a16_tail_workspace(MemorySpan backing) {
    const auto offset = (-reinterpret_cast<std::uintptr_t>(backing.data)) & (P::kAlignment - 1);
    return static_cast<float*>(backing.subspan(offset, P::kPartialBytes).data);
}

} // namespace ninfer::ops::detail
