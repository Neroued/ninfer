#pragma once

#include "ops/softmax_attention/kernels/causal/geometry.h"

namespace ninfer::ops::detail {

template <int TokenTile, int Warps, int KeyTile, int MinBlocks = 1, bool DynamicArena = true>
struct Fp8KvGroupedMmaSchedule {
    static_assert(TokenTile > 0 && Warps > 0 && Warps <= 16 && MinBlocks > 0);
    static_assert(KeyTile == 32 || KeyTile == 64);
    static constexpr int kTokenTile     = TokenTile;
    static constexpr int kWarps         = Warps;
    static constexpr int kThreads       = Warps * 32;
    static constexpr int kKeyRows       = KeyTile;
    static constexpr int kMinBlocks     = MinBlocks;
    static constexpr bool kDynamicArena = DynamicArena;
    static constexpr int kArenaBytes    = 4 * KeyTile * 256;
};

// Each warp owns 16 query rows through QK, online softmax, and PV.
template <int QueryTile, int KeyTile, int MaxRegisters = 255>
struct Fp8KvTiledMmaSchedule {
    static_assert(QueryTile == 16 || QueryTile == 32 || QueryTile == 64 || QueryTile == 128);
    static_assert(KeyTile == 32 || KeyTile == 64);
    static_assert(MaxRegisters > 0 && MaxRegisters <= 255);
    static constexpr int kQueryRows    = QueryTile;
    static constexpr int kKeyRows      = KeyTile;
    static constexpr int kRowTiles     = QueryTile / 16;
    static constexpr int kWarps        = kRowTiles;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kMaxRegisters = MaxRegisters;
    static constexpr int kQBytes       = QueryTile * 256;
    static constexpr int kQScaleBytes  = QueryTile * 4;
    static constexpr int kKBytes       = KeyTile * 256;
    static constexpr int kVBytes       = KeyTile * 256;
    static constexpr int kVStageBytes  = KeyTile * 256 * 2;
    static constexpr int kScaleBytes   = 2 * KeyTile * (1 + sizeof(unsigned));
    static constexpr int kSharedBytes =
        kQBytes + kQScaleBytes + kKBytes + kVBytes + kVStageBytes + kScaleBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

template <int DChunk>
struct Fp8KvMergeSchedule {
    static_assert(DChunk > 0 && DChunk <= 256);
    static constexpr int kDChunk  = DChunk;
    static constexpr int kThreads = 256;
};

} // namespace ninfer::ops::detail
