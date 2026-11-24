#pragma once

#include "ops/softmax_attention/kernels/causal/geometry.h"

namespace ninfer::ops::detail {

template <int TokenTile, int Warps, int KeyTile, int MinBlocks = 1, bool DynamicArena = true>
struct K8V4KvGroupedMmaSchedule {
    static_assert(TokenTile > 0 && Warps > 0 && Warps <= 16 && MinBlocks > 0);
    static_assert(KeyTile == 32 || KeyTile == 64);
    static constexpr int kTokenTile     = TokenTile;
    static constexpr int kWarps         = Warps;
    static constexpr int kThreads       = Warps * 32;
    static constexpr int kKeyRows       = KeyTile;
    static constexpr int kMinBlocks     = MinBlocks;
    static constexpr bool kDynamicArena = DynamicArena;
    static constexpr int kArenaBytes    = 7 * KeyTile * 256 / 2;
};

// Each warp owns 16 query rows through QK, online softmax, and PV.
template <int QueryTile, int KeyTile, int MaxRegisters = 255>
struct K8V4KvTiledMmaSchedule {
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
    static constexpr int kVBytes       = KeyTile * 128;
    static constexpr int kVStageBytes  = KeyTile * 256 * 2;
    static constexpr int kScaleBytes   = KeyTile * (2 + 16);
    static constexpr int kSharedBytes =
        kQBytes + kQScaleBytes + kKBytes + kVBytes + kVStageBytes + kScaleBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

// Inverse rotation consumes the complete normalized D256 row in one CTA.
struct K8V4KvMergeSchedule {
    static constexpr int kDChunk  = 256;
    static constexpr int kThreads = 256;
};

} // namespace ninfer::ops::detail
