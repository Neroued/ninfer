#pragma once

#include "ops/softmax_attention/kernels/causal/geometry.h"
#include <cstdint>

namespace ninfer::ops::detail {

// Query rows pack token/head pairs. KV warps cooperate on disjoint key columns.
template <int QueryRows, int KeyRows = 32, int KVWarps = 1, int MinBlocks = 2, int FixedWidth = 0>
struct Bf16KvGroupedMmaSchedule {
    static_assert(QueryRows == 16 || QueryRows == 32 || QueryRows == 64);
    static_assert(KeyRows == 32 || KeyRows == 64);
    static_assert(KVWarps == 1 || KVWarps == 2 || KVWarps == 4);
    static_assert(KeyRows % (16 * KVWarps) == 0);
    static_assert(QueryRows <= 2 * KeyRows);
    static_assert(FixedWidth == 0 || FixedWidth == 1);
    static constexpr int kFixedWidth = FixedWidth;
    static constexpr int kQueryRows  = QueryRows;
    static constexpr int kKeyRows    = KeyRows;
    static constexpr int kWarpsQ     = QueryRows / 16;
    static constexpr int kWarpsKV    = KVWarps;
    static constexpr int kWarps      = kWarpsQ * kWarpsKV;
    static_assert(kWarps <= 8);
    static constexpr int kThreads            = 32 * kWarps;
    static constexpr int kLaunchBoundThreads = kThreads < 128 ? 128 : kThreads;
    static constexpr int kMinBlocks          = MinBlocks;
};

// Existing single-buffer Q/K/V pipeline. Page-local staging admits key tiles
// dividing the 64-row physical page, not arbitrary multiples of the MMA shape.
template <int QueryTile = 64, int KeyTile = 64, int MinBlocks = 1>
struct Bf16KvTiledMmaSchedule {
    static_assert(QueryTile == 16 || QueryTile == 32 || QueryTile == 64 || QueryTile == 128);
    static_assert(KeyTile == 16 || KeyTile == 32 || KeyTile == 64);
    static_assert(MinBlocks > 0);
    static constexpr int kQueryRows = QueryTile;
    static constexpr int kKeyRows   = KeyTile;
    static constexpr int kWarps     = QueryTile / 16;
    static constexpr int kThreads   = kWarps * 32;
    static constexpr int kMinBlocks = MinBlocks;
};

template <int DChunk, int Threads = 256>
struct Bf16KvMergeSchedule {
    static_assert(Threads == 128 || Threads == 256);
    static_assert(DChunk > 0 && DChunk <= Threads);
    static constexpr int kDChunk  = DChunk;
    static constexpr int kThreads = Threads;
    static constexpr int kWarps   = Threads / 32;
};

template <class G, class S>
struct alignas(16) Bf16KvGroupedStorage {
    static constexpr int kStateRows = S::kWarpsKV > 1 ? S::kWarpsKV* S::kQueryRows : 1;

    union {
        std::uint16_t qkv[2 * S::kKeyRows * G::kHeadDim];
        float reduction[S::kWarpsKV > 1 ? kStateRows* G::kHeadDim : 1];
    };

    // A rolling page window, not a bound on the KV partition length.
    static constexpr int kPageWindow = 32;
    std::int64_t record_offsets[kPageWindow];
    float maximum[kStateRows];
    float sum[kStateRows];
};

template <class Geometry, class Schedule>
inline constexpr int bf16_kv_tiled_shared_bytes =
    (Schedule::kQueryRows + 2 * Schedule::kKeyRows) * Geometry::kHeadDim * 2;

} // namespace ninfer::ops::detail
