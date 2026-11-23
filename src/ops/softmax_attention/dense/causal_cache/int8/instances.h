#pragma once

#include "ops/softmax_attention/kernels/causal/int8/schedule.cuh"

namespace ninfer::ops::detail {

template <class G, int Tokens>
struct Int8KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles  = (Tokens * G::GroupSize + 15) / 16;
    static constexpr bool kWideKeys = G::GroupSize == 8 && kRowTiles >= 3;
    // A 512-thread block caps registers at 128 per thread. Four wide query tiles
    // use two consumers each to keep the score and value accumulators in registers.
    static constexpr int kWarps     = kWideKeys && kRowTiles == 3 ? 4 * kRowTiles
                                      : kRowTiles <= 2            ? 8
                                                                  : 2 * kRowTiles;
    static constexpr int kMinBlocks = kRowTiles >= 3 ? 1 : 2;
    using Schedule = Int8KvGroupedMmaSchedule<Tokens, kWarps, kWideKeys ? 64 : 32, kMinBlocks>;
    using Merge    = Int8KvMergeSchedule<G::QHeads == 24 ? 256 : 64>;
};

using Int8KvTiledInstance = Int8KvTiledMmaSchedule<64, 64, 128>;

} // namespace ninfer::ops::detail
