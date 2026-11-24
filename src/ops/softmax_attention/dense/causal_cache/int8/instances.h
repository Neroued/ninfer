#pragma once

#include "ops/softmax_attention/kernels/causal/int8/schedule.cuh"

namespace ninfer::ops::detail {

template <class G, int Tokens>
struct Int8KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles  = (Tokens * G::GroupSize + 15) / 16;
    static constexpr bool kWideKeys = G::GroupSize == 8 && kRowTiles >= 3;
    using Schedule                  = Int8KvGroupedMmaSchedule<Tokens,
                                              kWideKeys        ? 4 * kRowTiles
                                                               : kRowTiles <= 2 ? 8
                                                                                : 2 * kRowTiles,
                                              kWideKeys ? 64 : 32, kWideKeys ? 1 : 2>;
    using Merge                     = Int8KvMergeSchedule<G::QHeads == 24 ? 256 : 64>;
};

using Int8KvTiledInstance = Int8KvTiledMmaSchedule<64, 64, 128>;

} // namespace ninfer::ops::detail
