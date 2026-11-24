#pragma once

#include "ops/softmax_attention/kernels/causal/k8v4/schedule.cuh"

namespace ninfer::ops::detail {

template <class G, int Tokens>
struct K8V4KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles = (Tokens * G::GroupSize + 15) / 16;
    using Schedule                 = K8V4KvGroupedMmaSchedule<Tokens,
                                              kRowTiles == 4   ? 16
                                                              : kRowTiles == 3 ? 12
                                                                               : 8,
                                              Tokens == 1 ? 32 : 64, Tokens == 1 ? 2 : 1>;
    using Merge                    = K8V4KvMergeSchedule;
};

using K8V4KvTiledInstance = K8V4KvTiledMmaSchedule<128, 64>;

} // namespace ninfer::ops::detail
