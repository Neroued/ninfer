#pragma once
#include "ops/softmax_attention/kernels/causal/nvfp4/schedule.cuh"

namespace ninfer::ops::detail {
template <class G, int Tokens>
struct Nvfp4KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles = (Tokens * G::GroupSize + 15) / 16;
    using Schedule =
        Nvfp4KvGroupedMmaSchedule<Tokens, kRowTiles <= 2 ? 8 : 4 * kRowTiles,
                                  kRowTiles <= 2 ? 32 : 64, kRowTiles <= 2 ? 2 : 1, true>;
    using Merge = Nvfp4KvMergeSchedule;
};

using Nvfp4KvTiledInstance = Nvfp4KvTiledMmaSchedule<64, 12>;
} // namespace ninfer::ops::detail
