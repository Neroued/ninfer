#pragma once

#include "ops/softmax_attention/qsa/sparse_gqa/grouped.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"

namespace ninfer::ops::detail::qsa {

template <KvCacheStorage Storage>
struct SparseGqaGroupedInstance {
    static constexpr int kQueries = 8;
    static constexpr int kKeyRows = 32;
    static constexpr int kParts   = kSparseQuantized<Storage> ? 2 : 1;
    static constexpr int kWarps   = kQueries * 12 / 16 * kParts;

    template <bool Vector, SparseGqaLoad Load>
    static constexpr int shared_bytes = std::max(
        (sizeof(SparseGqaGroupedTile<Storage, kKeyRows, kWarps, Vector, Load>) + 15) / 16 * 16 +
            kQueries * 12 * 256 * sizeof(std::uint16_t),
        std::size_t(kQueries * 12 * 256 * sizeof(float)));
};

void launch_sparse_gqa_fp8_grouped(const SparseGqaArguments& arguments, int kv_heads, int splits,
                                   int queries, CudaExecutionView execution);

} // namespace ninfer::ops::detail::qsa
