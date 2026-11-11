#pragma once

#include "ops/softmax_attention/qsa/geometry.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops::detail::qsa {

enum class SparseGqaWarpSplit { Sequence, Output };

enum class SparseGqaLoad { Direct, Staged, InPlace, Aliased, Reused };

struct SparseGqaPage {
    const std::uint8_t* data;
    std::int64_t row, head; // Byte strides within one record.
};

struct SparseGqaWork {
    int count, complete, size, tiles, splits;
};

struct SparseGqaArguments {
    const __nv_bfloat16 *query, *gate;
    __nv_bfloat16* out;
    SparseGqaPage key, value, key_scale, value_scale;
    const int *positions, *valid, *table_rows;
    const std::int64_t* tables;
    const int *ids, *counts;
    std::int64_t table_stride;
    float *numerator, *maximum, *sum;
    int tokens, batch, splits, key_rows;
    std::int64_t begin;
    int min_split_tiles = 1;
    bool direct_single  = false;

    __device__ SparseGqaWork work(std::int64_t query_id) const {
        const int b = int(query_id / tokens), t = int(query_id % tokens);
        if (t >= valid[b]) return {};
        const int p = positions[query_id], c = counts[query_id];
        const int n     = c * 4 + (p + 1) % 4;
        const int tiles = (n + key_rows - 1) / key_rows;
        int partitions  = tiles < splits ? tiles : splits;
        if (min_split_tiles > 1 && partitions > 1) {
            const int capacity = tiles / min_split_tiles;
            const int limit    = capacity > 0 ? capacity : 1;
            partitions         = limit < partitions ? limit : partitions;
        }
        return {c, (p + 1) / 4, n, tiles, partitions};
    }
};

// Head geometry stays constant through the contraction and output address calculations.
template <int KvHeads>
struct SparseGqaKernelArguments : SparseGqaArguments {
    static constexpr int kv_heads = KvHeads, query_heads = HeadGeometry<KvHeads>::query_heads,
                         ratio = HeadGeometry<KvHeads>::ratio;

    __device__ static unsigned group_query(unsigned x) { return x / kv_heads; }

    __device__ static unsigned group_head(unsigned x) { return x % kv_heads; }

    __device__ static unsigned head_query(unsigned x) { return x / query_heads; }

    __device__ static unsigned head_index(unsigned x) { return x % query_heads; }

    __device__ std::int64_t partial_row(std::int64_t q, int split, int h) const {
        return ((std::int64_t(split) * batch * tokens + q) * query_heads + h);
    }
};

template <int Queries>
__device__ __forceinline__ bool sparse_gqa_all_selected(const SparseGqaArguments& a,
                                                        std::int64_t first, int count = Queries) {
    const auto total = std::int64_t(a.tokens) * a.batch;
    const auto last  = first + count < total ? first + count : total;
    if (first / a.tokens != (last - 1) / a.tokens) return false;
    for (auto q = first; q < last; ++q) {
        const auto work = a.work(q);
        if (work.count != work.complete) return false;
    }
    return true;
}

} // namespace ninfer::ops::detail::qsa
