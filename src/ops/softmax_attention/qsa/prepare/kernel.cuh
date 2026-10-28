#pragma once

#include "ops/softmax_attention/qsa/prepare/norm_rope.cuh"
#include "ops/softmax_attention/qsa/geometry.h"
#include "ops/softmax_attention/qsa/prepare/arguments.cuh"
#include "ops/softmax_attention/qsa/prepare/cache.cuh"

namespace ninfer::ops::detail::qsa {

__device__ __forceinline__ int coordinate(const PrepareArguments& a, int t, int b, int axis) {
    return a.rope[t + std::int64_t(a.tokens) * ((a.axes == 1 ? 0 : axis) + a.axes * b)];
}

template <bool BoundedPhase>
__device__ __forceinline__ void prepare_raw(const PrepareArguments& a, int t, int b, int lane,
                                            std::int64_t m) {
    const int record = a.offsets[b] + t;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        *a.raw_records.at(lane + 32 * j, record, b) = a.ik[m * 128 + lane + 32 * j];
    if (lane < 3) *a.rope_records.at(lane, record, b) = coordinate(a, t, b, lane);
    const int p = a.positions[m];
    if (p % 4 != 3) return;
    const int base = a.positions[std::int64_t(b) * a.tokens], first = p - 3;
    float pooled[4]{};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int token = first + j;
        const auto* raw = token < base ? a.tail.at(0, token - (base - base % 4), b)
                                       : a.ik + (std::int64_t(b) * a.tokens + token - base) * 128;
#pragma unroll
        for (int c = 0; c < 4; ++c) pooled[c] += 0.25F * __bfloat162float(raw[lane + 32 * c]);
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) pooled[c] = __bfloat162float(__float2bfloat16_rn(pooled[c]));
    const int axis = lane % 3;
    const int rho =
        first < base ? *a.first_rope.at(axis, 0, b) : coordinate(a, first - base, b, axis);
    const float2 sc = prepare_sincos<BoundedPhase>(a.frequency[lane], rho);
    norm_rotate<128>(pooled, a.wik, a.epsilon, sc, lane);
    const auto page = a.tables[p / 64 + std::int64_t(a.table_rows[b]) * a.table_stride];
    auto* out       = a.index_keys.at((p / 4) % 16, 0, page);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[lane + 32 * j] = __float2bfloat16_rn(pooled[j]);
}

template <int KvHeads, bool Queries>
struct PrepareGeometry {
    static constexpr int warps       = 4;
    static constexpr int query_heads = Queries ? HeadGeometry<KvHeads>::query_heads : 0;
    static constexpr int key_task    = Queries ? query_heads + 4 : 0;
    static constexpr int raw_task    = key_task + KvHeads;
    static constexpr int groups      = (raw_task + 1 + warps - 1) / warps;
};

// Each warp owns one head/role; CTAs share the token's rotary coefficients.
// Compression reads immutable projections/tail, never records.
// A carried block retains its first token's three axes even when current coordinates use A=1.
template <KvCacheStorage Storage, bool BoundedPhase, bool VectorWeights, bool TokenMajor,
          int KvHeads, bool Queries>
__global__ __launch_bounds__(128) void prepare_kernel(const __grid_constant__ PrepareArguments a) {
    using Geometry = PrepareGeometry<KvHeads, Queries>;
    const int t    = TokenMajor ? blockIdx.x / Geometry::groups : blockIdx.x;
    const int b = blockIdx.y, lane = threadIdx.x % 32;
    const int group = TokenMajor ? blockIdx.x % Geometry::groups : blockIdx.z;
    const int task  = group * Geometry::warps + threadIdx.x / 32;
    const auto m    = t + std::int64_t(b) * a.tokens;
    const bool live = t < a.valid[b];
    __shared__ float2 coefficients[32];
    if (live && threadIdx.x < 32)
        coefficients[lane] =
            prepare_sincos<BoundedPhase>(a.frequency[lane], coordinate(a, t, b, lane % 3));
    __syncthreads();
    if constexpr (Queries) {
        if (task < Geometry::query_heads) {
            const auto offset = (m * Geometry::query_heads + task) * 256;
            vector_head<256, VectorWeights>(a.q + offset, a.wq, a.out_q + offset, a.epsilon,
                                            coefficients, lane, live);
            return;
        }
        if (task < Geometry::key_task) {
            const auto offset = (m * 4 + task - Geometry::query_heads) * 128;
            vector_head<128, VectorWeights>(a.iq + offset, a.wiq, a.out_iq + offset, a.epsilon,
                                            coefficients, lane, live);
            return;
        }
    }
    if (live && task < Geometry::raw_task) {
        if constexpr ((Storage == KvCacheStorage::Fp8E4M3Row256 ||
                       Storage == KvCacheStorage::BFloat16) &&
                      KvHeads == 4) {
            prepare_kv_vector<Storage, KvHeads>(a, task - Geometry::key_task, t, b, lane,
                                                coefficients);
        } else if constexpr (Storage == KvCacheStorage::Nvfp4Group16 ||
                             Storage == KvCacheStorage::Fp8KeyNvfp4Value) {
            __shared__ float scratch[KvHeads][256];
            prepare_kv<Storage, KvHeads>(a, task - Geometry::key_task, t, b, lane,
                                         coefficients[lane], scratch[task - Geometry::key_task]);
        } else {
            prepare_kv<Storage, KvHeads>(a, task - Geometry::key_task, t, b, lane,
                                         coefficients[lane]);
        }
    } else if (live && task == Geometry::raw_task) {
        prepare_raw<BoundedPhase>(a, t, b, lane, m);
    }
}

} // namespace ninfer::ops::detail::qsa
