#pragma once

#include "ops/softmax_attention/qsa/sparse_gqa/tile.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/epilogue.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail::qsa {
template <KvCacheStorage Storage, int N, int Parts, bool Vector, SparseGqaLoad Load,
          int PrefixQueries = 0, int KvHeads>
__launch_bounds__(Parts * 32) __global__
    void sparse_gqa_columns_kernel(const __grid_constant__ SparseGqaKernelArguments<KvHeads> a) {
    static_assert(N >= 16 && N % 16 == 0 && Parts > 0 && 256 % (Parts * 8) == 0);
    extern __shared__ __align__(16) unsigned char shared[];
    auto& s =
        *reinterpret_cast<SparseGqaStorage<Storage, N, Parts, Load, SparseGqaWarpSplit::Output>*>(
            shared);
    const auto q   = a.begin + a.group_query(blockIdx.x);
    const int head = a.group_head(blockIdx.x), split = blockIdx.y;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    if constexpr (PrefixQueries > 0) {
        const auto first = a.begin + (q - a.begin) / PrefixQueries * PrefixQueries;
        if (sparse_gqa_all_selected<PrefixQueries>(a, first)) return;
    }
    const auto work = a.work(q);
    if (work.size == 0) {
        if (a.numerator == nullptr)
            for (int i = tid; i < 12 * 128; i += Parts * 32)
                *reinterpret_cast<unsigned*>(
                    a.out + (q * a.query_heads + head * 12 + i / 128) * 256 + (i % 128) * 2) = 0;
        return;
    }
    if (split >= work.splits) return;
    const int start = split * work.tiles / work.splits * N;
    const int end   = min(work.size, (split + 1) * work.tiles / work.splits * N);
    auto* ks        = s.kv;
    auto* vs        = s.kv + N * 256;
    sparse_gqa_prepare_query<Storage, Parts, 1>(a, q, q + 1, head, ks);
    unsigned qf[16][4];
    const int ar = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int d = k * 16 + (lane >> 4) * 8;
        common::ldmatrix_x4(qf[k][0], qf[k][1], qf[k][2], qf[k][3],
                            common::smem_addr(ks + ar * 256 + common::swizzle_xor_64(ar, d)));
    }
    __syncthreads();
    // Warps share K/V but own disjoint output dimensions. QK and softmax are replicated;
    // the PV accumulators need no cross-warp reduction.
    constexpr int Columns = 256 / Parts / 8;
    float acc[Columns][4]{};
    OnlineSoftmaxRow state[2];
    constexpr float scale = common::kLog2E / 16.0f;
    const int gid = lane >> 2, lid = lane & 3;
    sparse_gqa_addresses<Storage>(a, s, work, q, head, start, end);
    sparse_gqa_stage<Storage, N, Parts, Vector, true>(a, s, start, end);
    common::cp_commit();
    for (int k0 = start; k0 < end; k0 += N) {
        common::cp_wait<0>();
        __syncthreads();
        sparse_gqa_expand_stage<Storage, true>(s);
        sparse_gqa_stage<Storage, N, Parts, Vector, false>(a, s, k0, end);
        common::cp_commit();
        float score[N / 8][4]{};
#pragma unroll
        for (int n = 0; n < N / 8; ++n) {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                unsigned bf[2];
                const int row = n * 8 + (lane & 7), d = k * 16 + ((lane >> 3) & 1) * 8;
                common::ldmatrix_x2(
                    bf[0], bf[1],
                    common::smem_addr(ks + row * 256 + common::swizzle_xor_64(row, d)));
                sparse_gqa_qk<Storage>(score[n], qf[k], bf);
            }
        }
        common::cp_wait<0>();
        __syncthreads();
        sparse_gqa_expand_stage<Storage, false>(s);
        if (k0 + N < end) {
            sparse_gqa_addresses<Storage>(a, s, work, q, head, k0 + N, end);
            sparse_gqa_stage<Storage, N, Parts, Vector, true>(a, s, k0 + N, end);
            common::cp_commit();
        }
        float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int n = 0; n < N / 8; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (k0 + n * 8 + lid * 2 + (j & 1) >= end) score[n][j] = -CUDART_INF_F;
                maximum[j / 2] = fmaxf(maximum[j / 2], score[n][j]);
            }
        }
        const float alpha[2] = {
            state[0].update(common::warp_max<4>(maximum[0], 0xffffffffu), scale),
            state[1].update(common::warp_max<4>(maximum[1], 0xffffffffu), scale)};
        unsigned pf[N / 16][4];
        float tile_sum[2]{};
#pragma unroll
        for (int n = 0; n < N / 8; ++n) {
            const float p0 = state[0].probability(score[n][0], scale);
            const float p1 = state[0].probability(score[n][1], scale);
            const float p2 = state[1].probability(score[n][2], scale);
            const float p3 = state[1].probability(score[n][3], scale);
            tile_sum[0] += p0 + p1;
            tile_sum[1] += p2 + p3;
            pf[n / 2][(n % 2) * 2]     = common::pack_f16x2(p0, p1);
            pf[n / 2][(n % 2) * 2 + 1] = common::pack_f16x2(p2, p3);
        }
        state[0].accumulate(alpha[0], tile_sum[0]);
        state[1].accumulate(alpha[1], tile_sum[1]);
#pragma unroll
        for (int n = 0; n < Columns; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[n][j] *= alpha[j / 2];
#pragma unroll
            for (int k = 0; k < N / 16; ++k) {
                unsigned vf[2];
                const int row = k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7),
                          d   = warp * (256 / Parts) + n * 8;
                common::ldmatrix_x2_t(
                    vf[0], vf[1],
                    common::smem_addr(vs + row * 256 + common::swizzle_xor_64(row, d)));
                common::mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1],
                                pf[k][2], pf[k][3], vf[0], vf[1]);
            }
        }
        __syncthreads();
    }
    state[0].finish();
    state[1].finish();
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int h = gid + j * 8;
        if (h >= 12) continue;
        const auto row    = a.partial_row(q, split, head * 12 + h);
        const bool direct = a.numerator == nullptr;
        const float inv   = direct ? __frcp_rn(state[j].sum) : 1.0f;
#pragma unroll
        for (int n = 0; n < Columns; ++n) {
            const int d   = warp * (256 / Parts) + n * 8 + lid * 2;
            const float x = acc[n][2 * j] * inv, y = acc[n][2 * j + 1] * inv;
            if (direct) {
                if constexpr (kSparseRotatedValue<Storage>)
                    *reinterpret_cast<float2*>(s.partial + h * 256 + d) = make_float2(x, y);
                else
                    sparse_gqa_store(a, q, head * 12 + h, d, x, y);
            } else
                *reinterpret_cast<float2*>(a.numerator + row * 256 + d) = make_float2(x, y);
        }
        if (!direct && warp == 0 && lid == 0) {
            a.maximum[row] = state[j].maximum * scale;
            a.sum[row]     = state[j].sum;
        }
    }
    if constexpr (kSparseRotatedValue<Storage>) {
        if (a.numerator == nullptr) {
            __syncthreads();
            sparse_gqa_inverse_rows<Parts, 1>(a, q, q + 1, head, s.partial);
        }
    }
}
} // namespace ninfer::ops::detail::qsa
