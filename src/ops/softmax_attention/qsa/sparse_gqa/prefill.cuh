#pragma once
#include "ops/softmax_attention/qsa/sparse_gqa/kernel.cuh"

namespace ninfer::ops::detail::qsa {

template <KvCacheStorage Storage, int N, int Queries, bool Vector,
          SparseGqaLoad Load = SparseGqaLoad::Direct, int KvHeads>
__device__ __forceinline__ void
sparse_gqa_all_select(SparseGqaKernelArguments<KvHeads> a,
                      SparseGqaStorage<Storage, N, Queries, Load>& s, std::int64_t first, int head,
                      std::uint16_t* qs) {
    constexpr int Nt = N / 8, Ks = N / 16;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const auto total   = std::int64_t(a.tokens) * a.batch;
    const int b        = first / a.tokens;
    const auto last    = min(first + Queries, std::int64_t(b) * a.tokens + a.valid[b]);
    const int end      = last > first ? a.positions[last - 1] + 1 : 0;
    const auto q       = first + warp;
    const bool live    = q < last;
    const int position = live ? a.positions[q] : -1;
    auto* ks           = s.kv;
    auto* vs           = s.kv + N * 256;
    sparse_gqa_prepare_query<Storage, Queries, Queries>(a, first, last, head, qs);
    unsigned qf[4];
    const int ar     = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
    float acc[32][4] = {};
    OnlineSoftmaxRow state[2];
    const int gid = lane >> 2, lid = lane & 3;
    constexpr float scale = common::kLog2E / 16;
    const SparseGqaWork work{end / 4, end / 4, end, 0, 1};
    unsigned key_scale = 127;
    if (end) {
        sparse_gqa_addresses<Storage>(a, s, work, last - 1, head, 0, end);
        key_scale = sparse_gqa_stage<Storage, N, Queries, Vector, true>(a, s, 0, end);
        common::cp_commit();
    }
    for (int start = 0; start < end; start += N) {
        const bool key_half = sparse_gqa_wait_stage<Storage, true>(s, key_scale);
        sparse_gqa_expand_stage<Storage, true>(s, key_half, key_scale);
        unsigned value_scale =
            sparse_gqa_stage<Storage, N, Queries, Vector, false>(a, s, start, end);
        common::cp_commit();
        float score[Nt][4] = {};
#pragma unroll
        for (int n = 0; n < Nt; ++n) {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                unsigned bf[2];
                const int r = n * 8 + (lane & 7), d = k * 16 + ((lane >> 3) & 1) * 8;
                common::ldmatrix_x2(bf[0], bf[1],
                                    common::smem_addr(ks + r * 256 + common::swizzle_xor_64(r, d)));
                const int qd = k * 16 + (lane >> 4) * 8;
                common::ldmatrix_x4(
                    qf[0], qf[1], qf[2], qf[3],
                    common::smem_addr(qs + ar * 256 + common::swizzle_xor_64(ar, qd)));
                sparse_gqa_qk<Storage>(score[n], qf, bf);
            }
        }
        const bool value_half = sparse_gqa_wait_stage<Storage, false>(s, value_scale);
        sparse_gqa_expand_stage<Storage, false>(s, value_half, value_scale);
        if (start + N < end) {
            sparse_gqa_addresses<Storage>(a, s, work, last - 1, head, start + N, end);
            key_scale = sparse_gqa_stage<Storage, N, Queries, Vector, true>(a, s, start + N, end);
            common::cp_commit();
        }
        float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int n = 0; n < Nt; ++n)
            for (int j = 0; j < 4; ++j) {
                if (start + n * 8 + lid * 2 + (j & 1) > position) score[n][j] = -CUDART_INF_F;
                maximum[j / 2] = fmaxf(maximum[j / 2], score[n][j]);
            }
        const float alpha[2] = {
            state[0].update(common::warp_max<4>(maximum[0], 0xffffffffu), scale),
            state[1].update(common::warp_max<4>(maximum[1], 0xffffffffu), scale)};
        unsigned pf[Ks][4];
        float sum[2] = {};
#pragma unroll
        for (int n = 0; n < Nt; ++n) {
            const float p0 = state[0].probability(score[n][0], scale),
                        p1 = state[0].probability(score[n][1], scale),
                        p2 = state[1].probability(score[n][2], scale),
                        p3 = state[1].probability(score[n][3], scale);
            sum[0] += p0 + p1;
            sum[1] += p2 + p3;
            pf[n / 2][n % 2 * 2]     = common::pack_f16x2(p0, p1);
            pf[n / 2][n % 2 * 2 + 1] = common::pack_f16x2(p2, p3);
        }
        state[0].accumulate(alpha[0], sum[0]);
        state[1].accumulate(alpha[1], sum[1]);
#pragma unroll
        for (int n = 0; n < 32; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[n][j] *= alpha[j / 2];
#pragma unroll
            for (int k = 0; k < Ks; ++k) {
                unsigned vf[2];
                const int r = k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), d = n * 8;
                common::ldmatrix_x2_t(
                    vf[0], vf[1], common::smem_addr(vs + r * 256 + common::swizzle_xor_64(r, d)));
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
        if (q >= total || h >= 12) continue;
        const float inv = live ? __frcp_rn(state[j].sum) : 0;
#pragma unroll
        for (int n = 0; n < 32; ++n) {
            const int d = n * 8 + lid * 2;
            if (live) {
                if constexpr (kSparseRotatedValue<Storage>)
                    *reinterpret_cast<float2*>(s.partial + (warp * 16 + h) * 256 + d) =
                        make_float2(acc[n][j * 2] * inv, acc[n][j * 2 + 1] * inv);
                else
                    sparse_gqa_store(a, q, head * 12 + h, d, acc[n][j * 2] * inv,
                                     acc[n][j * 2 + 1] * inv);
            } else
                *reinterpret_cast<unsigned*>(a.out + (q * a.query_heads + head * 12 + h) * 256 +
                                             d) = 0;
        }
    }
    if constexpr (kSparseRotatedValue<Storage>) {
        __syncthreads();
        sparse_gqa_inverse_rows<Queries, Queries>(a, first, last, head, s.partial);
    }
}

template <KvCacheStorage Storage, int N, int Queries, bool Vector,
          SparseGqaLoad Load = SparseGqaLoad::Direct, int KvHeads>
__launch_bounds__(Queries * 32) __global__
    void sparse_gqa_all_select_kernel(SparseGqaKernelArguments<KvHeads> a) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto& s          = *reinterpret_cast<SparseGqaStorage<Storage, N, Queries, Load>*>(smem);
    const auto first = a.begin + std::int64_t(a.group_query(blockIdx.x)) * Queries,
               total = std::int64_t(a.tokens) * a.batch;
    const int head   = a.group_head(blockIdx.x);
    auto* qs         = reinterpret_cast<std::uint16_t*>(
        smem + (sizeof(SparseGqaStorage<Storage, N, Queries, Load>) + 15) / 16 * 16);
    if (sparse_gqa_all_selected<Queries>(a, first)) {
        sparse_gqa_all_select<Storage, N, Queries, Vector, Load>(a, s, first, head, qs);
    } else {
#pragma unroll 1
        for (int i = 0; i < Queries && first + i < total; ++i) {
            sparse_gqa_compute<Storage, N, Queries, Vector, Load>(a, s, first + i, head, 0);
            __syncthreads();
        }
    }
}
} // namespace ninfer::ops::detail::qsa
