#pragma once
#include "ops/softmax_attention/qsa/sparse_gqa/grouped_tile.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/epilogue.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail::qsa {
template <KvCacheStorage Storage, int N, int Parts, int Queries, bool Vector, SparseGqaLoad Load,
          int KvHeads>
__global__ __launch_bounds__(Queries* Parts * 24) void sparse_gqa_grouped_kernel(
    const __grid_constant__ SparseGqaKernelArguments<KvHeads> a) {
    static_assert(Queries % 4 == 0 && N % 16 == 0 && 256 % (Parts * 8) == 0);
    // Twelve heads per query occupy consecutive MMA rows, without per-query padding.
    constexpr int Rows = Queries * 12, Warps = Rows / 16 * Parts, Columns = 256 / Parts / 8;
    extern __shared__ __align__(16) unsigned char packed_smem[];
    using Tile       = SparseGqaGroupedTile<Storage, N, Warps, Vector, Load>;
    auto& tile       = *reinterpret_cast<Tile*>(packed_smem);
    auto& s          = static_cast<typename Tile::Base&>(tile);
    auto* qs   = reinterpret_cast<std::uint16_t*>(packed_smem + (sizeof(Tile) + 15) / 16 * 16);
    const auto first = a.begin + std::int64_t(a.group_query(blockIdx.x)) * Queries,
               total = std::int64_t(a.tokens) * a.batch;
    const int head = a.group_head(blockIdx.x), tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    if (!sparse_gqa_all_selected<Queries>(a, first)) return;
    const int b     = first / a.tokens;
    const auto last = min(first + Queries, std::int64_t(b) * a.tokens + a.valid[b]);
    const int end   = last > first ? a.positions[last - 1] + 1 : 0;
    if constexpr (kSparseQuantized<Storage>) {
        for (int row = warp; row < Rows; row += Warps) {
            const auto q = first + row / 12;
            const int h  = row % 12;
            float values[8]{};
            if (q < last) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    values[i] = __bfloat162float(
                        a.query[(q * a.query_heads + head * 12 + h) * 256 + lane + 32 * i]);
                kv::normalized_hadamard_d256_inplace(values, lane);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
                qs[row * 256 + common::swizzle_xor_64(row, lane + 32 * i)] =
                    __half_as_ushort(__float2half_rn(values[i]));
        }
    } else {
        for (int i = tid; i < Rows * 32; i += Warps * 32) {
            const int row = i / 32, d = i % 32 * 8;
            auto* dst    = qs + row * 256 + common::swizzle_xor_64(row, d);
            const auto q = first + row / 12;
            if (q < last)
                common::cp_async<16>(dst, a.query +
                                              (q * a.query_heads + head * 12 + row % 12) * 256 + d);
            else
                common::store_vec(dst, make_int4(0, 0, 0, 0));
        }
        common::cp_commit();
        common::cp_wait<0>();
    }
    __syncthreads();
    const int rowbase = warp / Parts * 16, gid = lane / 4, lid = lane % 4, part = warp % Parts;
    int positions[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const auto q = first + (rowbase + gid + 8 * j) / 12;
        positions[j] = q < last ? a.positions[q] : -1;
    }
    auto* ks = s.kv;
    auto* vs = s.kv + N * 256;
    float acc[Columns][4]{};
    OnlineSoftmaxRow state[2];
    constexpr float scale = common::kLog2E / 16;
    const SparseGqaWork work{end / 4, end / 4, end, 0, 1};
    typename Tile::State scale_state;
    if (end) {
        sparse_gqa_addresses<Storage>(a, s, work, last - 1, head, 0, end);
        scale_state = tile.prefetch(a, 0, end);
        common::cp_commit();
    }
    for (int k0 = 0; k0 < end; k0 += N) {
        tile.wait_key(scale_state);
        tile.template expand<true>(scale_state);
        if constexpr (Tile::joint_scales)
            tile.stage_value(a, k0, end);
        else
            tile.stage_value(scale_state, a, k0, end);
        common::cp_commit();
        float score[N / 8][4]{};
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            unsigned qf[4];
            const int qr = rowbase + (lane & 7) + ((lane >> 3) & 1) * 8,
                      qd = k * 16 + (lane >> 4) * 8;
            common::ldmatrix_x4(qf[0], qf[1], qf[2], qf[3],
                                common::smem_addr(qs + qr * 256 + common::swizzle_xor_64(qr, qd)));
#pragma unroll
            for (int n = 0; n < N / 8; ++n) {
                unsigned bf[2];
                const int r = n * 8 + (lane & 7), d = k * 16 + ((lane >> 3) & 1) * 8;
                common::ldmatrix_x2(bf[0], bf[1],
                                    common::smem_addr(ks + r * 256 + common::swizzle_xor_64(r, d)));
                sparse_gqa_qk<Storage>(score[n], qf, bf);
            }
        }
        if constexpr (Tile::joint_scales)
            tile.wait_value();
        else
            tile.wait_value(scale_state);
        tile.template expand<false>(scale_state);
        if (k0 + N < end) {
            sparse_gqa_addresses<Storage>(a, s, work, last - 1, head, k0 + N, end);
            scale_state = tile.prefetch(a, k0 + N, end);
            common::cp_commit();
        }
        float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int n = 0; n < N / 8; ++n)
            for (int j = 0; j < 4; ++j) {
                if (k0 + n * 8 + lid * 2 + j % 2 > positions[j / 2]) score[n][j] = -CUDART_INF_F;
                maximum[j / 2] = fmaxf(maximum[j / 2], score[n][j]);
            }
        const float alpha[2] = {
            state[0].update(common::warp_max<4>(maximum[0], 0xffffffffu), scale),
            state[1].update(common::warp_max<4>(maximum[1], 0xffffffffu), scale)};
        unsigned pf[N / 16][4];
        float sum[2]{};
#pragma unroll
        for (int n = 0; n < N / 8; ++n) {
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
        for (int n = 0; n < Columns; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[n][j] *= alpha[j / 2];
#pragma unroll
            for (int k = 0; k < N / 16; ++k) {
                unsigned vf[2];
                const int r = k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7),
                          d = part * (256 / Parts) + n * 8;
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
    // All Q/K/V reads have finished; the entire allocation now holds FP32 value rows.
    auto* result = reinterpret_cast<float*>(packed_smem);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int row = rowbase + gid + j * 8, h = row % 12;
        const auto q = first + row / 12;
        if (q >= total) continue;
        const bool live = q < last;
        const float inv = live ? __frcp_rn(state[j].sum) : 0;
#pragma unroll
        for (int n = 0; n < Columns; ++n) {
            const int d = part * (256 / Parts) + n * 8 + lid * 2;
            if (live) {
                if constexpr (kSparseRotatedValue<Storage>)
                    *reinterpret_cast<float2*>(result + row * 256 + d) =
                        make_float2(acc[n][2 * j] * inv, acc[n][2 * j + 1] * inv);
                else
                    sparse_gqa_store(a, q, head * 12 + h, d, acc[n][2 * j] * inv,
                                     acc[n][2 * j + 1] * inv);
            } else
                *reinterpret_cast<unsigned*>(a.out + (q * a.query_heads + head * 12 + h) * 256 +
                                             d) = 0;
        }
    }
    if constexpr (kSparseRotatedValue<Storage>) {
        __syncthreads();
        for (int row = warp; row < Rows; row += Warps) {
            const auto q = first + row / 12;
            if (q >= last) continue;
            float values[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) values[i] = result[row * 256 + lane + 32 * i];
            sparse_gqa_inverse_gate(a, q, head * 12 + row % 12, values);
        }
    }
}
} // namespace ninfer::ops::detail::qsa
