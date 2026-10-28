#pragma once

#include "ops/softmax_attention/qsa/sparse_gqa/kernel.cuh"

namespace ninfer::ops::detail::qsa {

// The six-head geometry occupies the eight-column side of both MMA contractions.
// QK is K*Q^T and PV is V^T*P: all eight key groups of a warp cooperate on two heads.
__device__ __forceinline__ float sparse_gqa_key_max(float x) {
#pragma unroll
    for (int delta = 4; delta <= 16; delta *= 2)
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, delta));
    return x;
}

__device__ __forceinline__ float sparse_gqa_key_sum(float x) {
#pragma unroll
    for (int delta = 4; delta <= 16; delta *= 2) x += __shfl_xor_sync(0xffffffffu, x, delta);
    return x;
}

template <int Tiles>
__device__ __forceinline__ void
compact_probabilities(float (&score)[Tiles][4], OnlineSoftmaxRow (&state)[2], float (&alpha)[2],
                      unsigned (&pf)[Tiles][2], int key_begin, int position) {
    constexpr float scale = common::kLog2E / 16;
    const int lane = threadIdx.x % 32, gid = lane / 4, lid = lane % 4;
    float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
    for (int n = 0; n < Tiles; ++n) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int ordinal = key_begin + n * 16 + gid + (j / 2) * 8;
            if (ordinal > position) score[n][j] = -CUDART_INF_F;
            maximum[j % 2] = fmaxf(maximum[j % 2], score[n][j]);
        }
    }
    alpha[0] = state[0].update(sparse_gqa_key_max(maximum[0]), scale);
    alpha[1] = state[1].update(sparse_gqa_key_max(maximum[1]), scale);
    float tile_sum[2]{};
#pragma unroll
    for (int n = 0; n < Tiles; ++n) {
        float p[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            p[j] = state[j % 2].probability(score[n][j], scale);
            tile_sum[j % 2] += p[j];
        }
        const unsigned packed[2] = {common::pack_f16x2(p[0], p[1]), common::pack_f16x2(p[2], p[3])};
        const int source         = lid * 8 + gid / 2;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const unsigned lo = __shfl_sync(0xffffffffu, packed[j], source);
            const unsigned hi = __shfl_sync(0xffffffffu, packed[j], source + 4);
            pf[n][j]          = __byte_perm(lo, hi, (gid & 1) ? 0x7632 : 0x5410);
        }
    }
    state[0].accumulate(alpha[0], tile_sum[0]);
    state[1].accumulate(alpha[1], tile_sum[1]);
}

template <KvCacheStorage Storage, int N, int Warps, bool Vector, SparseGqaLoad Load, bool Prefix>
__device__ __forceinline__ void
sparse_gqa_compact(SparseGqaKernelArguments<4> a,
                   SparseGqaStorage<Storage, N, Warps, Load, SparseGqaWarpSplit::Sequence, 8>& s,
                   std::int64_t first, int head, int split, int packed = Prefix ? Warps : 1) {
    constexpr int PaddedHeads = 8;
    constexpr int Queries = Prefix ? Warps : 1, NK = Prefix ? N : N / Warps, Tiles = NK / 16;
    static_assert(NK % 16 == 0 && (!Prefix || Warps <= N));
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int gid = lane / 4, lid = lane % 4;
    const auto total = std::int64_t(a.tokens) * a.batch;
    const int batch  = first / a.tokens;
    const auto last  = min(first + packed, std::int64_t(batch) * a.tokens + a.valid[batch]);
    const auto q     = first + (Prefix ? warp : 0);
    const bool live  = q < last;
    auto work        = a.work(first);
    if constexpr (Prefix) {
        const int end   = last > first ? a.positions[last - 1] + 1 : 0;
        const int tiles = (end + N - 1) / N;
        work            = {end / 4, end / 4, end, tiles, min(a.splits, tiles)};
    }
    if (!work.size) {
        // Split kernels leave empty-query output ownership to the following merge.
        if (!a.numerator)
            for (int i = threadIdx.x; i < packed * 6 * 128; i += Warps * 32) {
                const auto dstq = first + i / (6 * 128);
                if (dstq < total)
                    *reinterpret_cast<unsigned*>(
                        a.out + (dstq * 24 + head * 6 + i / 128 % 6) * 256 + i % 128 * 2) = 0;
            }
        return;
    }
    if (split >= work.splits) return;
    const int start    = split * work.tiles / work.splits * N;
    const int end      = min(work.size, (split + 1) * work.tiles / work.splits * N);
    const int position = Prefix ? (live ? a.positions[q] : -1) : end - 1;
    auto* ks           = s.kv;
    auto* vs =
        s.kv + (Load == SparseGqaLoad::Aliased || Load == SparseGqaLoad::Reused ? 0 : N * 256);
    sparse_gqa_prepare_query<Storage, Warps, Queries, 8>(a, first, last, head, ks);
    unsigned qf[16][2];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int r = (Prefix ? warp * 8 : 0) + (lane & 7);
        const int d = k * 16 + ((lane >> 3) & 1) * 8;
        common::ldmatrix_x2(qf[k][0], qf[k][1],
                            common::smem_addr(ks + r * 256 + common::swizzle_xor_64(r, d)));
    }
    __syncthreads();
    float acc[16][4]{};
    OnlineSoftmaxRow state[2];
    constexpr float scale = common::kLog2E / 16;
    const auto address_q  = Prefix ? last - 1 : first;
    sparse_gqa_addresses<Storage>(a, s, work, address_q, head, start, end);
    sparse_gqa_stage<Storage, N, Warps, Vector, true>(a, s, start, end);
    common::cp_commit();
    for (int k0 = start; k0 < end; k0 += N) {
        common::cp_wait<0>();
        __syncthreads();
        sparse_gqa_expand_stage<Storage, true>(s);
        if constexpr (Load != SparseGqaLoad::Reused) {
            sparse_gqa_stage<Storage, N, Warps, Vector, false>(a, s, k0, end);
            common::cp_commit();
        }
        float score[Tiles][4]{};
#pragma unroll
        for (int n = 0; n < Tiles; ++n) {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                unsigned kf[4];
                const int r =
                    (Prefix ? 0 : warp * NK) + n * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int d = k * 16 + (lane >> 4) * 8;
                common::ldmatrix_x4(kf[0], kf[1], kf[2], kf[3],
                                    common::smem_addr(ks + r * 256 + common::swizzle_xor_64(r, d)));
                if constexpr (kSparseQuantized<Storage>)
                    common::mma_f16(score[n][0], score[n][1], score[n][2], score[n][3], kf[0],
                                    kf[1], kf[2], kf[3], qf[k][0], qf[k][1]);
                else
                    common::mma_bf16(score[n][0], score[n][1], score[n][2], score[n][3], kf[0],
                                     kf[1], kf[2], kf[3], qf[k][0], qf[k][1]);
            }
        }
        float alpha[2];
        unsigned pf[Tiles][2];
        // Four-warp tiles hide V latency and shorten score-fragment lifetimes here.
        // The one-warp schedule retains its measured late-softmax order.
        if constexpr (Warps > 1)
            compact_probabilities(score, state, alpha, pf, k0 + (Prefix ? 0 : warp * NK), position);
        if constexpr (Load == SparseGqaLoad::Reused) {
            __syncthreads();
            sparse_gqa_stage<Storage, N, Warps, Vector, false>(a, s, k0, end);
            common::cp_commit();
        }
        common::cp_wait<0>();
        __syncthreads();
        sparse_gqa_expand_stage<Storage, false>(s);
        if (Load != SparseGqaLoad::Reused && k0 + N < end) {
            sparse_gqa_addresses<Storage>(a, s, work, address_q, head, k0 + N, end);
            sparse_gqa_stage<Storage, N, Warps, Vector, true>(a, s, k0 + N, end);
            common::cp_commit();
        }
        if constexpr (Warps == 1)
            compact_probabilities(score, state, alpha, pf, k0 + warp * NK, position);
#pragma unroll
        for (int n = 0; n < 16; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[n][j] *= alpha[j % 2];
#pragma unroll
            for (int k = 0; k < Tiles; ++k) {
                unsigned vf[4];
                const int r = (Prefix ? 0 : warp * NK) + k * 16 + (lane & 7) + (lane >> 4) * 8;
                const int d = n * 16 + ((lane >> 3) & 1) * 8;
                common::ldmatrix_x4_t(
                    vf[0], vf[1], vf[2], vf[3],
                    common::smem_addr(vs + r * 256 + common::swizzle_xor_64(r, d)));
                common::mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], vf[0], vf[1], vf[2],
                                vf[3], pf[k][0], pf[k][1]);
            }
        }
        __syncthreads();
        if (Load == SparseGqaLoad::Reused && k0 + N < end) {
            sparse_gqa_addresses<Storage>(a, s, work, address_q, head, k0 + N, end);
            sparse_gqa_stage<Storage, N, Warps, Vector, true>(a, s, k0 + N, end);
            common::cp_commit();
        }
    }
    float maximum[2] = {state[0].maximum * scale, state[1].maximum * scale};
    float sum[2]     = {sparse_gqa_key_sum(state[0].sum), sparse_gqa_key_sum(state[1].sum)};
    if constexpr (!Prefix && Warps > 1) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = warp * PaddedHeads + lid * 2 + j;
            if (gid == 0) {
                s.statistics.maximum[row] = maximum[j];
                s.statistics.sum[row]     = sum[j];
            }
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                s.partial[row * 256 + n * 16 + gid]     = acc[n][j];
                s.partial[row * 256 + n * 16 + gid + 8] = acc[n][j + 2];
            }
        }
        __syncthreads();
        if (warp) return;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = lid * 2 + j;
            maximum[j]    = -CUDART_INF_F;
            sum[j]        = 0;
#pragma unroll
            for (int w = 0; w < Warps; ++w)
                maximum[j] = fmaxf(maximum[j], s.statistics.maximum[w * PaddedHeads + row]);
            float weights[Warps];
#pragma unroll
            for (int w = 0; w < Warps; ++w) {
                weights[w] =
                    softmax_log2_state_weight(s.statistics.maximum[w * PaddedHeads + row],
                                              s.statistics.sum[w * PaddedHeads + row], maximum[j]);
                sum[j] += weights[w] * s.statistics.sum[w * PaddedHeads + row];
            }
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                float x = 0, y = 0;
#pragma unroll
                for (int w = 0; w < Warps; ++w) {
                    x += weights[w] * s.partial[(w * PaddedHeads + row) * 256 + n * 16 + gid];
                    y += weights[w] * s.partial[(w * PaddedHeads + row) * 256 + n * 16 + gid + 8];
                }
                acc[n][j]     = x;
                acc[n][j + 2] = y;
            }
        }
    }
    if (q >= total || q >= first + packed) return;
    if (!a.numerator) {
        const float inv[2] = {live ? __frcp_rn(sum[0]) : 0, live ? __frcp_rn(sum[1]) : 0};
        // Redistribute FP32 attention before the gate. A compact loop keeps rounded
        // reciprocal's exceptional paths out of the fully unrolled MMA fragment stores,
        // and all lanes participate in the gate instead of only the live fragment columns.
#pragma unroll
        for (int n = 0; n < 16; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int h = lid * 2 + j % 2, d = n * 16 + gid + j / 2 * 8;
                s.partial[((Prefix ? warp * PaddedHeads : 0) + h) * 256 + d] =
                    acc[n][j] * inv[j % 2];
            }
        }
        __syncwarp();
        if constexpr (Storage == KvCacheStorage::BFloat16 ||
                      Storage == KvCacheStorage::Int8Group64 ||
                      Storage == KvCacheStorage::Nvfp4Group16) {
            if (live)
                sparse_gqa_compact_store_vector<kSparseRotatedValue<Storage>>(
                    a, q, head, s.partial + (Prefix ? warp * PaddedHeads : 0) * 256);
            else
                for (int i = lane; i < 6 * 128; i += 32)
                    *reinterpret_cast<unsigned*>(a.out + (q * 24 + head * 6 + i / 128) * 256 +
                                                 i % 128 * 2) = 0;
        } else {
#pragma unroll 1
            for (int i = lane; i < 6 * 128; i += 32) {
                const int h = i / 128, d = i % 128 * 2;
                if (live) {
                    const auto value = *reinterpret_cast<const float2*>(
                        s.partial + ((Prefix ? warp * PaddedHeads : 0) + h) * 256 + d);
                    sparse_gqa_store(a, q, head * 6 + h, d, value.x, value.y);
                } else
                    *reinterpret_cast<unsigned*>(a.out + (q * 24 + head * 6 + h) * 256 + d) = 0;
            }
        }
        return;
    }
#pragma unroll
    for (int n = 0; n < 16; ++n) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float neighbor = __shfl_xor_sync(0xffffffffu, acc[n][j], 4);
            const int h = lid * 2 + j % 2, d = n * 16 + gid + j / 2 * 8;
            if (!(gid & 1) && h < 6) {
                const auto row = a.partial_row(q, split, head * 6 + h);
                *reinterpret_cast<float2*>(a.numerator + row * 256 + d) =
                    make_float2(acc[n][j], neighbor);
            }
        }
    }
    if (gid == 0 && lid < 3) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const auto row = a.partial_row(q, split, head * 6 + lid * 2 + j);
            a.maximum[row] = maximum[j];
            a.sum[row]     = sum[j];
        }
    }
}

template <KvCacheStorage Storage, int N, int Warps, bool Vector, SparseGqaLoad Load,
          bool Prefix = false>
__launch_bounds__(Warps * 32) __global__
    void sparse_gqa_compact_kernel(SparseGqaKernelArguments<4> a, int slab_queries) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto& s = *reinterpret_cast<
        SparseGqaStorage<Storage, N, Warps, Load, SparseGqaWarpSplit::Sequence, 8>*>(smem);
    const auto group = std::int64_t(a.group_query(blockIdx.x));
    auto first       = a.begin + group * (Prefix ? Warps : 1);
    int packed       = Prefix ? Warps : 1;
    if constexpr (Storage != KvCacheStorage::Fp8E4M3Row256 && Prefix) {
        // Anchor complete packets at the slab end. Growing a fixed-end query window adds
        // an older leading packet without repartitioning all existing query/KV work.
        const int leading = (slab_queries - 1) % Warps + 1;
        first             = a.begin + (group ? leading + (group - 1) * Warps : 0);
        if (!group) packed = leading;
    }
    const int head = a.group_head(blockIdx.x);
    if constexpr (Prefix) {
        if (sparse_gqa_all_selected<Warps>(a, first, packed))
            sparse_gqa_compact<Storage, N, Warps, Vector, Load, true>(a, s, first, head, blockIdx.y,
                                                                      packed);
        else {
            const auto total = std::int64_t(a.tokens) * a.batch;
            for (int i = 0; i < packed && first + i < total; ++i) {
                sparse_gqa_compact<Storage, N, Warps, Vector, Load, false>(a, s, first + i, head,
                                                                           blockIdx.y);
                __syncthreads();
            }
        }
    } else
        sparse_gqa_compact<Storage, N, Warps, Vector, Load, false>(a, s, first, head, blockIdx.y);
}
} // namespace ninfer::ops::detail::qsa
