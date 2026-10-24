#pragma once

#include "ops/softmax_attention/kpool_mla/sparse_mla/arguments.h"
#include "ops/softmax_attention/kernels/softmax.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::kpool {

// Probabilities are private BF16 MMA operands. Use the native FTZ exponential
// without software emulation for FP32 subnormals.
// Keep an unchanged running maximum exact so PV need not rescale its large
// accumulator on tiles that do not increase the maximum.
__device__ __forceinline__ float update_latent_softmax(OnlineSoftmaxRow& state, float maximum,
                                                       float scale) {
    const float previous = state.maximum;
    state.maximum        = fmaxf(previous, maximum);
    state.scaled_maximum = state.maximum * scale;
    return previous == state.maximum
               ? 1.f
               : common::exp2_approx_ftz(__fmaf_rn(previous, scale, -state.scaled_maximum));
}

template <int Tokens, int HeadGroups, bool IndependentHeads>
struct MlaAttentionStorage {
    static constexpr int Stages = 2, Parts = 2, Phases = IndependentHeads ? 2 : 1;
    __nv_bfloat16 latent[Stages * (Tokens / 4) * 2304];

    union {
        float4 scores[Parts - 1][Tokens / 8][32];
        __nv_bfloat16 probabilities[16 * Tokens];
    } tile[Phases][HeadGroups];

    std::uint64_t full[Stages], empty[IndependentHeads ? Stages : 1];
    float alpha[Phases][16 * HeadGroups];
    int chunks[512];
};

// Two warps divide the latent features for each group of 16 heads. The same
// gathered pool serves QK and PV, and is reused across all head groups in the CTA.
// Independent head groups double-buffer P as well as KV; completion barriers let
// one group begin its next QK while another finishes PV on the preceding tile.
template <int Tokens, int HeadGroups, bool IndependentHeads>
__global__ __launch_bounds__(64 * HeadGroups, HeadGroups == 2 ? 2 : 1) void mla_attention_kernel(
    const __grid_constant__ SparseMlaArguments a, const __grid_constant__ CUtensorMap map) {
    if (a.direct) return;
    constexpr int Parts = 2, Stages = 2;
    constexpr int Threads = 32 * Parts * HeadGroups;
    const int local = blockIdx.x / (4 / HeadGroups), split = blockIdx.y;
    const int group      = threadIdx.x / (32 * Parts);
    const int head_base  = blockIdx.x % (4 / HeadGroups) * (16 * HeadGroups);
    const int first_head = head_base + group * 16;
    const auto work      = a.work(local);
    if (!work.size) return;
    const int tiles  = (work.size + Tokens - 1) / Tokens;
    const int splits = min(a.splits, tiles);
    if (split >= splits) return;
    const int start = split * tiles / splits * Tokens;
    const int stop  = min(work.size, (split + 1) * tiles / splits * Tokens);
    const int tid = threadIdx.x, lane = tid % 32, part = tid / 32 % Parts;
    const int row = lane / 4, col = lane % 4;
    const auto* table =
        a.tables + std::int64_t(a.table_rows[(a.begin + local) / a.tokens]) * a.table_stride;
    if (work.size == 1) {
        // No complete pool was selected. The only member of J is the first token
        // of the incomplete tail, so u=c independently of query and U_K.
        const int token   = work.end - 1;
        const auto* value = reinterpret_cast<const __nv_bfloat16*>(
                                reinterpret_cast<const char*>(a.latent) + table[token / 64]) +
                            (token % 64) * 512;
        for (int i = tid; i < 16 * HeadGroups * 256; i += Threads) {
            const int head = head_base + i / 256, d = (i % 256) * 2;
            const auto pair = *reinterpret_cast<const __nv_bfloat162*>(value + d);
            *reinterpret_cast<__nv_bfloat162*>(a.mapped + (local * 64 + head) * 512 + d) = pair;
        }
        return;
    }
    extern __shared__ __align__(1024) unsigned char partitioned_shared[];
    auto& s = *reinterpret_cast<MlaAttentionStorage<Tokens, HeadGroups, IndependentHeads>*>(
        partitioned_shared);
    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < Stages; ++i) {
            common::cta_mbarrier_init(&s.full[i], 1);
            if constexpr (IndependentHeads) common::cta_mbarrier_init(&s.empty[i], Threads);
        }
        common::cta_mbarrier_fence_init();
    }
    // Resolve only this split's selected pools once. Keeping the indirection out
    // of the mainloop lets the idle QK partner prefetch while softmax runs.
    for (int i = start / 4 + tid; i < min(work.pools, (stop + 3) / 4); i += Threads) {
        const int token =
            (work.pools == work.end / 4 ? i : a.ids[std::int64_t(work.row) * 512 + i]) * 4;
        s.chunks[i] =
            int((std::uint64_t(__ldg(table + token / 64)) >> 7) + unsigned(token % 64) * 8);
    }
    __syncthreads();
    unsigned qf[32 / Parts][4];
    // Stage Q through the initially free latent buffer. Vector global reads and
    // LDSM replace scalar fragment loads; retire them before the first TMA tile.
    for (int i = tid; i < HeadGroups * 16 * 64; i += Threads) {
        const int h = i / 64, d = (i % 64) * 8;
        common::cp_async<16, Cache::cg>(s.latent + ((h * 512 + d) ^ ((h & 7) * 8)),
                                        a.mapped + (local * 64 + head_base + h) * 512 + d);
    }
    common::cp_commit();
    common::cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 32 / Parts; ++k) {
        const int h = group * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int d = part * (512 / Parts) + k * 16 + (lane >> 4) * 8;
        common::ldmatrix_x4(qf[k][0], qf[k][1], qf[k][2], qf[k][3],
                            common::smem_addr(s.latent + ((h * 512 + d) ^ ((h & 7) * 8))));
    }
    __syncthreads();
    float acc[64 / Parts][4]{};
    OnlineSoftmaxRow state[2];
    const float scale     = a.scale * common::kLog2E;
    const auto heads_sync = [&] {
        if (group == 0)
            asm volatile("bar.sync 1, %0;" ::"n"(32 * Parts) : "memory");
        else if (group == 1)
            asm volatile("bar.sync 2, %0;" ::"n"(32 * Parts) : "memory");
        else if (group == 2)
            asm volatile("bar.sync 3, %0;" ::"n"(32 * Parts) : "memory");
        else
            asm volatile("bar.sync 4, %0;" ::"n"(32 * Parts) : "memory");
    };
    // A TMA transaction transposes a complete Kpool4 (4 KiB). Padding successive
    // pools by 512 bytes gives the 8-token LDSM fragments distinct bank phases.
    const auto offset = [](int t, int d) {
        const int logical = t / 4 * 2304 + d / 64 * 256 + (t % 4) * 64 + d % 64;
        return logical ^ (((logical / 64) & 7) * 8);
    };
    const auto stage = [&](int k0, int buffer) {
        if (part == 0) return;
        if constexpr (IndependentHeads) {
            const int generation = (k0 - start) / Tokens / Stages;
            if (generation) common::cta_mbarrier_wait(&s.empty[buffer], (generation - 1) & 1);
        }
        constexpr int PoolsPerWarp = (Tokens / 4 + HeadGroups - 1) / HeadGroups;
        const int producer         = group * 32 + lane;
        const int tail = work.end % 4, tail_pool = work.pools - k0 / 4;
        const bool has_tail =
            tail && tail_pool >= 0 && tail_pool < Tokens / 4 && k0 + tail_pool * 4 < stop;
        if (producer == 0)
            common::cta_mbarrier_arrive_expect_tx(&s.full[buffer],
                                                  Tokens * 1024 - (has_tail ? 4096 : 0));
#pragma unroll
        for (int j = 0; j < PoolsPerWarp; ++j) {
            const int pool = group * PoolsPerWarp + j;
            if (lane == 0 && pool < Tokens / 4 && !(has_tail && pool == tail_pool)) {
                const int ordinal  = k0 + pool * 4;
                const bool present = ordinal < stop;
                const int chunk    = present ? s.chunks[ordinal / 4] : 0;
                common::tma_load_3d(s.latent + buffer * (Tokens / 4) * 2304 + pool * 2304, &map, 0,
                                    present ? 0 : 4, chunk, &s.full[buffer]);
            }
        }
        if (has_tail) {
            const int token = work.end / 4 * 4;
            const auto* base =
                reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const char*>(a.latent) +
                                                       __ldg(table + token / 64)) +
                (token % 64) * 512;
#pragma unroll
            for (int i = producer; i < 4 * 64; i += HeadGroups * 32) {
                const int t = i / 64, d = i % 64 * 8;
                auto* dst = s.latent + buffer * (Tokens / 4) * 2304 + offset(tail_pool * 4 + t, d);
                if (t < tail)
                    common::cp_async<16, Cache::cg>(dst, base + t * 512 + d);
                else
                    common::store_vec(dst, make_int4(0, 0, 0, 0));
            }
            common::cp_commit();
        }
    };
    stage(start, 0);
    for (int k0 = start, iteration = 0; k0 < stop; k0 += Tokens, ++iteration) {
        const int buffer    = iteration % Stages;
        const auto* latent  = s.latent + buffer * (Tokens / 4) * 2304;
        const int tail_pool = work.pools - k0 / 4;
        if (work.end % 4 && tail_pool >= 0 && tail_pool < Tokens / 4) {
            if (part == 1) common::cp_wait<0>();
            // The incomplete pool uses cp.async, outside the TMA byte count.
            if constexpr (IndependentHeads) __syncthreads();
        }
        if constexpr (IndependentHeads) {
            common::cta_mbarrier_wait(&s.full[buffer], (iteration / Stages) & 1);
        } else {
            if (tid == 0) common::cta_mbarrier_wait(&s.full[buffer], (iteration / Stages) & 1);
            // Retire preceding PV reads before reusing its latent/P storage.
            __syncthreads();
        }
        const int phase = IndependentHeads ? buffer : 0;
        float score[Tokens / 8][4]{};
        const unsigned qk_address = common::smem_addr(
            latent + offset(lane & 7, part * (512 / Parts) + ((lane >> 3) & 1) * 8));
        const unsigned pv_address = common::smem_addr(
            latent + offset((lane & 7) + ((lane >> 3) & 1) * 8, part * (512 / Parts)));
#pragma unroll
        for (int k = 0; k < 32 / Parts; ++k) {
#pragma unroll
            for (int n = 0; n < Tokens / 8; ++n) {
                unsigned bf[2];
                common::ldmatrix_x2(bf[0], bf[1],
                                    (qk_address ^ ((k % 4) * 32) ^ ((k / 4 % 2) * 64)) +
                                        (k / 4) * 512 + n * 9216);
                common::mma_bf16(score[n][0], score[n][1], score[n][2], score[n][3], qf[k][0],
                                 qf[k][1], qf[k][2], qf[k][3], bf[0], bf[1]);
            }
        }
        if (part != 0) {
#pragma unroll
            for (int n = 0; n < Tokens / 8; ++n) {
                s.tile[phase][group].scores[part - 1][n][lane] =
                    make_float4(score[n][0], score[n][1], score[n][2], score[n][3]);
            }
        }
        heads_sync();
        if (part == 0) {
            float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
            for (int n = 0; n < Tokens / 8; ++n) {
                const float4 other = s.tile[phase][group].scores[0][n][lane];
                score[n][0] += other.x;
                score[n][1] += other.y;
                score[n][2] += other.z;
                score[n][3] += other.w;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (k0 + n * 8 + col * 2 + j % 2 >= stop) score[n][j] = -CUDART_INF_F;
                    maximum[j / 2] = fmaxf(maximum[j / 2], score[n][j]);
                }
            }
            const float alpha[2] = {
                update_latent_softmax(state[0], common::warp_max<4>(maximum[0]), scale),
                update_latent_softmax(state[1], common::warp_max<4>(maximum[1]), scale)};
            // All partial scores are now in registers before the union becomes P.
            __syncwarp();
            float sum[2]{};
#pragma unroll
            for (int n = 0; n < Tokens / 8; ++n)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float p0 = common::exp2_approx_ftz(
                        __fmaf_rn(score[n][2 * j], scale, -state[j].scaled_maximum));
                    const float p1 = common::exp2_approx_ftz(
                        __fmaf_rn(score[n][2 * j + 1], scale, -state[j].scaled_maximum));
                    sum[j] += p0 + p1;
                    const int r = row + j * 8, d = n * 8 + col * 2;
                    *reinterpret_cast<unsigned*>(s.tile[phase][group].probabilities +
                                                 ((r * Tokens + d) ^ ((r & 7) << 3))) =
                        common::pack_bf16x2(p0, p1);
                }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                state[j].accumulate(alpha[j], sum[j]);
                if (col == 0) s.alpha[phase][group * 16 + row + j * 8] = alpha[j];
            }
        } else if (k0 + Tokens < stop) {
            // This warp's QK partial is already consumed. Overlap the next pool
            // transactions with its partner's softmax and this tile's PV.
            stage(k0 + Tokens, 1 - buffer);
        }
        heads_sync();
        const float alpha[2] = {s.alpha[phase][group * 16 + row],
                                s.alpha[phase][group * 16 + row + 8]};
        if (__any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
            for (int n = 0; n < 64 / Parts; ++n)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[n][j] *= alpha[j / 2];
        }
#pragma unroll 1
        for (int k = 0; k < Tokens / 16; ++k) {
            unsigned pf[4];
            const int pr = (lane & 7) + ((lane >> 3) & 1) * 8, pd = k * 16 + (lane >> 4) * 8;
            common::ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                                common::smem_addr(s.tile[phase][group].probabilities +
                                                  ((pr * Tokens + pd) ^ ((pr & 7) << 3))));
#pragma unroll
            for (int n = 0; n < 64 / Parts; ++n) {
                unsigned vf[2];
                common::ldmatrix_x2_t(vf[0], vf[1],
                                      (pv_address ^ ((n % 8) * 16) ^ ((n / 8 % 2) * 64)) +
                                          (n / 8) * 512 + k * 18432);
                common::mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2],
                                 pf[3], vf[0], vf[1]);
            }
        }
        if constexpr (IndependentHeads) common::cta_mbarrier_arrive(&s.empty[buffer]);
    }
    // The final inverse reuses alpha; retire every PV reader before overwriting it.
    __syncthreads();
    if (part == 0) {
        state[0].finish();
        state[1].finish();
#pragma unroll
        for (int j = 0; j < 2; ++j)
            if (col == 0) {
                if (splits == 1)
                    s.alpha[0][group * 16 + row + j * 8] = __frcp_rn(state[j].sum);
                else {
                    const auto i     = a.partial(local, split, first_head + row + j * 8);
                    a.maximum[i]     = state[j].maximum * scale;
                    a.denominator[i] = state[j].sum;
                }
            }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int h     = first_head + row + j * 8;
        const float inv = splits == 1 ? s.alpha[0][group * 16 + row + j * 8] : 1.f;
#pragma unroll
        for (int n = 0; n < 64 / Parts; ++n) {
            const int d = part * (512 / Parts) + n * 8 + col * 2;
            if (splits == 1)
                *reinterpret_cast<unsigned*>(a.mapped + (local * 64 + h) * 512 + d) =
                    common::pack_bf16x2(acc[n][2 * j] * inv, acc[n][2 * j + 1] * inv);
            else
                *reinterpret_cast<float2*>(a.numerator + a.partial(local, split, h) * 512 + d) =
                    make_float2(acc[n][2 * j], acc[n][2 * j + 1]);
        }
    }
}

__global__
__launch_bounds__(256) void mla_merge_kernel(const __grid_constant__ SparseMlaArguments a) {
    if (a.splits == 1) return;
    const int local = blockIdx.x / 64, head = blockIdx.x % 64, tid = threadIdx.x;
    const auto work = a.work(local);
    if (!work.size) return;
    const int splits = min(a.splits, (work.size + a.tile - 1) / a.tile);
    if (splits == 1) return;
    __shared__ float weights[32], denominator;
    if (tid < 32) {
        float maximum = tid < splits ? a.maximum[a.partial(local, tid, head)] : -CUDART_INF_F;
        maximum       = common::warp_max(maximum);
        float sum     = 0;
        if (tid < splits) {
            const auto i = a.partial(local, tid, head);
            weights[tid] = common::exp2_approx_ftz(a.maximum[i] - maximum);
            sum          = weights[tid] * a.denominator[i];
        }
        sum = common::warp_sum(sum);
        if (tid == 0) denominator = sum;
    }
    __syncthreads();
    float x = 0, y = 0;
    for (int s = 0; s < splits; ++s) {
        const auto value = *reinterpret_cast<const float2*>(
            a.numerator + a.partial(local, s, head) * 512 + tid * 2);
        x += weights[s] * value.x;
        y += weights[s] * value.y;
    }
    const float inv = __frcp_rn(denominator);
    *reinterpret_cast<unsigned*>(a.mapped + (local * 64 + head) * 512 + tid * 2) =
        common::pack_bf16x2(x * inv, y * inv);
}
} // namespace ninfer::ops::detail::kpool
