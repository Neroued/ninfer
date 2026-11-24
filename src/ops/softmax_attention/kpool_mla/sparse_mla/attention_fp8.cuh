#pragma once

#include "ops/softmax_attention/kpool_mla/sparse_mla/attention.cuh"
#include "ops/common/quant/fp8.cuh"

namespace ninfer::ops::detail::kpool {

template <int Tokens, int HeadGroups>
struct MlaFp8Storage {
    static constexpr int Chunk = Tokens == 64 ? 32 : Tokens;
    __align__(16) std::uint8_t codes[2 * (Chunk / 4) * 2560];
    __align__(16) std::uint8_t scales[2 * Chunk * 4];
    __nv_bfloat16 values[Chunk / 4 * 2304];

    union {
        float4 scores[Tokens / 8][32];
        __nv_bfloat16 probabilities[16 * Tokens];
    } tile[HeadGroups];

    float alpha[16 * HeadGroups];
    std::uint64_t full[2];
    // A 256-byte-aligned record plus the four-bit within-page pool ordinal.
    std::uint64_t pools[512];
};

// Two warps per 16 heads divide QK's G128 groups and PV's output features.
// FP8 codes are read once; their BF16 values stay on chip for PV.
template <int Tokens, int HeadGroups>
__global__
__launch_bounds__(64 * HeadGroups, HeadGroups == 2 ? 2 : 1) void mla_attention_fp8_kernel(
    const __grid_constant__ SparseMlaArguments a, const __grid_constant__ CUtensorMap map) {
    if (a.direct) return;
    constexpr int Parts = 2, Threads = 64 * HeadGroups;
    constexpr int Chunk = Tokens == 64 ? 32 : Tokens;
    const int local = blockIdx.x / (4 / HeadGroups), split = blockIdx.y;
    const int tid = threadIdx.x, lane = tid % 32, part = tid / 32 % 2;
    const int group = tid / 64, row = lane / 4, col = lane % 4;
    const int head_base  = blockIdx.x % (4 / HeadGroups) * (16 * HeadGroups);
    const int first_head = head_base + group * 16;
    const auto work      = a.work(local);
    if (!work.size) return;
    const int tiles = (work.size + Tokens - 1) / Tokens;
    // The 64-token route serves full column waves, so its plan is always unsplit.
    const int splits = Tokens == 64 ? 1 : min(a.splits, tiles);
    if (split >= splits) return;
    const int start = split * tiles / splits * Tokens;
    const int stop  = min(work.size, (split + 1) * tiles / splits * Tokens);
    const auto* table =
        a.tables + std::int64_t(a.table_rows[(a.begin + local) / a.tokens]) * a.table_stride;
    if (work.size == 1) {
        const int token    = work.end - 1;
        const auto record  = __ldg(table + token / 64);
        const auto* codes  = a.latent_fp8 + record + (token % 64) * 512;
        const auto* scales = a.latent_scales + record + (token % 64) * 4;
        for (int i = tid; i < 16 * HeadGroups * 256; i += Threads) {
            const int head = head_base + i / 256, d = i % 256 * 2;
            *reinterpret_cast<unsigned*>(a.mapped + (local * 64 + head) * 512 + d) =
                common::fp8_dequant_bf16x2(*reinterpret_cast<const std::uint16_t*>(codes + d),
                                           scales[d / 128]);
        }
        return;
    }
    extern __shared__ __align__(1024) unsigned char shared[];
    auto& s = *reinterpret_cast<MlaFp8Storage<Tokens, HeadGroups>*>(shared);
    if (tid == 0) {
        common::cta_mbarrier_init(&s.full[0], 1);
        common::cta_mbarrier_init(&s.full[1], 1);
        common::cta_mbarrier_fence_init();
    }
    for (int i = start / 4 + tid; i < min(work.pools, (stop + 3) / 4); i += Threads) {
        const int pool = work.pools == work.end / 4 ? i : a.ids[std::int64_t(work.row) * 512 + i];
        s.pools[i]     = (std::uint64_t(__ldg(table + pool / 16)) << 4) | unsigned(pool % 16);
    }
    __syncthreads();
    unsigned qf[8][4], qs[2];
    // The initially dead K/V staging region holds the complete Q8 head tile.
    // Coalesced vectors and LDSM avoid the scattered scalar fragment loads.
    auto* query_tile   = shared;
    auto* query_scales = shared + HeadGroups * 16 * 512;
    for (int i = tid; i < HeadGroups * 16 * 32; i += Threads) {
        const int h = i / 32, d = i % 32 * 16;
        common::cp_async<16>(query_tile + ((h * 512 + d) ^ ((h & 7) << 4)),
                             a.query_fp8 + (local * 64 + head_base + h) * 512 + d);
    }
    if (tid < HeadGroups * 4)
        common::cp_async<16>(query_scales + tid * 16,
                             a.query_scales + (local * 64 + head_base) * 4 + tid * 16);
    common::cp_commit();
    common::cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int h = group * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int d = part * 256 + k * 32 + (lane >> 4) * 16;
        common::ldmatrix_x4(qf[k][0], qf[k][1], qf[k][2], qf[k][3],
                            common::smem_addr(query_tile + ((h * 512 + d) ^ ((h & 7) << 4))));
    }
#pragma unroll
    for (int g = 0; g < 2; ++g)
        qs[g] = query_scales[(group * 16 + row + (col % 2) * 8) * 4 + part * 2 + g];
    __syncthreads();
    const auto code_offset = [](int token, int d) {
        const int logical = token / 4 * 2560 + d / 128 * 512 + (token % 4) * 128 + d % 128;
        return logical ^ (((logical / 128) & 7) * 16);
    };
    const auto value_offset = [](int token, int d) {
        const int logical = token / 4 * 2304 + d / 64 * 256 + (token % 4) * 64 + d % 64;
        return logical ^ (((logical / 64) & 7) * 8);
    };
    const auto pool_offsets = [&](int index) {
        if (index < work.pools) {
            const auto packed  = s.pools[index];
            const auto record  = packed >> 4;
            const auto ordinal = packed & 15;
            return make_ulonglong2(record + ordinal * 2048, record + ordinal * 16);
        }
        const int token   = work.end / 4 * 4;
        const auto record = __ldg(table + token / 64);
        return make_ulonglong2(record + (token % 64) * 512, record + (token % 64) * 4);
    };
    const auto stage = [&](int k0, int buffer) {
        const int pools = min(Chunk / 4, max(0, min(work.pools, stop / 4) - k0 / 4));
        if (tid == 0) common::cta_mbarrier_arrive_expect_tx(&s.full[buffer], pools * 2048);
        if (tid < pools) {
            const auto offsets = pool_offsets(k0 / 4 + tid);
            common::tma_load_3d(s.codes + (buffer * Chunk / 4 + tid) * 2560, &map, 0, 0,
                                int(offsets.x / 128), &s.full[buffer]);
        }
        for (int i = pools * 128 + tid; i < Chunk * 32; i += Threads) {
            const int t = i / 32, d = (i % 32) * 16;
            const auto offsets = k0 + t < stop ? pool_offsets((k0 + t) / 4) : make_ulonglong2(0, 0);
            common::cp_async_zfill<16, Cache::cg>(
                s.codes + buffer * (Chunk / 4) * 2560 + code_offset(t, d),
                a.latent_fp8 + offsets.x + (t % 4) * 512 + d, k0 + t < stop ? 16 : 0);
        }
        if (tid < Chunk / 4) {
            const int ordinal  = k0 + tid * 4;
            const auto offsets = ordinal < stop ? pool_offsets(ordinal / 4) : make_ulonglong2(0, 0);
            common::cp_async_zfill<16, Cache::cg>(s.scales + buffer * Chunk * 4 + tid * 16,
                                                  a.latent_scales + offsets.y,
                                                  min(4, max(0, stop - ordinal)) * 4);
        }
        common::cp_commit();
    };
    float acc[32][4]{};
    OnlineSoftmaxRow state[2];
    const float scale = a.scale * common::kLog2E;
    stage(start, 0);
    if constexpr (Tokens == 64) stage(start + Chunk, 1);
    for (int k0 = start, iteration = 0; k0 < stop; k0 += Tokens, ++iteration) {
        const int buffer = Tokens == 64 ? 0 : iteration % 2;
        common::cp_wait<0>();
        if (tid == 0) {
            if constexpr (Tokens == 64) {
                common::cta_mbarrier_wait(&s.full[0], iteration & 1);
                common::cta_mbarrier_wait(&s.full[1], iteration & 1);
            } else
                common::cta_mbarrier_wait(&s.full[buffer], (iteration / 2) & 1);
        }
        // Also retires preceding PV reads before overwriting the single V buffer.
        __syncthreads();
        if constexpr (Tokens != 64)
            if (k0 + Tokens < stop) stage(k0 + Tokens, 1 - buffer);
        const auto* codes        = s.codes + buffer * (Tokens / 4) * 2560;
        const auto* scales       = s.scales + buffer * Tokens * 4;
        const auto decode_values = [&](int chunk) {
            for (int i = tid; i < Chunk * 64; i += Threads) {
                const int t = i / 64, d = i % 64 * 8;
                const uint2 raw      = common::load_vec<uint2>(codes + code_offset(t + chunk, d));
                const unsigned scale = scales[(t + chunk) * 4 + d / 128];
                common::store_vec(s.values + value_offset(t, d),
                                  make_uint4(common::fp8_dequant_bf16x2(raw.x, scale),
                                             common::fp8_dequant_bf16x2(raw.x >> 16, scale),
                                             common::fp8_dequant_bf16x2(raw.y, scale),
                                             common::fp8_dequant_bf16x2(raw.y >> 16, scale)));
            }
        };
        if constexpr (Tokens != 64) {
            decode_values(0);
            __syncthreads();
        }
        float score[Tokens / 8][4]{};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
#pragma unroll
            for (int n = 0; n < Tokens / 8; ++n) {
                unsigned bf[2];
                common::ldmatrix_x2(
                    bf[0], bf[1],
                    common::smem_addr(codes +
                                      code_offset(n * 8 + (lane & 7),
                                                  part * 256 + k * 32 + ((lane >> 3) & 1) * 16)));
                common::mma_fp8_e4m3_scaled(score[n][0], score[n][1], score[n][2], score[n][3],
                                            qf[k][0], qf[k][1], qf[k][2], qf[k][3], bf[0], bf[1],
                                            qs[k / 4],
                                            scales[(n * 8 + row) * 4 + part * 2 + k / 4]);
            }
        }
        if (part != 0) {
#pragma unroll
            for (int n = 0; n < Tokens / 8; ++n)
                s.tile[group].scores[n][lane] =
                    make_float4(score[n][0], score[n][1], score[n][2], score[n][3]);
        }
        __syncthreads();
        if (part == 0) {
            float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
            for (int n = 0; n < Tokens / 8; ++n) {
                const float4 other = s.tile[group].scores[n][lane];
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
                    *reinterpret_cast<unsigned*>(s.tile[group].probabilities +
                                                 ((r * Tokens + d) ^ ((r & 7) << 3))) =
                        common::pack_bf16x2(p0, p1);
                }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                state[j].accumulate(alpha[j], sum[j]);
                if (col == 0) s.alpha[group * 16 + row + j * 8] = alpha[j];
            }
        }
        if constexpr (Tokens != 64) __syncthreads();
#pragma unroll
        for (int chunk = 0; chunk < Tokens / Chunk; ++chunk) {
            if constexpr (Tokens == 64) {
                // QK has consumed both K halves. After widening each half,
                // replace only that half with the next tile while its PV runs.
                // Reuse V after every preceding PV reader retires.
                if (chunk) __syncthreads();
                decode_values(chunk * Chunk);
                __syncthreads();
                if (k0 + Tokens < stop) stage(k0 + Tokens + chunk * Chunk, chunk);
            }
            const float alpha[2] = {s.alpha[group * 16 + row], s.alpha[group * 16 + row + 8]};
            if (chunk == 0 && __any_sync(0xffffffffu, alpha[0] != 1.F || alpha[1] != 1.F)) {
#pragma unroll
                for (int n = 0; n < 32; ++n)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[n][j] *= alpha[j / 2];
            }
            const unsigned pv_address = common::smem_addr(
                s.values + value_offset((lane & 7) + ((lane >> 3) & 1) * 8, part * 256));
#pragma unroll 1
            for (int k = 0; k < Chunk / 16; ++k) {
                unsigned pf[4];
                const int pr = (lane & 7) + ((lane >> 3) & 1) * 8,
                          pd = chunk * Chunk + k * 16 + (lane >> 4) * 8;
                common::ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                                    common::smem_addr(s.tile[group].probabilities +
                                                      ((pr * Tokens + pd) ^ ((pr & 7) << 3))));
#pragma unroll
                for (int n = 0; n < 32; ++n) {
                    unsigned vf[2];
                    common::ldmatrix_x2_t(vf[0], vf[1],
                                          (pv_address ^ ((n % 8) * 16) ^ ((n / 8 % 2) * 64)) +
                                              (n / 8) * 512 + k * 18432);
                    common::mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1],
                                     pf[2], pf[3], vf[0], vf[1]);
                }
            }
        }
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
                    s.alpha[group * 16 + row + j * 8] = __frcp_rn(state[j].sum);
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
        const float inv = splits == 1 ? s.alpha[group * 16 + row + j * 8] : 1.f;
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

} // namespace ninfer::ops::detail::kpool
