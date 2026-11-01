#pragma once
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_instances.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"
#include <mma.h>

namespace ninfer::ops::detail::swiglu_ffn {
template <int Tokens, class Output>
__global__ __launch_bounds__(128) void down_vector(const float* x, const __nv_bfloat16* w,
                                                   Output out) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
    float sum[Tokens] = {};
#pragma unroll 1
    for (int k = lane * 4; k < 1024; k += 128) {
        const auto raw = common::load_vec<uint2>(w + std::int64_t(row) * 1024 + k);
        const auto a   = common::bf16x2_bits_to_float2(raw.x),
                   b   = common::bf16x2_bits_to_float2(raw.y);
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            const float4 v = common::load_vec<float4>(x + std::int64_t(t) * 1024 + k);
            sum[t]         = fmaf(v.x, a.x, sum[t]);
            sum[t]         = fmaf(v.y, a.y, sum[t]);
            sum[t]         = fmaf(v.z, b.x, sum[t]);
            sum[t]         = fmaf(v.w, b.y, sum[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        const float v = common::warp_reduce_sum(sum[t]);
        if (!lane) out.store(row, t, v);
    }
}

// The small-call FP32 product is rounded once to TF32; BF16 weights are exact TF32 operands.
template <int Rows, int Tokens, int BK, int Stages, class Output>
__global__ __launch_bounds__((Rows / 16) * (Tokens / 8) *
                             32) void tf32_down_kernel(const float* x, const __nv_bfloat16* weights,
                                                       Output out, int tokens) {
    constexpr int Threads = (Rows / 16) * (Tokens / 8) * 32;

    struct Shared {
        __nv_bfloat16 w[Stages][Rows * BK];
        float x[Stages][Tokens * BK];
    };

    extern __shared__ __align__(16) unsigned char storage[];
    auto& s       = *reinterpret_cast<Shared*>(storage);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int row0 = blockIdx.x * Rows, token0 = blockIdx.y * Tokens;
    auto wi    = [](int row, int k) { return row * BK + (k ^ ((row & 7) * 8)); };
    auto xi    = [](int token, int k) { return token * BK + (k ^ ((token & 7) * 4)); };
    auto stage = [&](int slot, int kt) {
        for (int v = tid; v < Rows * (BK / 8); v += Threads) {
            int r = v / (BK / 8), k = (v % (BK / 8)) * 8;
            common::cp_async<16, ops::Cache::cg>(
                s.w[slot] + wi(r, k), weights + std::int64_t(row0 + r) * 1024 + kt * BK + k);
        }
        for (int v = tid; v < Tokens * (BK / 4); v += Threads) {
            int t = v / (BK / 4), k = (v % (BK / 4)) * 4;
            bool valid        = token0 + t < tokens;
            const auto offset = valid ? std::int64_t(token0 + t) * 1024 + kt * BK + k : 0;
            common::cp_async_zfill<16, ops::Cache::cg>(s.x[slot] + xi(t, k), x + offset,
                                                       valid ? 16 : 0);
        }
        common::cp_commit();
    };
    for (int i = 0; i < Stages; ++i) stage(i, i);
    float a[4]   = {};
    const int wr = warp / (Tokens / 8), wt = warp % (Tokens / 8), g = lane / 4, t = lane & 3;
    for (int kt = 0; kt < 1024 / BK; ++kt) {
        int slot = kt % Stages;
        if (kt + Stages <= 1024 / BK)
            common::cp_wait<Stages - 1>();
        else
            common::cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; k += 8) {
            unsigned af[4];
            af[0] = unsigned(__bfloat16_as_ushort(s.w[slot][wi(wr * 16 + g, k + t)])) << 16;
            af[1] = unsigned(__bfloat16_as_ushort(s.w[slot][wi(wr * 16 + g + 8, k + t)])) << 16;
            af[2] = unsigned(__bfloat16_as_ushort(s.w[slot][wi(wr * 16 + g, k + t + 4)])) << 16;
            af[3] = unsigned(__bfloat16_as_ushort(s.w[slot][wi(wr * 16 + g + 8, k + t + 4)])) << 16;
            const float b0 = s.x[slot][xi(wt * 8 + g, k + t)],
                        b1 = s.x[slot][xi(wt * 8 + g, k + t + 4)];
            unsigned bf0, bf1;
            bf0 = __float_as_uint(nvcuda::wmma::__float_to_tf32(b0));
            bf1 = __float_as_uint(nvcuda::wmma::__float_to_tf32(b1));
            common::mma_tf32_bits(a[0], a[1], a[2], a[3], af[0], af[1], af[2], af[3], bf0, bf1);
        }
        __syncthreads();
        if (kt + Stages < 1024 / BK) stage(slot, kt + Stages);
    }
    const int row = row0 + wr * 16 + g, token = token0 + wt * 8 + 2 * t;
    if (token < tokens) {
        out.store(row, token, a[0]);
        out.store(row + 8, token, a[2]);
    }
    if (token + 1 < tokens) {
        out.store(row, token + 1, a[1]);
        out.store(row + 8, token + 1, a[3]);
    }
}

template <int Rows, class Output>
void tf32_down(const float* x, const Weight& w, Output out, int tokens, cudaStream_t stream) {
    constexpr int T = 16, BK = 128, S = 2, Threads = (Rows / 16) * 2 * 32;
    constexpr int Bytes  = S * (Rows * BK * 2 + T * BK * 4);
    constexpr auto entry = tf32_down_kernel<Rows, T, BK, S, Output>;
    const int bytes      = common::prepare_shared<Bytes, entry, true>();
    entry<<<dim3(4096 / Rows, (tokens + T - 1) / T), Threads, bytes, stream>>>(
        x, (const __nv_bfloat16*)w.qdata, out, tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool FullTokens, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void compensated_down_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight, Output output,
    Epilogue epilogue, int rows, int input_rows, int token_offset, int count) {
    const int M           = rows;
    const int K           = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int tokens      = token_offset + count;
    const int split_k     = K;
    const int split_begin = 0;
    const int split_end   = split_begin + split_k;
    constexpr int BM      = Schedule::kBlockRows;
    constexpr int BN      = Schedule::kBlockTokens;
    constexpr int BK      = Schedule::kBlockK;
    constexpr int MT      = Schedule::kMmaRows;
    constexpr int NT      = Schedule::kMmaTokens;
    constexpr int S       = Schedule::kStages;
    constexpr int THREADS = Schedule::kThreads;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* As = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* Bs = As + S * BM * BK;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int tiles_m = bf16_predicated_rows<Schedule> ? (M + BM - 1) / BM : M / BM;
    const int tiles_n = count / BN + static_cast<int>(count % BN != 0);
    int tile_m        = 0;
    int tile_n        = 0;
    bf16_mma_tile_coordinates<Schedule>(static_cast<int>(blockIdx.x), tiles_m, tiles_n, tile_m,
                                        tile_n);
    const int m0 = tile_m * BM;
    const int n0 = token_offset + tile_n * BN;

    float accum[2][MT][NT][4] = {};

    auto stage_inputs = [&](int stage, int k_tile) {
        const int k0  = split_begin + k_tile * BK;
        auto* a_stage = As + stage * BM * BK;
        auto* b_stage = Bs + stage * BN * BK;

#pragma unroll 1
        for (int item = tid; item < BM * (BK / 8); item += THREADS) {
            const int row = item / (BK / 8);
            const int k8  = item - row * (BK / 8);
            const int kk  = k8 * 8;
            auto* dst     = &a_stage[row * BK + bf16_mma_shared_col<Schedule>(row, kk)];
            if constexpr (bf16_predicated_rows<Schedule> || bf16_predicated_k<Schedule>) {
                const bool valid = (!bf16_predicated_rows<Schedule> || m0 + row < M) &&
                                   (!bf16_predicated_k<Schedule> || k0 + kk < split_end);
                const auto source = valid ? static_cast<std::int64_t>(m0 + row) * K + k0 + kk : 0;
                common::cp_async_zfill<16, Schedule::kWeightCache>(dst, weight + source,
                                                                   valid ? 16 : 0);
            } else {
                common::cp_async<16, Schedule::kWeightCache>(
                    dst, &weight[static_cast<std::int64_t>(m0 + row) * K + k0 + kk]);
            }
        }

#pragma unroll
        for (int plane = 0; plane < 2; ++plane) {
            auto* plane_stage = b_stage + plane * S * BN * BK;
#pragma unroll 1
            for (int item = tid; item < BN * (BK / 8); item += THREADS) {
                const int col   = item / (BK / 8);
                const int k8    = item - col * (BK / 8);
                const int kk    = k8 * 8;
                auto* dst       = &plane_stage[col * BK + bf16_mma_shared_col<Schedule>(col, kk)];
                const int token = n0 + col;
                if constexpr (FullTokens && !bf16_predicated_k<Schedule>) {
                    common::cp_async<16, Schedule::kActivationCache>(
                        dst, &x[static_cast<std::int64_t>(plane) * K * tokens +
                                static_cast<std::int64_t>(token) * K + k0 + kk]);
                } else {
                    const bool valid = (FullTokens || token < tokens) &&
                                       (!bf16_predicated_k<Schedule> || k0 + kk < split_end);
                    const auto source = valid ? static_cast<std::int64_t>(token) * K + k0 + kk : 0;
                    common::cp_async_zfill<16, Schedule::kActivationCache>(
                        dst, x + static_cast<std::int64_t>(plane) * K * tokens + source,
                        valid ? 16 : 0);
                }
            }
        }
    };

    const int kTiles = split_k / BK + (bf16_predicated_k<Schedule> && split_k % BK != 0);
#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        if (stage < kTiles) {
            stage_inputs(stage, stage);
            common::cp_commit();
        }
    }

#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        const int stage = k_tile % S;
        if (k_tile + S <= kTiles) {
            common::cp_wait<S - 1>();
        } else {
            // Once the producer stops refilling the ring, fewer than S groups remain. Drain them
            // instead of applying the steady-state wait count to the final consumer stage.
            common::cp_wait<0>();
        }
        __syncthreads();

#pragma unroll
        for (int plane = 0; plane < 2; ++plane)
            bf16_mma_compute_stage<Schedule>(
                As + stage * BM * BK, Bs + (plane * S + stage) * BN * BK, accum[plane], warp, lane);

        __syncthreads();
        const int next = k_tile + S;
        if (next < kTiles) {
            stage_inputs(stage, next);
            common::cp_commit();
        }
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
#pragma unroll
            for (int j = 0; j < 4; ++j) accum[0][mi][ni][j] += accum[1][mi][ni][j];
    bf16_finish_mma_tile<Schedule, FullTokens>(output, epilogue, shared_raw, accum[0], m0, n0, M,
                                               tokens, warp, lane);
}

template <class Schedule, class Output>
void compensated_down(const __nv_bfloat16* planes, const Weight& w, int tokens, Output output,
                      cudaStream_t stream) {
    using S             = Bf16ScheduleInstance<Schedule, 1024>;
    constexpr int bytes = S::kStages * (S::kBlockRows + 2 * S::kBlockTokens) * S::kBlockK * 2;
    const auto launch   = [&]<bool Full>() {
        constexpr auto kernel = compensated_down_kernel<S, Full, Output, LinearIdentityEpilogue>;
        const int shared      = common::prepare_shared<bytes, kernel, true>();
        const int tiles =
            (4096 / S::kBlockRows) * ((tokens + S::kBlockTokens - 1) / S::kBlockTokens);
        kernel<<<tiles, S::kThreads, shared, stream>>>(
            planes, static_cast<const __nv_bfloat16*>(w.qdata), output, {}, 4096, 1024, 0, tokens);
        CUDA_CHECK(cudaGetLastError());
    };
    if (tokens % S::kBlockTokens == 0)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}

} // namespace ninfer::ops::detail::swiglu_ffn
