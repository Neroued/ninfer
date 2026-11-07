#pragma once

#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/quant/fp8.cuh"
#include "ops/linear/kernels/fp8/native_common.cuh"
#include "ops/linear/kernels/mxfp8/a8_plan.h"
#include <type_traits>

namespace ninfer::ops::detail {

// The shared quantizer has one CUDA registration owner across all consumers.
template <class Input, bool PadCodes = false>
void launch_mxfp8_a8_quantize(const Input* x, int k, int tokens, Mxfp8A8Workspace out,
                              cudaStream_t stream, int code_tokens = 0);

extern template void launch_mxfp8_a8_quantize<__nv_bfloat16, false>(const __nv_bfloat16*, int, int,
                                                                    Mxfp8A8Workspace, cudaStream_t,
                                                                    int);
extern template void launch_mxfp8_a8_quantize<__nv_bfloat16, true>(const __nv_bfloat16*, int, int,
                                                                   Mxfp8A8Workspace, cudaStream_t,
                                                                   int);
extern template void launch_mxfp8_a8_quantize<float, false>(const float*, int, int,
                                                            Mxfp8A8Workspace, cudaStream_t, int);
extern template void launch_mxfp8_a8_quantize<float, true>(const float*, int, int, Mxfp8A8Workspace,
                                                           cudaStream_t, int);

// Both code and scale planes cover padded_tokens. Padding represents exact zero;
// only the first tokens rows are read from the input.
template <class Input>
void launch_mxfp8_a8_quantize_padded(const Input* x, int k, int tokens, Mxfp8A8Workspace out,
                                     cudaStream_t stream, int padded_tokens);
extern template void launch_mxfp8_a8_quantize_padded<__nv_bfloat16>(const __nv_bfloat16*, int, int,
                                                                    Mxfp8A8Workspace, cudaStream_t,
                                                                    int);
extern template void launch_mxfp8_a8_quantize_padded<float>(const float*, int, int,
                                                            Mxfp8A8Workspace, cudaStream_t, int);

template <class S>
__device__ __forceinline__ int mxfp8_shared_byte(int row, int col) {
    return col ^ ((row & (S::kBlockK / 16 - 1)) * 16);
}

template <class S, class Epilogue>
struct Mxfp8A8Storage {
    static constexpr int kCodeBytes = S::kStages * (S::kBlockRows + S::kBlockTokens) * S::kBlockK;
    static constexpr int kScaleBytes =
        S::kStages * (S::kBlockRows + S::kBlockTokens) * (S::kBlockK / 32);
    static constexpr int kStagingBytes = kCodeBytes + kScaleBytes;
    static constexpr int kBytes        = kStagingBytes > linear_epilogue_bytes<S, Epilogue>
                                             ? kStagingBytes
                                             : linear_epilogue_bytes<S, Epilogue>;
};

template <class S, bool Full, class Output, class Epilogue, class Rows>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void mxfp8_a8_mma_kernel(
    Mxfp8A8Operands p, Output output, Epilogue epilogue, Rows rows) {
    constexpr int BR = S::kBlockRows, BT = S::kBlockTokens, BK = S::kBlockK, ST = S::kStages;
    constexpr int groups = BK / 32;
    using Storage        = Mxfp8A8Storage<S, Epilogue>;
    auto* raw            = common::shared_storage<Storage::kBytes>();
    auto* ws             = raw;
    auto* xs             = ws + ST * BR * BK;
    auto* sw             = raw + Storage::kCodeBytes;
    auto* sx             = sw + ST * BR * groups;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wr = warp / S::kWarpsTokens, wt = warp % S::kWarpsTokens;
    int row_tile, token_tile;
    native_fp8_tile<S>(blockIdx.x, p.rows, p.tokens, row_tile, token_tile);
    const int row_begin   = row_tile * (BR / (Rows::kPaired ? 2 : 1));
    const int token_begin = token_tile * BT;
    if constexpr (requires { output.clear_if_empty(0, 0, 0, 0); })
        if (output.clear_if_empty(row_begin, token_begin, min(BR, p.rows - row_begin),
                                  min(BT, p.tokens - token_begin)))
            return;
    const int k      = S::kStaticK ? S::kStaticK : p.k;
    const int tiles  = k / BK;
    const auto stage = [&](int slot, int kt) {
        for (int i = tid; i < BR * (BK / 16); i += S::kThreads) {
            const int r = i / (BK / 16), col = i % (BK / 16) * 16;
            const int row = rows.weight_row(row_begin, r, p.rows);
            common::cp_async_zfill<16, Cache::cg>(
                ws + slot * BR * BK + r * BK + mxfp8_shared_byte<S>(r, col),
                p.codes + static_cast<std::int64_t>(row < p.rows ? row : 0) * k + kt * BK + col,
                row < p.rows ? 16 : 0);
        }
        for (int i = tid; i < BT * (BK / 16); i += S::kThreads) {
            const int t = i / (BK / 16), col = i % (BK / 16) * 16;
            const int token = token_begin + t;
            common::cp_async_zfill<16, Cache::cg>(
                xs + slot * BT * BK + t * BK + mxfp8_shared_byte<S>(t, col),
                p.x + static_cast<std::int64_t>(token < p.tokens ? token : 0) * k + kt * BK + col,
                Full || token < p.tokens ? 16 : 0);
        }
        if constexpr (groups >= 4) {
            for (int i = tid; i < BR * (groups / 4); i += S::kThreads) {
                const int r = i / (groups / 4), g = i % (groups / 4) * 4;
                const int row = rows.weight_row(row_begin, r, p.rows);
                common::cp_async_zfill<4>(
                    sw + slot * BR * groups + i * 4,
                    p.scales + static_cast<std::int64_t>(row < p.rows ? row : 0) * (k / 32) +
                        kt * groups + g,
                    row < p.rows ? 4 : 0);
            }
            for (int i = tid; i < BT * (groups / 4); i += S::kThreads) {
                const int t = i / (groups / 4), g = i % (groups / 4) * 4, token = token_begin + t;
                common::cp_async_zfill<4>(
                    sx + slot * BT * groups + i * 4,
                    p.x_scales +
                        static_cast<std::int64_t>(token < p.tokens ? token : 0) * (k / 32) +
                        kt * groups + g,
                    Full || token < p.tokens ? 4 : 0);
            }
        } else {
            for (int i = tid; i < BR * groups; i += S::kThreads) {
                const int r = i / groups, g = i % groups,
                          row = rows.weight_row(row_begin, r, p.rows);
                sw[slot * BR * groups + i] =
                    row < p.rows
                        ? p.scales[static_cast<std::int64_t>(row) * (k / 32) + kt * groups + g]
                        : 127;
            }
            for (int i = tid; i < BT * groups; i += S::kThreads) {
                const int t = i / groups, g = i % groups, token = token_begin + t;
                sx[slot * BT * groups + i] =
                    Full || token < p.tokens
                        ? p.x_scales[static_cast<std::int64_t>(token) * (k / 32) + kt * groups + g]
                        : 127;
            }
        }
    };
#pragma unroll
    for (int slot = 0; slot < ST; ++slot) {
        stage(slot, slot);
        common::cp_commit();
    }
    float acc[S::kMmaRows][S::kMmaTokens][4]{};
#pragma unroll 1
    for (int kt = 0; kt < tiles; ++kt) {
        const int slot = kt % ST;
        if (kt + ST <= tiles)
            common::cp_wait<ST - 1>();
        else
            common::cp_wait<0>();
        __syncthreads();
        constexpr int scale_words = (groups + 3) / 4;
        unsigned weight_scales[S::kMmaRows][scale_words];
        unsigned input_scales[S::kMmaTokens][scale_words];
#pragma unroll
        for (int r = 0; r < S::kMmaRows; ++r) {
            const int row = wr * S::kWarpRows + r * 16 + (lane >> 2) + ((lane & 3) == 1 ? 8 : 0);
#pragma unroll
            for (int word = 0; word < scale_words; ++word) {
                if constexpr (groups >= 4)
                    weight_scales[r][word] = common::load_vec<unsigned>(sw + slot * BR * groups +
                                                                        row * groups + word * 4);
                else
                    weight_scales[r][word] =
                        common::load_vec<std::uint16_t>(sw + slot * BR * groups + row * groups);
            }
        }
#pragma unroll
        for (int t = 0; t < S::kMmaTokens; ++t) {
            const int token = wt * S::kWarpTokens + t * 8 + (lane >> 2);
#pragma unroll
            for (int word = 0; word < scale_words; ++word) {
                if constexpr (groups >= 4)
                    input_scales[t][word] = common::load_vec<unsigned>(sx + slot * BT * groups +
                                                                       token * groups + word * 4);
                else
                    input_scales[t][word] =
                        common::load_vec<std::uint16_t>(sx + slot * BT * groups + token * groups);
            }
        }
        unsigned af[2][S::kMmaRows][4], bf[2][S::kMmaTokens][2];
        const auto load = [&](int frag, int step) {
#pragma unroll
            for (int r = 0; r < S::kMmaRows; ++r) {
                const int row = wr * S::kWarpRows + r * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int col = step * 32 + (lane >> 4) * 16;
                common::ldmatrix_x4(af[frag][r][0], af[frag][r][1], af[frag][r][2], af[frag][r][3],
                                    common::smem_addr(ws + slot * BR * BK + row * BK +
                                                      mxfp8_shared_byte<S>(row, col)));
            }
#pragma unroll
            for (int t = 0; t < S::kMmaTokens; ++t) {
                const int token = wt * S::kWarpTokens + t * 8 + (lane & 7),
                          col   = step * 32 + ((lane >> 3) & 1) * 16;
                common::ldmatrix_x2(bf[frag][t][0], bf[frag][t][1],
                                    common::smem_addr(xs + slot * BT * BK + token * BK +
                                                      mxfp8_shared_byte<S>(token, col)));
            }
        };
        load(0, 0);
#pragma unroll
        for (int step = 0; step < groups; ++step) {
            const int frag = step & 1;
            if (step + 1 < groups) load(frag ^ 1, step + 1);
#pragma unroll
            for (int r = 0; r < S::kMmaRows; ++r) {
                const unsigned scale_a = (weight_scales[r][step / 4] >> ((step % 4) * 8)) & 255u;
#pragma unroll
                for (int t = 0; t < S::kMmaTokens; ++t) {
                    const unsigned scale_b = (input_scales[t][step / 4] >> ((step % 4) * 8)) & 255u;
                    auto& a                = acc[r][t];
                    common::mma_fp8_e4m3_scaled(a[0], a[1], a[2], a[3], af[frag][r][0],
                                                af[frag][r][1], af[frag][r][2], af[frag][r][3],
                                                bf[frag][t][0], bf[frag][t][1], scale_a, scale_b);
                }
            }
        }
        __syncthreads();
        if (kt + ST < tiles) {
            stage(slot, kt + ST);
            common::cp_commit();
        }
    }
    native_fp8_finish<S, Full>(output, epilogue, raw, acc, row_begin, token_begin, p.rows,
                               p.tokens);
}

template <class S, class Output, class Epilogue, class Rows = Fp8IdentityRows>
void launch_mxfp8_a8_mma(const Mxfp8A8Operands& p, Output output, Epilogue epilogue,
                         cudaStream_t stream, Rows rows = {}) {
    validate_native_fp8<S>(p);
    if (!p.x_scales || p.k % S::kBlockK || p.k / S::kBlockK < S::kStages)
        throw std::invalid_argument("MXFP8 MMA requires complete K tiles and activation scales");
    const int grid =
        common::div_up(p.rows, S::kBlockRows) * common::div_up(p.tokens, S::kBlockTokens);
    const auto launch = [&]<bool Full>() {
        constexpr int bytes   = Mxfp8A8Storage<S, Epilogue>::kBytes;
        constexpr auto kernel = mxfp8_a8_mma_kernel<S, Full, Output, Epilogue, Rows>;
        const auto shared     = common::prepare_shared<bytes, kernel>();
        kernel<<<grid, S::kThreads, shared, stream>>>(p, output, epilogue, rows);
        CUDA_CHECK(cudaGetLastError());
    };
    if (p.tokens % S::kBlockTokens == 0 && p.rows % S::kBlockRows == 0)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}

} // namespace ninfer::ops::detail
