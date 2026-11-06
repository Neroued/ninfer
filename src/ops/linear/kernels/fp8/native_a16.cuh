#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/host/launch.h"
#include "ops/common/quant/fp8.cuh"
#include "ops/common/numeric/math.h"
#include "ops/common/numeric/convert.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/fp8/native_common.cuh"

#include <type_traits>

namespace ninfer::ops::detail {

template <class Operands>
__device__ __forceinline__ float native_fp8_scale(const Operands& p, int row, int k) {
    if constexpr (std::is_same_v<typename Operands::ScaleType, float>)
        return p.scales[static_cast<std::int64_t>(row / 128) * (p.k / 128) + k / 128];
    else
        return common::ue8m0_to_float(
            p.scales[static_cast<std::int64_t>(row) * (p.k / 32) + k / 32]);
}

template <class S>
inline constexpr bool native_fp8_pair_across_warps = [] {
    if constexpr (requires { S::kPairAcrossWarps; })
        return S::kPairAcrossWarps;
    else
        return false;
}();

template <class S, class Operands, class Output, class Epilogue, class Rows>
__device__ __forceinline__ void
native_fp8_a16_simt_body(Operands p, Output output, Epilogue epilogue, Rows rows, int row_tile,
                         int token_begin, float* pair_scratch = nullptr) {
    constexpr int BT = S::kBlockTokens, RPW = S::kRowsPerWarp, V = S::kValuesPerLane;
    constexpr int C           = S::kChains;
    constexpr bool paired     = Rows::kPaired;
    constexpr bool cross_warp = native_fp8_pair_across_warps<S>;
    static_assert(!cross_warp || (paired && S::kWarps % 2 == 0));
    static_assert(!paired || cross_warp || RPW % 2 == 0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int k         = S::kStaticK ? S::kStaticK : p.k;
    const int row_begin = cross_warp ? (row_tile * (S::kWarps / 2) + warp / 2) * RPW
                                     : (row_tile * S::kWarps + warp) * (RPW / (paired ? 2 : 1));
    float accum[RPW][BT][C]{};
#pragma unroll 1
    for (int offset = lane * V; offset < k; offset += 32 * V) {
        float x[BT][V];
#pragma unroll
        for (int t = 0; t < BT; ++t) {
            const bool valid = token_begin + t < p.tokens;
            const auto* source =
                p.x + static_cast<std::int64_t>(valid ? token_begin + t : 0) * k + offset;
            if constexpr (std::is_same_v<typename Operands::InputType, __nv_bfloat16>) {
#pragma unroll
                for (int base = 0; base < V; base += 8) {
                    const uint4 packed = valid ? common::load_vec<uint4>(source + base) : uint4{};
                    const unsigned words[]{packed.x, packed.y, packed.z, packed.w};
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 value        = common::bf16x2_bits_to_float2(words[pair]);
                        x[t][base + 2 * pair]     = value.x;
                        x[t][base + 2 * pair + 1] = value.y;
                    }
                }
            } else {
#pragma unroll
                for (int base = 0; base < V; base += 4) {
                    const float4 value = valid ? common::load_vec<float4>(source + base) : float4{};
                    x[t][base]         = value.x;
                    x[t][base + 1]     = value.y;
                    x[t][base + 2]     = value.z;
                    x[t][base + 3]     = value.w;
                }
            }
            if constexpr (requires { p.x_low; }) {
                const auto* low =
                    p.x_low + static_cast<std::int64_t>(valid ? token_begin + t : 0) * k + offset;
#pragma unroll
                for (int base = 0; base < V; base += 8) {
                    const uint4 packed = valid ? common::load_vec<uint4>(low + base) : uint4{};
                    const unsigned words[]{packed.x, packed.y, packed.z, packed.w};
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 value = common::bf16x2_bits_to_float2(words[pair]);
                        x[t][base + 2 * pair] += value.x;
                        x[t][base + 2 * pair + 1] += value.y;
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row =
                rows.weight_row(row_begin, cross_warp ? (warp & 1) * RPW + r : r, p.rows);
            if (row < p.rows) {
                const auto* ptr   = p.codes + static_cast<std::int64_t>(row) * k + offset;
                const float scale = native_fp8_scale(p, row, offset);
                const auto packed = common::load_vec<std::conditional_t<V == 8, uint2, uint4>>(ptr);
                const auto* pair  = reinterpret_cast<const std::uint16_t*>(&packed);
                float partial[BT][C]{};
#pragma unroll
                for (int v = 0; v < V; v += 2) {
                    const float2 code = common::fp8_code2_to_float2(pair[v / 2]);
                    const float w0    = S::kPostScale ? code.x : code.x * scale;
                    const float w1    = S::kPostScale ? code.y : code.y * scale;
#pragma unroll
                    for (int t = 0; t < BT; ++t) {
                        auto& target        = S::kPostScale ? partial[t] : accum[r][t];
                        target[v % C]       = fmaf(w0, x[t][v], target[v % C]);
                        target[(v + 1) % C] = fmaf(w1, x[t][v + 1], target[(v + 1) % C]);
                    }
                }
                if constexpr (S::kPostScale) {
#pragma unroll
                    for (int t = 0; t < BT; ++t) {
                        float sum = 0;
#pragma unroll
                        for (int c = 0; c < C; ++c) sum += partial[t][c];
                        accum[r][t][0] = fmaf(sum, scale, accum[r][t][0]);
                    }
                }
            }
        }
    }
    float values[RPW][BT];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
#pragma unroll
        for (int t = 0; t < BT; ++t) {
            float value = 0;
#pragma unroll
            for (int c = 0; c < C; ++c) value += accum[r][t][c];
            values[r][t] = common::warp_reduce_sum(value);
        }
    }
    if constexpr (cross_warp) {
        // The containing kernel lends W*RPW*BT floats; no other body scratch is retained.
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < RPW; ++r)
#pragma unroll
                for (int t = 0; t < BT; ++t) pair_scratch[(warp * RPW + r) * BT + t] = values[r][t];
        }
        __syncthreads();
        if (lane == 0 && !(warp & 1)) {
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                const int row = row_begin + r;
#pragma unroll
                for (int t = 0; t < BT; ++t)
                    if (row < p.rows / 2 && token_begin + t < p.tokens)
                        epilogue.apply_pair(output, row, token_begin + t,
                                            pair_scratch[(warp * RPW + r) * BT + t],
                                            pair_scratch[((warp + 1) * RPW + r) * BT + t]);
            }
        }
    } else if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RPW / (paired ? 2 : 1); ++r) {
            const int row = row_begin + r;
            if (row < p.rows / (paired ? 2 : 1)) {
                if constexpr (paired) {
#pragma unroll
                    for (int t = 0; t < BT; ++t)
                        if (token_begin + t < p.tokens)
                            epilogue.apply_pair(output, row, token_begin + t, values[r][t],
                                                values[r + RPW / 2][t]);
                } else {
                    linear_finish_row(output, epilogue, row, token_begin, values[r],
                                      min(BT, p.tokens - token_begin));
                }
            }
        }
    }
}

template <class S, class Operands, class Output, class Epilogue, class Rows>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void native_fp8_a16_simt_kernel(
    Operands p, Output output, Epilogue epilogue, Rows rows) {
    if constexpr (native_fp8_pair_across_warps<S>) {
        __shared__ float pairs[S::kWarps * S::kRowsPerWarp * S::kBlockTokens];
        native_fp8_a16_simt_body<S>(p, output, epilogue, rows, blockIdx.x,
                                    blockIdx.y * S::kBlockTokens, pairs);
    } else {
        native_fp8_a16_simt_body<S>(p, output, epilogue, rows, blockIdx.x,
                                    blockIdx.y * S::kBlockTokens);
    }
}

template <class S>
inline constexpr bool native_fp8_register_weights = [] {
    if constexpr (requires { S::kRegisterWeights; })
        return S::kRegisterWeights;
    else
        return false;
}();

template <class S, class Operands, class Epilogue>
struct NativeFp8A16Storage {
    static constexpr int kPlanes =
        std::is_same_v<typename Operands::InputType, float> || requires(Operands p) {
            p.x_low;
        } ? 2 : 1;
    static constexpr int kCodeBytes = S::kStages * S::kBlockRows * S::kBlockK;
    static constexpr int kScaleBytes =
        S::kStages * S::kBlockRows *
        (std::is_same_v<typename Operands::ScaleType, float> ? int(sizeof(float))
                                                             : S::kBlockK / 32);
    static constexpr int kActivationElements = S::kStages * S::kBlockTokens * S::kBlockK * kPlanes;
    static constexpr int kWeightElements =
        native_fp8_register_weights<S> ? 0 : S::kBlockRows * S::kBlockK;
    static constexpr int kStagingBytes =
        kCodeBytes + kScaleBytes + (kWeightElements + kActivationElements) * 2;
    static constexpr int kBytes = kStagingBytes > linear_epilogue_bytes<S, Epilogue>
                                      ? kStagingBytes
                                      : linear_epilogue_bytes<S, Epilogue>;
    static_assert(kBytes <= 99 * 1024, "native A16 tile exceeds shared-memory capacity");
};

template <class S, bool Full, class Operands, class Output, class Epilogue, class Rows>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void native_fp8_a16_mma_kernel(
    Operands p, Output output, Epilogue epilogue, Rows rows) {
    constexpr int BR = S::kBlockRows, BT = S::kBlockTokens, BK = S::kBlockK;
    constexpr bool fp32_scale       = std::is_same_v<typename Operands::ScaleType, float>;
    constexpr bool fp32_input       = std::is_same_v<typename Operands::InputType, float>;
    constexpr bool register_weights = native_fp8_register_weights<S>;
    static_assert(!fp32_scale || BK == 128);
    using Storage        = NativeFp8A16Storage<S, Operands, Epilogue>;
    constexpr int planes = Storage::kPlanes, ST = S::kStages;
    auto* raw     = common::shared_storage<Storage::kBytes>();
    auto* codes   = raw;
    auto* scales  = codes + Storage::kCodeBytes;
    auto* ws      = reinterpret_cast<__nv_bfloat16*>(scales + Storage::kScaleBytes);
    auto* all_x   = ws + Storage::kWeightElements;
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
            const int lr = i / (BK / 16), col = i % (BK / 16) * 16;
            const int row        = rows.weight_row(row_begin, lr, p.rows);
            const int shared_col = register_weights ? col ^ ((lr & (BK / 16 - 1)) * 16) : col;
            common::cp_async_zfill<16, Cache::cg>(
                codes + slot * BR * BK + lr * BK + shared_col,
                p.codes + static_cast<std::int64_t>(row < p.rows ? row : 0) * k + kt * BK + col,
                row < p.rows ? 16 : 0);
        }
        if constexpr (!fp32_scale) {
            if constexpr (BK >= 128) {
                for (int i = tid; i < BR * (BK / 128); i += S::kThreads) {
                    const int lr = i / (BK / 128), col = i % (BK / 128) * 4;
                    const int row = rows.weight_row(row_begin, lr, p.rows);
                    common::cp_async_zfill<4>(
                        scales + slot * BR * (BK / 32) + i * 4,
                        p.scales + static_cast<std::int64_t>(row < p.rows ? row : 0) * (k / 32) +
                            kt * (BK / 32) + col,
                        row < p.rows ? 4 : 0);
                }
            } else {
                for (int i = tid; i < BR * (BK / 32); i += S::kThreads) {
                    const int lr = i / (BK / 32), col = i % (BK / 32);
                    const int row = rows.weight_row(row_begin, lr, p.rows);
                    scales[slot * BR * (BK / 32) + i] =
                        row < p.rows ? p.scales[static_cast<std::int64_t>(row) * (k / 32) +
                                                kt * (BK / 32) + col]
                                     : 127;
                }
            }
        } else {
            for (int row = tid; row < BR; row += S::kThreads) {
                const int source_row = rows.weight_row(row_begin, row, p.rows);
                const bool valid     = source_row < p.rows;
                common::cp_async_zfill<4>(
                    scales + (slot * BR + row) * sizeof(float),
                    p.scales +
                        (valid ? static_cast<std::int64_t>(source_row / 128) * (k / 128) + kt : 0),
                    valid ? 4 : 0);
            }
        }
        auto* xs = all_x + slot * BT * BK * planes;
        for (int item = tid; item < BT * BK / 8; item += S::kThreads) {
            const int lt = item / (BK / 8), col = item % (BK / 8) * 8, token = token_begin + lt;
            if constexpr (!fp32_input) {
                common::cp_async_zfill<16>(
                    xs + lt * BK + common::swizzle_xor_64(lt, col),
                    p.x + static_cast<std::int64_t>(token < p.tokens ? token : 0) * k + kt * BK +
                        col,
                    Full || token < p.tokens ? 16 : 0);
                if constexpr (requires { p.x_low; })
                    common::cp_async_zfill<16>(
                        xs + BT * BK + lt * BK + common::swizzle_xor_64(lt, col),
                        p.x_low + static_cast<std::int64_t>(token < p.tokens ? token : 0) * k +
                            kt * BK + col,
                        Full || token < p.tokens ? 16 : 0);
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const float value =
                        Full || token < p.tokens
                            ? p.x[static_cast<std::int64_t>(token) * k + kt * BK + col + v]
                            : 0;
                    const auto high                                   = __float2bfloat16_rn(value);
                    xs[lt * BK + common::swizzle_xor_64(lt, col + v)] = high;
                    xs[BT * BK + lt * BK + common::swizzle_xor_64(lt, col + v)] =
                        __float2bfloat16_rn(value - __bfloat162float(high));
                }
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
        const auto* current_codes  = codes + slot * BR * BK;
        const auto* current_scales = scales + slot * (Storage::kScaleBytes / ST);
        auto* xs                   = all_x + slot * BT * BK * planes;
        if constexpr (!register_weights) {
            for (int item = tid; item < BR * BK / 8; item += S::kThreads) {
                const int lr = item / (BK / 8), col = item % (BK / 8) * 8;
                const uint2 packed = common::load_vec<uint2>(current_codes + lr * BK + col);
                const auto* pairs  = reinterpret_cast<const std::uint16_t*>(&packed);
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const auto value = [&] {
                        if constexpr (fp32_scale)
                            return common::fp8_e4m3x2_to_bf16x2_bits(pairs[pair]);
                        else
                            return common::fp8_dequant_bf16x2(
                                pairs[pair], current_scales[lr * (BK / 32) + col / 32]);
                    }();
                    common::store_vec(ws + lr * BK + common::swizzle_xor_64(lr, col + pair * 2),
                                      value);
                }
            }
            __syncthreads();
        }
        constexpr int scale_words = (BK + 127) / 128;
        unsigned packed_scales[S::kMmaRows][2][scale_words]{};
        if constexpr (register_weights && !fp32_scale) {
#pragma unroll
            for (int r = 0; r < S::kMmaRows; ++r) {
                const int lr = wr * S::kWarpRows + r * 16 + (lane >> 2);
#pragma unroll
                for (int side = 0; side < 2; ++side) {
                    const auto* source = current_scales + (lr + side * 8) * (BK / 32);
#pragma unroll
                    for (int word = 0; word < scale_words; ++word) {
                        if constexpr (BK >= 128)
                            packed_scales[r][side][word] =
                                common::load_vec<unsigned>(source + word * 4);
                        else
                            packed_scales[r][side][word] = common::load_vec<std::uint16_t>(source);
                    }
                }
            }
        }
        float partial[S::kMmaRows][S::kMmaTokens][4]{};
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            unsigned af[S::kMmaRows][4];
#pragma unroll
            for (int r = 0; r < S::kMmaRows; ++r) {
                if constexpr (register_weights) {
                    const int lr    = wr * S::kWarpRows + r * 16 + (lane >> 2);
                    const int col   = kk + (lane & 3) * 2;
                    const auto pair = [&](int row, int column, unsigned scale) {
                        const int byte = column ^ ((row & (BK / 16 - 1)) * 16);
                        const auto bits =
                            common::load_vec<std::uint16_t>(current_codes + row * BK + byte);
                        if constexpr (fp32_scale)
                            return common::fp8_e4m3x2_to_bf16x2_bits(bits);
                        else
                            return common::fp8_dequant_bf16x2(bits,
                                                              static_cast<std::uint8_t>(scale));
                    };
                    const unsigned first = packed_scales[r][0][kk / 128] >> (((kk & 127) / 32) * 8);
                    const unsigned second =
                        packed_scales[r][1][kk / 128] >> (((kk & 127) / 32) * 8);
                    af[r][0] = pair(lr, col, first);
                    af[r][1] = pair(lr + 8, col, second);
                    af[r][2] = pair(lr, col + 8, first);
                    af[r][3] = pair(lr + 8, col + 8, second);
                } else {
                    const int lr  = wr * S::kWarpRows + r * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int col = kk + (lane >> 4) * 8;
                    common::ldmatrix_x4(
                        af[r][0], af[r][1], af[r][2], af[r][3],
                        common::smem_addr(ws + lr * BK + common::swizzle_xor_64(lr, col)));
                }
            }
#pragma unroll
            for (int plane = 0; plane < planes; ++plane) {
                unsigned bf[S::kMmaTokens][2];
#pragma unroll
                for (int t = 0; t < S::kMmaTokens; ++t) {
                    const int lt  = wt * S::kWarpTokens + t * 8 + (lane & 7);
                    const int col = kk + ((lane >> 3) & 1) * 8;
                    common::ldmatrix_x2(bf[t][0], bf[t][1],
                                        common::smem_addr(xs + plane * BT * BK + lt * BK +
                                                          common::swizzle_xor_64(lt, col)));
                }
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
#pragma unroll
                    for (int t = 0; t < S::kMmaTokens; ++t) {
                        auto& a = fp32_scale ? partial[r][t] : acc[r][t];
                        common::mma_bf16(a[0], a[1], a[2], a[3], af[r][0], af[r][1], af[r][2],
                                         af[r][3], bf[t][0], bf[t][1]);
                    }
                }
            }
        }
        if constexpr (fp32_scale) {
#pragma unroll
            for (int r = 0; r < S::kMmaRows; ++r) {
                const int lr           = wr * S::kWarpRows + r * 16 + (lane >> 2);
                const auto* row_scales = reinterpret_cast<const float*>(current_scales);
                const float s0 = row_scales[lr], s1 = row_scales[lr + 8];
#pragma unroll
                for (int t = 0; t < S::kMmaTokens; ++t) {
                    acc[r][t][0] = fmaf(partial[r][t][0], s0, acc[r][t][0]);
                    acc[r][t][1] = fmaf(partial[r][t][1], s0, acc[r][t][1]);
                    acc[r][t][2] = fmaf(partial[r][t][2], s1, acc[r][t][2]);
                    acc[r][t][3] = fmaf(partial[r][t][3], s1, acc[r][t][3]);
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

template <class S, class Operands, class Output, class Epilogue, class Rows = Fp8IdentityRows>
void launch_native_fp8_a16_simt(const Operands& p, Output output, Epilogue epilogue,
                                cudaStream_t stream, Rows rows = {}) {
    validate_native_fp8<S>(p);
    const dim3 grid(common::div_up(p.rows, S::kBlockRows),
                    common::div_up(p.tokens, S::kBlockTokens));
    native_fp8_a16_simt_kernel<S><<<grid, S::kThreads, 0, stream>>>(p, output, epilogue, rows);
    CUDA_CHECK(cudaGetLastError());
}

template <class S, class Operands, class Output, class Epilogue, class Rows = Fp8IdentityRows>
void launch_native_fp8_a16_mma(const Operands& p, Output output, Epilogue epilogue,
                               cudaStream_t stream, Rows rows = {}) {
    validate_native_fp8<S>(p);
    if (p.k % S::kBlockK || p.k / S::kBlockK < S::kStages)
        throw std::invalid_argument("native A16 MMA requires complete K stages");
    const int grid =
        common::div_up(p.rows, S::kBlockRows) * common::div_up(p.tokens, S::kBlockTokens);
    const auto launch = [&]<bool Full>() {
        constexpr int bytes = NativeFp8A16Storage<S, Operands, Epilogue>::kBytes;
        constexpr auto kernel =
            native_fp8_a16_mma_kernel<S, Full, Operands, Output, Epilogue, Rows>;
        const auto shared = common::prepare_shared<bytes, kernel>();
        kernel<<<grid, S::kThreads, shared, stream>>>(p, output, epilogue, rows);
        CUDA_CHECK(cudaGetLastError());
    };
    if (p.tokens % S::kBlockTokens == 0 && p.rows % S::kBlockRows == 0)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}

} // namespace ninfer::ops::detail
