#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/host/launch.h"
#include "ops/common/numeric/math.h"
#include "ops/common/quant/fp8.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/fp8/native_schedule.cuh"

#include <type_traits>
#include <utility>

namespace ninfer::ops::detail {

// Each warp owns whole K128 scale groups; only the final FP32 reduction crosses warps.
template <int BlockRows, int BlockTokens, int KWarps, int WarpK = 128, int Stages = 2,
          int MinBlocksPerSm = 1>
struct NativeFp8A16SlicedKSchedule {
    static constexpr int kStaticK        = 0;
    static constexpr int kBlockRows      = BlockRows;
    static constexpr int kBlockTokens    = BlockTokens;
    static constexpr int kKWarps         = KWarps;
    static constexpr int kWarpK          = WarpK;
    static constexpr int kBlockK         = KWarps * WarpK;
    static constexpr int kStages         = Stages;
    static constexpr int kThreads        = KWarps * 32;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kMmaRows        = (BlockRows + 15) / 16;
    static constexpr int kMmaTokens      = (BlockTokens + 7) / 8;
    static_assert(BlockRows == 8 || BlockRows == 16 || BlockRows == 32);
    static_assert(BlockTokens > 0 && BlockTokens <= 32);
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(WarpK == 128 || WarpK == 256);
    static_assert(Stages == 1 || Stages == 2);
    static_assert(MinBlocksPerSm > 0);
};

template <class S, class Input, class Scale>
struct NativeFp8SlicedKStorage {
    static constexpr int kPlanes     = std::is_same_v<Input, float> ? 2 : 1;
    static constexpr int kCodes      = S::kStages * S::kBlockRows * S::kBlockK;
    static constexpr int kActivation = S::kStages * S::kBlockTokens * S::kBlockK * kPlanes * 2;
    static constexpr int kScales =
        std::is_same_v<Scale, std::uint8_t> ? S::kStages * S::kBlockRows * (S::kBlockK / 32) : 0;
    static constexpr int kStaging = kCodes + kActivation + kScales;
    static constexpr int kPartial =
        S::kKWarps * S::kMmaRows * 16 * S::kMmaTokens * 8 * sizeof(float);
    static constexpr int kBytes = kStaging > kPartial ? kStaging : kPartial;
};

template <class S, bool Swizzled>
struct NativeFp8SlicedRowTile {
    float* values;

    __device__ __forceinline__ static int offset(int row, int token) {
        if constexpr (Swizzled) {
            constexpr int group = S::kBlockTokens & -S::kBlockTokens;
            token ^= (row / (32 / group)) & (group - 1);
        }
        return row * S::kBlockTokens + token;
    }

    __device__ __forceinline__ float& at(int row, int token) const {
        return values[offset(row, token)];
    }
};

template <class Epilogue, class Output, int Tokens, class = void>
struct NativeFp8SlicedRowConsumer : std::false_type {};

template <class Epilogue, class Output, int Tokens>
struct NativeFp8SlicedRowConsumer<
    Epilogue, Output, Tokens,
    std::void_t<decltype(std::declval<const Epilogue&>().apply_row(
        std::declval<const Output&>(), 0, 0, std::declval<const float (&)[Tokens]>(), 0))>>
    : std::true_type {};

template <class S, class Input, class Scale, class Output, class Epilogue, class Rows>
__device__ __forceinline__ void
native_fp8_a16_sliced_k_body(const Input* __restrict__ x, const std::uint8_t* __restrict__ codes,
                             const Scale* __restrict__ scales, int n, int input_rows, int tokens,
                             Output output, Epilogue epilogue, Rows rows, int row_tile,
                             int token_begin, unsigned char* raw) {
    constexpr int BR = S::kBlockRows, BT = S::kBlockTokens, WK = S::kWarpK;
    constexpr int BK = S::kBlockK, KW = S::kKWarps, ST = S::kStages;
    constexpr int MR = S::kMmaRows, MT = S::kMmaTokens;
    constexpr bool fp32_scale    = std::is_same_v<Scale, float>;
    constexpr bool fp32_input    = std::is_same_v<Input, float>;
    using Storage                = NativeFp8SlicedKStorage<S, Input, Scale>;
    constexpr int planes         = Storage::kPlanes;
    constexpr bool row_consumer  = NativeFp8SlicedRowConsumer<Epilogue, Output, BT>::value;
    constexpr bool swizzled_rows = [] {
        if constexpr (requires { Epilogue::kSwizzleRowStorage; })
            return Epilogue::kSwizzleRowStorage;
        return false;
    }();
    using RowTile         = NativeFp8SlicedRowTile<S, swizzled_rows>;
    const auto row_offset = [](int row, int token) { return RowTile::offset(row, token); };
    auto* code_shared     = raw;
    auto* activation      = reinterpret_cast<__nv_bfloat16*>(raw + Storage::kCodes);
    auto* scale_shared    = raw + Storage::kCodes + Storage::kActivation;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, lid = lane & 3;
    const int k         = S::kStaticK ? S::kStaticK : input_rows;
    const int row_begin = row_tile * (BR / (Rows::kPaired ? 2 : 1));
    const int groups    = (k + BK - 1) / BK;
    float accum[MR][MT][4]{};

    const auto stage = [&](int slot, int group) {
        const int kb = group * BK;
        for (int item = tid; item < BR * (BK / 16); item += S::kThreads) {
            const int lr = item / (BK / 16), chunk = item % (BK / 16);
            const int row    = rows.weight_row(row_begin, lr, n);
            const bool valid = row < n && kb + chunk * 16 < k;
            common::cp_async_zfill<16, Cache::cg>(
                code_shared + slot * BR * BK + lr * BK + (chunk ^ (lr & 7)) * 16,
                codes + (valid ? static_cast<std::int64_t>(row) * k + kb + chunk * 16 : 0),
                valid ? 16 : 0);
        }
        if constexpr (!fp32_scale) {
            for (int item = tid; item < BR * (BK / 128); item += S::kThreads) {
                const int lr = item / (BK / 128), group128 = item % (BK / 128);
                const int row    = rows.weight_row(row_begin, lr, n);
                const bool valid = row < n && kb + group128 * 128 < k;
                common::cp_async_zfill<4>(
                    scale_shared + slot * BR * (BK / 32) + item * 4,
                    scales +
                        (valid ? static_cast<std::int64_t>(row) * (k / 32) + kb / 32 + group128 * 4
                               : 0),
                    valid ? 4 : 0);
            }
        }
        for (int item = lane; item < BT * (WK / 8); item += 32) {
            const int lt = item / (WK / 8), col = item % (WK / 8) * 8;
            const int token = token_begin + lt, kk = kb + warp * WK + col;
            const bool valid  = token < tokens && kk < k;
            auto* dst         = activation + (slot * KW * planes + warp * planes) * BT * WK;
            const int at      = lt * WK + common::swizzle_xor_64(lt, col);
            const auto offset = valid ? static_cast<std::int64_t>(token) * k + kk : 0;
            if constexpr (fp32_input) {
                const float4 first  = valid ? common::load_vec<float4>(x + offset) : float4{};
                const float4 second = valid ? common::load_vec<float4>(x + offset + 4) : float4{};
                const float values[]{first.x,  first.y,  first.z,  first.w,
                                     second.x, second.y, second.z, second.w};
#pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    const auto hi        = __floats2bfloat162_rn(values[i], values[i + 1]);
                    const auto hi_values = __bfloat1622float2(hi);
                    const auto lo =
                        __floats2bfloat162_rn(values[i] - hi_values.x, values[i + 1] - hi_values.y);
                    common::store_vec(dst + at + i, hi);
                    common::store_vec(dst + BT * WK + at + i, lo);
                }
            } else {
                common::cp_async_zfill<16, Cache::ca>(dst + at, x + offset, valid ? 16 : 0);
            }
        }
    };

#pragma unroll
    for (int slot = 0; slot < ST; ++slot) {
        if (slot < groups) {
            stage(slot, slot);
            common::cp_commit();
        }
    }
#pragma unroll 1
    for (int group = 0; group < groups; ++group) {
        if (group + ST <= groups)
            common::cp_wait<ST - 1>();
        else
            common::cp_wait<0>();
        __syncthreads();
        const int slot   = group % ST;
        const int warp_k = group * BK + warp * WK;
        const auto* xs   = activation + (slot * KW * planes + warp * planes) * BT * WK;
        const auto pair  = [&](int lr, int col, unsigned scale_word) {
            if constexpr (BR < 16)
                if (lr >= BR) return 0u;
            const int column = warp * WK + col;
            const int byte   = ((column / 16) ^ (lr & 7)) * 16 + (column & 15);
            const auto bits =
                common::load_vec<std::uint16_t>(code_shared + slot * BR * BK + lr * BK + byte);
            if constexpr (fp32_scale) {
                return common::fp8_e4m3x2_to_bf16x2_bits(bits);
            } else {
                const auto scale =
                    static_cast<std::uint8_t>(scale_word >> (((col & 127) >> 5) * 8));
                return common::fp8_dequant_bf16x2(bits, scale);
            }
        };
#pragma unroll
        for (int scale_group = 0; scale_group < WK; scale_group += 128) {
            unsigned packed_scales[MR][2]{};
            if constexpr (!fp32_scale) {
#pragma unroll
                for (int mi = 0; mi < MR; ++mi) {
                    const int lr       = mi * 16 + gid;
                    const auto* source = scale_shared + slot * BR * (BK / 32) + lr * (BK / 32) +
                                         warp * (WK / 32) + scale_group / 32;
                    packed_scales[mi][0] = common::load_vec<unsigned>(source);
                    if constexpr (BR >= 16)
                        packed_scales[mi][1] = common::load_vec<unsigned>(source + 8 * (BK / 32));
                }
            }
            float partial[MR][MT][4]{};
#pragma unroll
            for (int kk = scale_group; kk < scale_group + 128; kk += 16) {
                unsigned af[MR][4];
#pragma unroll
                for (int mi = 0; mi < MR; ++mi) {
                    const int lr = mi * 16 + gid, col = kk + lid * 2;
                    af[mi][0] = pair(lr, col, packed_scales[mi][0]);
                    af[mi][1] = pair(lr + 8, col, packed_scales[mi][1]);
                    af[mi][2] = pair(lr, col + 8, packed_scales[mi][0]);
                    af[mi][3] = pair(lr + 8, col + 8, packed_scales[mi][1]);
                }
#pragma unroll
                for (int plane = 0; plane < planes; ++plane) {
                    unsigned bf[MT][2];
#pragma unroll
                    for (int ni = 0; ni < MT; ++ni) {
                        // Narrow tiles repeat valid activation rows for unused MMA columns.
                        // Only the logical BT columns are stored by the epilogue.
                        const int lt  = BT < 8 || BT % 8 == 0 ? ni * 8 + (lane & 7) % BT
                                                              : (ni * 8 + (lane & 7)) % BT;
                        const int col = kk + ((lane >> 3) & 1) * 8;
                        common::ldmatrix_x2(bf[ni][0], bf[ni][1],
                                            common::smem_addr(xs + plane * BT * WK + lt * WK +
                                                              common::swizzle_xor_64(lt, col)));
                    }
#pragma unroll
                    for (int mi = 0; mi < MR; ++mi) {
#pragma unroll
                        for (int ni = 0; ni < MT; ++ni) {
                            auto& a = fp32_scale ? partial[mi][ni] : accum[mi][ni];
                            common::mma_bf16(a[0], a[1], a[2], a[3], af[mi][0], af[mi][1],
                                             af[mi][2], af[mi][3], bf[ni][0], bf[ni][1]);
                        }
                    }
                }
            }
            if constexpr (fp32_scale) {
#pragma unroll
                for (int mi = 0; mi < MR; ++mi) {
                    const int row0   = rows.weight_row(row_begin, mi * 16 + gid, n);
                    const int row1   = rows.weight_row(row_begin, mi * 16 + gid + 8, n);
                    const int column = warp_k + scale_group;
                    const float s0 =
                        row0 < n && column < k
                            ? scales[static_cast<std::int64_t>(row0 / 128) * (k / 128) +
                                     column / 128]
                            : 0;
                    const float s1 =
                        row1 < n && column < k
                            ? scales[static_cast<std::int64_t>(row1 / 128) * (k / 128) +
                                     column / 128]
                            : 0;
#pragma unroll
                    for (int ni = 0; ni < MT; ++ni) {
                        auto& a       = accum[mi][ni];
                        const auto& b = partial[mi][ni];
                        a[0]          = fmaf(b[0], s0, a[0]);
                        a[1]          = fmaf(b[1], s0, a[1]);
                        a[2]          = fmaf(b[2], s1, a[2]);
                        a[3]          = fmaf(b[3], s1, a[3]);
                    }
                }
            }
        }
        __syncthreads();
        if (group + ST < groups) {
            stage(slot, group + ST);
            common::cp_commit();
        }
    }

    auto* partial = reinterpret_cast<float*>(raw);
    if (warp != 0) {
#pragma unroll
        for (int mi = 0; mi < MR; ++mi) {
#pragma unroll
            for (int ni = 0; ni < MT; ++ni) {
                const auto& a = accum[mi][ni];
                common::store_vec(partial + ((warp * MR * MT + mi * MT + ni) * 32 + lane) * 4,
                                  make_float4(a[0], a[1], a[2], a[3]));
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int mi = 0; mi < MR; ++mi) {
#pragma unroll
            for (int ni = 0; ni < MT; ++ni) {
                auto& a = accum[mi][ni];
#pragma unroll
                for (int split = 1; split < KW; ++split) {
                    const float4 b = common::load_vec<float4>(
                        partial + ((split * MR * MT + mi * MT + ni) * 32 + lane) * 4);
                    a[0] += b.x;
                    a[1] += b.y;
                    a[2] += b.z;
                    a[3] += b.w;
                }
                const int lr = mi * 16 + gid, lt = ni * 8 + 2 * lid;
                if constexpr (Rows::kPaired || row_consumer) {
                    if (lt < BT) {
                        partial[row_offset(lr, lt)] = a[0];
                        if constexpr (BR >= 16) partial[row_offset(lr + 8, lt)] = a[2];
                    }
                    if (lt + 1 < BT) {
                        partial[row_offset(lr, lt + 1)] = a[1];
                        if constexpr (BR >= 16) partial[row_offset(lr + 8, lt + 1)] = a[3];
                    }
                } else {
                    const int row0 = rows.weight_row(row_begin, lr, n);
                    const int row1 = rows.weight_row(row_begin, lr + 8, n);
                    if (lt < BT && row0 < n && token_begin + lt < tokens)
                        output.store(row0, token_begin + lt,
                                     epilogue.apply(row0, token_begin + lt, a[0]));
                    if (lt + 1 < BT && row0 < n && token_begin + lt + 1 < tokens)
                        output.store(row0, token_begin + lt + 1,
                                     epilogue.apply(row0, token_begin + lt + 1, a[1]));
                    if (lr + 8 < BR && lt < BT && row1 < n && token_begin + lt < tokens)
                        output.store(row1, token_begin + lt,
                                     epilogue.apply(row1, token_begin + lt, a[2]));
                    if (lr + 8 < BR && lt + 1 < BT && row1 < n && token_begin + lt + 1 < tokens)
                        output.store(row1, token_begin + lt + 1,
                                     epilogue.apply(row1, token_begin + lt + 1, a[3]));
                }
            }
        }
    }
    if constexpr (Rows::kPaired || row_consumer) {
        __syncthreads();
        if constexpr (requires {
                          epilogue.template finish_rows<S>(output, RowTile{partial}, row_begin,
                                                           token_begin, n, tokens);
                      }) {
            epilogue.template finish_rows<S>(output, RowTile{partial}, row_begin, token_begin, n,
                                             tokens);
        } else if (tid < BR) {
            const int row = rows.weight_row(row_begin, tid, n);
            if constexpr (Rows::kPaired) {
                if (row < n / 2) {
                    int up = 0;
#pragma unroll
                    for (int r = 0; r < BR; ++r)
                        if (rows.weight_row(row_begin, r, n) == row + n / 2) up = r;
#pragma unroll
                    for (int t = 0; t < BT; ++t)
                        if (token_begin + t < tokens)
                            epilogue.apply_pair(output, row, token_begin + t,
                                                partial[row_offset(tid, t)],
                                                partial[row_offset(up, t)]);
                }
            } else if (row < n) {
                float values[BT];
#pragma unroll
                for (int t = 0; t < BT; ++t) values[t] = partial[row_offset(tid, t)];
                linear_finish_row(output, epilogue, row, token_begin, values,
                                  min(BT, tokens - token_begin));
            }
        }
    }
}

template <class S, class Input, class Scale, class Output, class Epilogue, class Rows>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void native_fp8_a16_sliced_k_kernel(
    const Input* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const Scale* __restrict__ scales, int n, int input_rows, int tokens, Output output,
    Epilogue epilogue, Rows rows) {
    using Storage         = NativeFp8SlicedKStorage<S, Input, Scale>;
    const int token_tiles = (tokens + S::kBlockTokens - 1) / S::kBlockTokens;
    native_fp8_a16_sliced_k_body<S>(
        x, codes, scales, n, input_rows, tokens, output, epilogue, rows, blockIdx.x / token_tiles,
        blockIdx.x % token_tiles * S::kBlockTokens, common::shared_storage<Storage::kBytes>());
}

template <class S, class Operands, class Output, class Epilogue, class Rows = Fp8IdentityRows>
void launch_native_fp8_a16_sliced_k(const Operands& p, Output output, Epilogue epilogue,
                                    cudaStream_t stream, Rows rows = {}) {
    static_assert(
        !requires(Operands p) { p.x_low; },
        "native FP8 sliced-K requires BF16 or FP32 input storage");
    using Input = typename Operands::InputType;
    using Scale = typename Operands::ScaleType;
    static_assert(std::is_same_v<Input, __nv_bfloat16> || std::is_same_v<Input, float>);
    static_assert(std::is_same_v<Scale, std::uint8_t> || std::is_same_v<Scale, float>);
    if (!p.x || !p.codes || !p.scales || p.rows <= 0 || p.k <= 0 || p.k % 128 || p.tokens <= 0 ||
        reinterpret_cast<std::uintptr_t>(p.x) % 16 ||
        reinterpret_cast<std::uintptr_t>(p.codes) % 16 ||
        reinterpret_cast<std::uintptr_t>(p.scales) % 4)
        throw std::invalid_argument("native FP8 sliced-K requires aligned operands and K128");
    if constexpr (S::kStaticK)
        if (p.k != S::kStaticK)
            throw std::invalid_argument("native FP8 sliced-K static K mismatch");
    const auto blocks = static_cast<std::int64_t>(common::div_up(p.rows, S::kBlockRows)) *
                        common::div_up(p.tokens, S::kBlockTokens);
    if (blocks > 2147483647LL)
        throw std::invalid_argument("native FP8 sliced-K grid exceeds CUDA grid.x");
    constexpr auto kernel = native_fp8_a16_sliced_k_kernel<S, Input, Scale, Output, Epilogue, Rows>;
    constexpr int bytes   = NativeFp8SlicedKStorage<S, Input, Scale>::kBytes;
    const auto shared     = common::prepare_shared<bytes, kernel>();
    kernel<<<static_cast<unsigned>(blocks), S::kThreads, shared, stream>>>(
        p.x, p.codes, p.scales, p.rows, p.k, p.tokens, output, epilogue, rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
