#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/host/launch.h"
#include "ops/common/numeric/math.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/q8/q8_operands.h"
#include "ops/linear/kernels/q8/q8_schedule.cuh"
#include "ops/linear/kernels/q8/q8_mma_rows.cuh"

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// Q8-G32 and FP16 scales keep their stored representation. Two TMA stages hold
// raw codes and BF16 activations. Consume the current activation tile into
// registers before reusing its shared buffer: this gives both inputs two stages
// of lookahead without a third shared tile. Scales are prefetched exactly as
// stored FP16 values into registers. No producer warp or global workspace.
template <int Rows, int Tokens, int WarpRows, int WarpTokens, int MinBlocks, int RasterRows = 0,
          int StaticK = 0, int BlockK = 64, bool CompactA = false, bool BiasedDecode = false,
          bool FullRowTiles = false, bool SlideTokenTail = false, int WeightComponents = 1,
          int ActivationComponents = 1>
struct Q8A16TmaMmaSchedule : Q8A16MmaSchedule<Rows, Tokens, BlockK, WarpRows, WarpTokens, 2,
                                              MinBlocks, Q8MmaFragmentPipeline::PingPong, Cache::cg,
                                              Cache::cg, Cache::cg, false, StaticK, 1> {
    static constexpr int kRasterRows           = RasterRows;
    static constexpr int kWeightComponents     = WeightComponents;
    static constexpr int kActivationComponents = ActivationComponents;
    static constexpr bool kCompactA = CompactA, kBiasedDecode = BiasedDecode;
    static constexpr bool kFullRowTiles   = FullRowTiles;
    static constexpr bool kSlideTokenTail = SlideTokenTail;
    static constexpr int kSharedBytes =
        ((Rows * BlockK * (2 * WeightComponents + 2) +
          2 * Tokens * BlockK * 2 * ActivationComponents + 16 + 1023) /
         1024) *
        1024;
    static_assert(WeightComponents == 1 || WeightComponents == 2);
    static_assert(ActivationComponents == 1 || ActivationComponents == 2);
    static_assert(Rows % 16 == 0 && Tokens % 8 == 0);
    static_assert(Rows <= 256 && Tokens <= 256 && RasterRows >= 0);
    static_assert(BlockK == 64 || BlockK == 128);
    static_assert(StaticK == 0 || StaticK % (2 * BlockK) == 0);
    static_assert(kSharedBytes <= 99 * 1024);
};

template <class Cfg>
struct Q8A16TmaStorage {
    alignas(1024) __nv_bfloat16 weights[Cfg::kWeightComponents * Cfg::kBlockRows * Cfg::kBlockK];
    alignas(1024)
        __nv_bfloat16 activations[2][Cfg::kActivationComponents * Cfg::kBlockTokens * Cfg::kBlockK];
    alignas(128) std::uint8_t codes[2][Cfg::kBlockRows * Cfg::kBlockK];
    alignas(8) std::uint64_t ready[2];
};

union alignas(16) Q8TmaBf16x8Bits {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

static_assert(sizeof(Q8TmaBf16x8Bits) == 16);

template <bool Biased>
__device__ __forceinline__ uint4 q8_tma_decode_eight(uint2 packed, std::uint16_t scale_bits) {
    const float scale = __half2float(__ushort_as_half(scale_bits));
    Q8TmaBf16x8Bits decoded;
    if constexpr (Biased) {
        // Every finite FP16 scale times 32896 is exact in FP32 (at most 19 bits).
        // FMA cancels the exact bias before rounding to int8(code) * scale.
        const float bias = -32896.0f * scale;
        const unsigned a = packed.x ^ 0x80808080u, b = packed.y ^ 0x80808080u;
        const auto value = [&](unsigned word, unsigned selector) {
            return __fmaf_rn(__uint_as_float(__byte_perm(word, 0x47000000u, selector)), scale,
                             bias);
        };
        decoded.pair[0] = __floats2bfloat162_rn(value(a, 0x7504), value(a, 0x7514));
        decoded.pair[1] = __floats2bfloat162_rn(value(a, 0x7524), value(a, 0x7534));
        decoded.pair[2] = __floats2bfloat162_rn(value(b, 0x7504), value(b, 0x7514));
        decoded.pair[3] = __floats2bfloat162_rn(value(b, 0x7524), value(b, 0x7534));
    } else {
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const unsigned word = (pair < 2 ? packed.x : packed.y) >> ((pair & 1) * 16);
            const int q0        = static_cast<int>(static_cast<std::int8_t>(word & 0xffu));
            const int q1        = static_cast<int>(static_cast<std::int8_t>((word >> 8) & 0xffu));
            decoded.pair[pair]  = __floats2bfloat162_rn(float(q0) * scale, float(q1) * scale);
        }
    }
    return decoded.raw;
}

struct Q8TmaCompensatedBits {
    uint4 high, low;
};

__device__ __forceinline__ Q8TmaCompensatedBits
q8_tma_decode_eight_compensated(uint2 packed, std::uint16_t scale_bits) {
    const float scale = __half2float(__ushort_as_half(scale_bits));
    Q8TmaBf16x8Bits high, low;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const unsigned word  = (pair < 2 ? packed.x : packed.y) >> ((pair & 1) * 16);
        const float a        = float(static_cast<std::int8_t>(word)) * scale;
        const float b        = float(static_cast<std::int8_t>(word >> 8)) * scale;
        high.pair[pair]      = __floats2bfloat162_rn(a, b);
        const float2 rounded = __bfloat1622float2(high.pair[pair]);
        low.pair[pair]       = __floats2bfloat162_rn(a - rounded.x, b - rounded.y);
    }
    return {high.raw, low.raw};
}

struct alignas(128) Q8A16TmaDescriptors {
    CUtensorMap x;
    CUtensorMap codes;
};

inline CUtensorMap q8_a16_tensor_map(void* address, CUtensorMapDataType type, std::uint64_t columns,
                                     std::uint64_t rows, std::uint64_t stride,
                                     std::uint32_t tile_columns, std::uint32_t tile_rows,
                                     CUtensorMapSwizzle swizzle) {
    CUtensorMap result{};
    const std::uint64_t dimensions[] = {columns, rows};
    const std::uint64_t strides[]    = {stride};
    const std::uint32_t box[]        = {tile_columns, tile_rows};
    const std::uint32_t step[]       = {1, 1};
    const auto status                = cuTensorMapEncodeTiled(
        &result, type, 2, address, dimensions, strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("Q8 A16 TMA descriptor: ") +
                                 (name ? name : "CUDA driver error"));
    }
    return result;
}

template <class Schedule>
inline CUtensorMap q8_tma_activation_map(const Q8LinearOperands& p) {
    if constexpr (Schedule::kBlockK == 64) {
        return q8_a16_tensor_map(const_cast<__nv_bfloat16*>(p.x), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                 p.k, std::uint64_t(p.tokens) * Schedule::kActivationComponents,
                                 std::uint64_t(p.k) * 2, 64, Schedule::kBlockTokens,
                                 CU_TENSOR_MAP_SWIZZLE_128B);
    } else {
        // A 128-element K tile spans two 128-byte swizzle sectors without repacking X.
        CUtensorMap result{};
        const std::uint64_t dimensions[]{64, std::uint64_t(p.k / 64),
                                         std::uint64_t(p.tokens) * Schedule::kActivationComponents};
        const std::uint64_t strides[]{128, std::uint64_t(p.k) * 2};
        const std::uint32_t box[]{64, 2, Schedule::kBlockTokens}, steps[]{1, 1, 1};
        const auto status = cuTensorMapEncodeTiled(
            &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<__nv_bfloat16*>(p.x),
            dimensions, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (status != CUDA_SUCCESS) throw std::runtime_error("Q8 A16 K128 activation TMA map");
        return result;
    }
}

template <class Schedule>
__device__ __forceinline__ int q8_tma_activation_col(int row, int column) {
    if constexpr (Schedule::kBlockK == 64) {
        return common::swizzle_xor_64(row, column);
    } else {
        const int sector = (row * Schedule::kBlockK + column) / 64;
        return (column & ~63) + (((((column & 63) >> 3) ^ (sector & 7)) << 3)) + (column & 7);
    }
}

template <class Cfg, bool Full, class Output, class Epilogue, class RowPolicy>
__global__ __launch_bounds__(Cfg::kThreads, Cfg::kMinBlocksPerSm) void q8_a16_tma_mma_kernel(
    Q8LinearOperands operands, const __grid_constant__ Q8A16TmaDescriptors maps, Output output,
    Epilogue epilogue, int token_begin, int token_count) {
    const auto* __restrict__ scales = operands.scales;
    const int m = operands.rows, k = Cfg::kStaticK > 0 ? Cfg::kStaticK : operands.k,
              n                     = token_begin + token_count;
    constexpr int BM                = Cfg::kBlockRows;
    constexpr int BN                = Cfg::kBlockTokens;
    constexpr int BK                = Cfg::kBlockK;
    constexpr int WM                = Cfg::kWarpRows;
    constexpr int WN                = Cfg::kWarpTokens;
    constexpr int MT                = Cfg::kMmaRows;
    constexpr int NT                = Cfg::kMmaTokens;
    constexpr int KSUB              = Cfg::kMmaKSteps;
    constexpr int kOutputRowsPerCta = RowPolicy::kOutputRowsPerCta;
    const RowPolicy row_policy{};

    using Storage = Q8A16TmaStorage<Cfg>;
    static_assert(sizeof(Storage) == Cfg::kSharedBytes);
    extern __shared__ __align__(1024) unsigned char scratch[];
    auto& storage = *reinterpret_cast<Storage*>(scratch);
    auto& As      = storage.weights;
    auto& Bs      = storage.activations;
    auto& Cr      = storage.codes;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Cfg::kWarpGridTokens;
    const int wn   = warp % Cfg::kWarpGridTokens;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    int bx = blockIdx.x, by = blockIdx.y;
    if constexpr (Cfg::kRasterRows > 0) {
        const int linear = int(blockIdx.y) * int(gridDim.x) + int(blockIdx.x);
        const int span   = Cfg::kRasterRows * int(gridDim.y);
        const int group  = linear / span;
        const int first  = group * Cfg::kRasterRows;
        const int active = min(Cfg::kRasterRows, int(gridDim.x) - first);
        const int within = linear - group * span;
        bx               = first + within % active;
        by               = within / active;
    }
    const int m0       = bx * kOutputRowsPerCta;
    const int owned_n0 = token_begin + by * BN;
    // The last tile can reread valid X columns instead of a large TMA zero-filled tail.
    // The epilogue retains disjoint output ownership even when input windows overlap.
    const int n0 = Cfg::kSlideTokenTail ? min(owned_n0, max(0, n - BN)) : owned_n0;
    const int kg = k / 32;

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            acc[mi][ni][0] = 0.0f;
            acc[mi][ni][1] = 0.0f;
            acc[mi][ni][2] = 0.0f;
            acc[mi][ni][3] = 0.0f;
        }
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    if (tid == 0) {
#pragma unroll
        for (int stage = 0; stage < Cfg::kStages; ++stage)
            common::cta_mbarrier_init(&storage.ready[stage], 1);
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();
    auto issue = [&](int stage, int kt) {
        if (tid == 0) {
            constexpr int bytes = BN * BK * 2 * Cfg::kActivationComponents + BM * BK;
            auto* barrier       = &storage.ready[stage];
            common::cta_mbarrier_arrive_expect_tx(barrier, bytes);
#pragma unroll
            for (int component = 0; component < Cfg::kActivationComponents; ++component) {
                auto* destination = Bs[stage] + component * BN * BK;
                const int column  = n0 + component * operands.tokens;
                if constexpr (BK == 64)
                    common::tma_load_2d(destination, &maps.x, kt * BK, column, barrier);
                else
                    common::tma_load_3d(destination, &maps.x, 0, kt * (BK / 64), column, barrier);
            }
#pragma unroll
            for (int panel = 0; panel < BM / RowPolicy::kTmaRowsPerPanel; ++panel) {
                constexpr int panel_rows = RowPolicy::kTmaRowsPerPanel;
                common::tma_load_2d(Cr[stage] + panel * panel_rows * BK, &maps.codes, kt * BK,
                                    row_policy.weight_row(m0, panel * panel_rows, m), barrier);
            }
        }
    };

    constexpr int ScaleVecs = (BM * (BK / 8) + Cfg::kThreads - 1) / Cfg::kThreads;
    std::uint16_t scale_regs[2][ScaleVecs];
    auto stage_scales = [&]<int Slot>(int kt) {
#pragma unroll
        for (int v = 0; v < ScaleVecs; ++v) {
            const int item = tid + v * Cfg::kThreads;
            if (item < BM * (BK / 8)) {
                const int row = item / (BK / 8), col = (item % (BK / 8)) * 8;
                const int grow = row_policy.weight_row(m0, row, m);
                scale_regs[Slot][v] =
                    (Cfg::kFullRowTiles || Full || grow < m)
                        ? common::load_ldg<std::uint16_t>(
                              scales + (std::int64_t(grow) * kg + kt * (BK / 32) + col / 32) * 2)
                        : 0;
            }
        }
    };
    auto dequant_w = [&]<int Slot>() {
        constexpr int kChunksPerRow = BK / 8;
#pragma unroll
        for (int v = 0; v < ScaleVecs; ++v) {
            const int item = tid + v * Cfg::kThreads;
            if (item >= BM * kChunksPerRow) continue;
            const int row      = item / kChunksPerRow;
            const int chunk    = item - row * kChunksPerRow;
            const int col      = chunk * 8;
            const uint2 packed = *reinterpret_cast<const uint2*>(&Cr[Slot][row * BK + col]);
            auto* destination  = &As[row * BK + common::swizzle_xor_64(row, col)];
            if constexpr (Cfg::kWeightComponents == 1) {
                common::store_vec(destination, q8_tma_decode_eight<Cfg::kBiasedDecode>(
                                                   packed, scale_regs[Slot][v]));
            } else {
                const auto decoded = q8_tma_decode_eight_compensated(packed, scale_regs[Slot][v]);
                common::store_vec(destination, decoded.high);
                common::store_vec(destination + BM * BK, decoded.low);
            }
        }
    };

    const int nkt = k / BK;
    stage_scales.template operator()<0>(0);
    stage_scales.template operator()<1>(1);
    issue(0, 0);
    issue(1, 1);
    auto compute = [&]<int Slot>(int kt) {
        if (tid == 0) common::cta_mbarrier_wait(&storage.ready[Slot], (kt / 2) & 1);
        __syncthreads();
        dequant_w.template operator()<Slot>();
        unsigned reg_b[Cfg::kActivationComponents][KSUB][NT][2];
#pragma unroll
        for (int component = 0; component < Cfg::kActivationComponents; ++component) {
#pragma unroll
            for (int ks = 0; ks < KSUB; ++ks) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const int br = wn * WN + ni * 8 + b_rin, bc = ks * 16 + b_koff;
                    common::ldmatrix_x2(
                        reg_b[component][ks][ni][0], reg_b[component][ks][ni][1],
                        common::smem_addr(&Bs[Slot][component * BN * BK + br * BK +
                                                    q8_tma_activation_col<Cfg>(br, bc)]));
                }
            }
        }
        // Every generic read of Cr/Bs must precede the next async-proxy write.
        // CTA synchronization alone does not order accesses across those proxies.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (kt + 2 < nkt) {
            stage_scales.template operator()<Slot>(kt + 2);
            issue(Slot, kt + 2);
        }

        const auto accumulate_plane = [&](const __nv_bfloat16* weight_plane, int component) {
            if constexpr (Cfg::kCompactA) {
                // Retain only the current and next row atom, independently of the warp tile height.
                unsigned af[2][4];
                const auto load_a = [&](int slot, int mi, int ks) {
                    const int ar = wm * WM + mi * 16 + a_rowoff, ac = ks * 16 + a_coloff;
                    common::ldmatrix_x4(
                        af[slot][0], af[slot][1], af[slot][2], af[slot][3],
                        common::smem_addr(&weight_plane[ar * BK + common::swizzle_xor_64(ar, ac)]));
                };
                load_a(0, 0, 0);
#pragma unroll
                for (int fragment = 0; fragment < KSUB * MT; ++fragment) {
                    const int slot = fragment & 1, mi = fragment % MT, ks = fragment / MT;
                    if (fragment + 1 < KSUB * MT)
                        load_a(slot ^ 1, (fragment + 1) % MT, (fragment + 1) / MT);
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni)
                        common::mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2],
                                         acc[mi][ni][3], af[slot][0], af[slot][1], af[slot][2],
                                         af[slot][3], reg_b[component][ks][ni][0],
                                         reg_b[component][ks][ni][1]);
                }
            } else {
                unsigned af[2][MT][4];

                auto load_fragments = [&](int slot, int ks) {
#pragma unroll
                    for (int mi = 0; mi < MT; ++mi) {
                        const int ar = wm * WM + mi * 16 + a_rowoff;
                        const int ac = ks * 16 + a_coloff;
                        common::ldmatrix_x4(
                            af[slot][mi][0], af[slot][mi][1], af[slot][mi][2], af[slot][mi][3],
                            common::smem_addr(
                                &weight_plane[ar * BK + common::swizzle_xor_64(ar, ac)]));
                    }
                };

                if constexpr (Cfg::kFragmentPipeline == Q8MmaFragmentPipeline::PingPong)
                    load_fragments(0, 0);
#pragma unroll
                for (int ks = 0; ks < KSUB; ++ks) {
                    const int slot =
                        Cfg::kFragmentPipeline == Q8MmaFragmentPipeline::PingPong ? ks & 1 : 0;
                    if constexpr (Cfg::kFragmentPipeline == Q8MmaFragmentPipeline::Serial) {
                        load_fragments(0, ks);
                    }
                    if (Cfg::kFragmentPipeline == Q8MmaFragmentPipeline::PingPong &&
                        ks + 1 < KSUB) {
                        load_fragments(slot ^ 1, ks + 1);
                    }
#pragma unroll
                    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                        for (int ni = 0; ni < NT; ++ni) {
                            common::mma_bf16(
                                acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                                af[slot][mi][0], af[slot][mi][1], af[slot][mi][2], af[slot][mi][3],
                                reg_b[component][ks][ni][0], reg_b[component][ks][ni][1]);
                        }
                    }
                }
            }
        };
#pragma unroll
        for (int plane = 0; plane < Cfg::kWeightComponents; ++plane)
#pragma unroll
            for (int component = 0; component < Cfg::kActivationComponents; ++component)
                accumulate_plane(As + plane * BM * BK, component);
    };
#pragma unroll 1
    for (int kt = 0; kt < nkt; kt += 2) {
        compute.template operator()<0>(kt);
        compute.template operator()<1>(kt + 1);
    }

    const auto output_tile = linear_output_tile<kOutputRowsPerCta>(output, m0);
    if constexpr (Cfg::kSlideTokenTail) {
        epilogue.template finish_token_window<Cfg, Full>(output_tile, scratch, acc, m0, n0, m, n,
                                                         owned_n0);
    } else if constexpr (requires {
                             epilogue.template finish_tile<Cfg, Full>(output_tile, scratch, acc, m0,
                                                                      n0, m, n);
                         }) {
        epilogue.template finish_tile<Cfg, Full>(output_tile, scratch, acc, m0, n0, m, n);
    } else {
        const auto store = [&](int row, int token, float value) {
            if (Full || (row < m && token < n))
                output_tile.store(row, token, epilogue.apply(row, token, value));
        };
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int r0 = m0 + wm * WM + mi * 16 + gid;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int c0   = n0 + wn * WN + ni * 8 + 2 * lid;
                const float* a = acc[mi][ni];
                store(r0, c0, a[0]);
                store(r0, c0 + 1, a[1]);
                store(r0 + 8, c0, a[2]);
                store(r0 + 8, c0 + 1, a[3]);
            }
        }
    }
}

template <class Schedule, class RowPolicy = Q8MmaIdentityRows<Schedule>, class Output,
          class Epilogue>
void launch_q8_a16_tma_mma(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                           cudaStream_t stream, int output_tokens = 0) {
    validate_q8_operands(operands);
    const int tokens = output_tokens ? output_tokens : operands.tokens;
    if (tokens <= 0 || tokens > operands.tokens)
        throw std::invalid_argument("Q8 TMA output extent exceeds activation storage");
    if constexpr (Schedule::kFullRowTiles) {
        if (operands.rows % Schedule::kBlockRows != 0)
            throw std::invalid_argument("Q8 A16 TMA schedule requires complete weight-row tiles");
    }
    if (operands.k != operands.padded_k || operands.k % (2 * Schedule::kBlockK) != 0)
        throw std::invalid_argument("Q8 A16 TMA requires padding-free pairs of K tiles");
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK)
            throw std::invalid_argument("Q8 A16 TMA static K mismatch");
    }
    static_assert(Schedule::kBlockRows % RowPolicy::kOutputRowsPerCta == 0);
    static_assert(Schedule::kBlockRows % RowPolicy::kTmaRowsPerPanel == 0);
    constexpr int ratio = Schedule::kBlockRows / RowPolicy::kOutputRowsPerCta;
    if (operands.rows % ratio != 0)
        throw std::invalid_argument("Q8 TMA row mapping requires complete row groups");
    Q8A16TmaDescriptors maps;
    maps.x = q8_tma_activation_map<Schedule>(operands);
    maps.codes =
        q8_a16_tensor_map(const_cast<std::uint8_t*>(operands.codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                          operands.k, operands.rows, operands.k, Schedule::kBlockK,
                          RowPolicy::kTmaRowsPerPanel, CU_TENSOR_MAP_SWIZZLE_NONE);
    common::for_each_token_slice(tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(common::div_up(operands.rows / ratio, RowPolicy::kOutputRowsPerCta),
                        common::div_up(count, Schedule::kBlockTokens));
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel =
                q8_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue, RowPolicy>;
            constexpr int bytes = linear_mma_shared_bytes<Schedule, Epilogue>;
            static_assert(bytes >= sizeof(Q8A16TmaStorage<Schedule>) && bytes <= 99 * 1024);
            (void)common::prepare_shared<bytes, kernel>();
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(operands, maps, output, epilogue,
                                                                offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (operands.rows % Schedule::kBlockRows == 0 && count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}
} // namespace ninfer::ops::detail
