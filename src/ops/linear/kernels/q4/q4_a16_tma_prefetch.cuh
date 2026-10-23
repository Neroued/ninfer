#pragma once

// Independent raw-code and activation pipelines consume native row-split Q4.
// Scales stay in registers; only the current scaled BF16 weight tile is shared.
// No weight repack, device allocation, or global workspace is needed.
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/host/launch.h"
#include "ops/common/numeric/math.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/q4/q4_operands.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <string>

namespace ninfer::ops::detail {
template <int Rows, int Tokens, int WarpRows, int WarpTokens, int MinBlocks = 1, int StaticK = 4096,
          int CodeGroup = 2, int RasterRows = 16, int Lookahead = 4, int KPhaseStride = 1>
struct Q4A16TmaPrefetchSchedule {
    static constexpr int kBlockRows = Rows, kBlockTokens = Tokens, kBlockK = 64,
                         kWarpRows = WarpRows, kWarpTokens = WarpTokens;
    static constexpr int kWarpGridRows = Rows / WarpRows, kWarpGridTokens = Tokens / WarpTokens;
    static constexpr int kWarps = kWarpGridRows * kWarpGridTokens, kThreads = kWarps * 32;
    static constexpr int kMmaRows = WarpRows / 16, kMmaTokens = WarpTokens / 8, kMmaKSteps = 4;
    static constexpr int kMinBlocksPerSm = MinBlocks, kStaticK = StaticK, kCodeGroup = CodeGroup,
                         kPhaseStride = KPhaseStride, kCodeStages = 2, kRasterRows = RasterRows,
                         kLookahead = Lookahead;
    static constexpr int kSharedBytes =
        ((Rows * 128 + Tokens * 256 + 2 * Rows * 32 * CodeGroup + 32 + 1023) / 1024) * 1024;
    static_assert(Rows > 0 && Tokens > 0 && MinBlocks > 0 && Lookahead > 0 && RasterRows >= 0);
    static_assert(CodeGroup == 1 || CodeGroup == 2);
    static_assert(KPhaseStride >= 0);
    static_assert(Rows % WarpRows == 0 && Tokens % WarpTokens == 0 && WarpRows % 16 == 0 &&
                  WarpTokens % 8 == 0);
    static_assert(StaticK >= 256 && StaticK % (128 * CodeGroup) == 0);
    static_assert(kThreads <= 1024 && kSharedBytes <= 99 * 1024);
};

template <class S>
struct Q4A16TmaPrefetchStorage {
    alignas(1024) __nv_bfloat16 weights[S::kBlockRows * 64];
    alignas(1024) __nv_bfloat16 activations[2][S::kBlockTokens * 64];
    alignas(128) std::uint8_t codes[S::kCodeStages][S::kBlockRows * 32 * S::kCodeGroup];
    alignas(8) std::uint64_t ready[2], codes_ready[S::kCodeStages];
};

// A decoded tile keeps the stored FP16 scale before its private BF16 cast.
// 32776 * scale is exact in FP32 for every finite FP16 scale, so FMA removes
// the exact bias before rounding the signed nibble product.
union alignas(16) Q4TmaBf16x8Bits {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

__device__ __forceinline__ uint4 q4_tma_decode_eight(unsigned packed, std::uint16_t scale_bits) {
    const float scale   = __half2float(__ushort_as_half(scale_bits));
    const float bias    = -32776.0f * scale;
    const unsigned even = (packed & 0x0f0f0f0fu) ^ 0x08080808u;
    const unsigned odd  = ((packed >> 4) & 0x0f0f0f0fu) ^ 0x08080808u;
    const auto value    = [&](unsigned word, unsigned selector) {
        return __fmaf_rn(__uint_as_float(__byte_perm(word, 0x47000000u, selector)), scale, bias);
    };
    Q4TmaBf16x8Bits decoded;
    decoded.pair[0] = __floats2bfloat162_rn(value(even, 0x7504), value(odd, 0x7504));
    decoded.pair[1] = __floats2bfloat162_rn(value(even, 0x7514), value(odd, 0x7514));
    decoded.pair[2] = __floats2bfloat162_rn(value(even, 0x7524), value(odd, 0x7524));
    decoded.pair[3] = __floats2bfloat162_rn(value(even, 0x7534), value(odd, 0x7534));
    return decoded.raw;
}

struct alignas(128) Q4A16TmaDescriptors {
    CUtensorMap x, codes;
};

inline CUtensorMap q4_a16_tensor_map(void* address, CUtensorMapDataType type, std::uint64_t columns,
                                     std::uint64_t rows, std::uint64_t stride,
                                     std::uint32_t tile_columns, std::uint32_t tile_rows,
                                     CUtensorMapSwizzle swizzle) {
    CUtensorMap result{};
    const std::uint64_t dimensions[]{columns, rows}, strides[]{stride};
    const std::uint32_t box[]{tile_columns, tile_rows}, step[]{1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &result, type, 2, address, dimensions, strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("Q4 A16 TMA descriptor: ") +
                                 (name ? name : "CUDA driver error"));
    }
    return result;
}

template <class S>
inline CUtensorMap q4_tma_activation_map(const Q4LinearOperands& p) {
    return q4_a16_tensor_map(const_cast<__nv_bfloat16*>(p.x), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.k,
                             p.tokens, std::uint64_t(p.k) * 2, 64, S::kBlockTokens,
                             CU_TENSOR_MAP_SWIZZLE_128B);
}

template <class Cfg, bool Full, class Output, class Epilogue>
__global__ __launch_bounds__(Cfg::kThreads, Cfg::kMinBlocksPerSm) void q4_a16_tma_prefetch_kernel(
    Q4LinearOperands operands, const __grid_constant__ Q4A16TmaDescriptors maps, Output output,
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
    constexpr int kOutputRowsPerCta = BM;

    using Storage = Q4A16TmaPrefetchStorage<Cfg>;
    static_assert(sizeof(Storage) == Cfg::kSharedBytes);
    extern __shared__ __align__(1024) unsigned char scratch[];
    auto& storage     = *reinterpret_cast<Storage*>(scratch);
    auto& As          = storage.weights;
    auto& Bs          = storage.activations;
    auto& Cr          = storage.codes;
    auto& ready       = storage.ready;
    auto& codes_ready = storage.codes_ready;

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
    const int m0        = bx * kOutputRowsPerCta;
    const int n0        = token_begin + by * BN;
    const int kg        = k / 64;
    constexpr int Group = Cfg::kCodeGroup, Stages = Cfg::kCodeStages;
    constexpr int part_groups = Cfg::kStaticK / (BK * Group);
    constexpr int part_tiles  = part_groups * Group;

    // Rotate complete quantization groups while pairing codes, scales, and X.
    const int phase_group = (bx * Cfg::kPhaseStride) % part_groups;
    const auto k_index    = [&](int tile) { return (tile + phase_group * Group) % part_tiles; };
    const auto code_index = [&](int group) { return (group + phase_group) % part_groups; };
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
        for (int i = 0; i < 2; ++i) common::cta_mbarrier_init(&ready[i], 1);
        for (int i = 0; i < Stages; ++i) common::cta_mbarrier_init(&codes_ready[i], 1);
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();
    auto issue_x = [&](int stage, int kt) {
        if (tid == 0) {
            auto* barrier = &ready[stage];
            common::cta_mbarrier_arrive_expect_tx(barrier, BN * BK * 2);
            common::tma_load_2d(Bs[stage], &maps.x, k_index(kt) * BK, n0, barrier);
        }
    };
    auto prefetch_codes = [&](int group) {
        if (group < part_groups) {
            if (tid == 0)
                asm volatile(
                    "cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(
                        &maps.codes),
                    "r"(code_index(group) * Group * BK / 2), "r"(m0)
                    : "memory");
        }
    };
    for (int group = 0; group < Cfg::kLookahead; ++group) prefetch_codes(group);
    auto issue_codes = [&](int stage, int group) {
        prefetch_codes(group + Cfg::kLookahead);
        if (tid == 0) {
            auto* barrier = &codes_ready[stage];
            common::cta_mbarrier_arrive_expect_tx(barrier, BM * BK * Group / 2);
            common::tma_load_2d(Cr[stage], &maps.codes, code_index(group) * Group * BK / 2, m0,
                                barrier);
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
                const int grow = m0 + row;
                scale_regs[Slot][v] =
                    kt < k / BK
                        ? common::load_ldg<std::uint16_t>(
                              scales +
                              (std::int64_t(grow) * kg + k_index(kt) * (BK / 64) + col / 64) * 2)
                        : 0;
            }
        }
    };
    auto dequant_w = [&]<int Slot>(int kt) {
        constexpr int kChunksPerRow = BK / 8;
#pragma unroll
        for (int v = 0; v < ScaleVecs; ++v) {
            const int item = tid + v * Cfg::kThreads;
            if (item >= BM * kChunksPerRow) continue;
            const int row         = item / kChunksPerRow;
            const int chunk       = item - row * kChunksPerRow;
            const int col         = chunk * 8;
            const unsigned packed = *reinterpret_cast<const unsigned*>(
                &Cr[(kt / Group) % Stages][row * BK * Group / 2 + (kt % Group) * BK / 2 + col / 2]);
            common::store_vec(&As[row * BK + common::swizzle_xor_64(row, col)],
                              q4_tma_decode_eight(packed, scale_regs[Slot][v]));
        }
    };

    constexpr int nkt = part_tiles;
    stage_scales.template operator()<0>(0);
    stage_scales.template operator()<1>(1);
    issue_x(0, 0);
    issue_x(1, 1);
    for (int stage = 0; stage < Stages; ++stage) issue_codes(stage, stage);
    auto compute = [&]<int Slot>(int kt) {
        if (tid == 0) common::cta_mbarrier_wait(&ready[Slot], (kt / 2) & 1);
        if (kt % Group == 0 && tid == 0)
            common::cta_mbarrier_wait(&codes_ready[(kt / Group) % Stages],
                                      (kt / (Group * Stages)) & 1);
        __syncthreads();
        dequant_w.template operator()<Slot>(kt);
        unsigned reg_b[KSUB][NT][2];
#pragma unroll
        for (int ks = 0; ks < KSUB; ++ks) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int br = wn * WN + ni * 8 + b_rin, bc = ks * 16 + b_koff;
                common::ldmatrix_x2(
                    reg_b[ks][ni][0], reg_b[ks][ni][1],
                    common::smem_addr(&Bs[Slot][br * BK + common::swizzle_xor_64(br, bc)]));
            }
        }
        // Every generic read of Cr/Bs must precede the next async-proxy write.
        // CTA synchronization alone does not order accesses across those proxies.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (kt + 2 < nkt) {
            stage_scales.template operator()<Slot>(kt + 2);
            issue_x(Slot, kt + 2);
        }

        if (kt % Group == Group - 1 && kt / Group + Stages < part_groups)
            issue_codes((kt / Group) % Stages, kt / Group + Stages);

        {
            // Retain only the current and next row atom, independently of the warp tile height.
            unsigned af[2][4];
            const auto load_a = [&](int slot, int mi, int ks) {
                const int ar = wm * WM + mi * 16 + a_rowoff, ac = ks * 16 + a_coloff;
                common::ldmatrix_x4(
                    af[slot][0], af[slot][1], af[slot][2], af[slot][3],
                    common::smem_addr(&As[ar * BK + common::swizzle_xor_64(ar, ac)]));
            };
            load_a(0, 0, 0);
#pragma unroll
            for (int fragment = 0; fragment < KSUB * MT; ++fragment) {
                const int slot = fragment & 1, mi = fragment % MT, ks = fragment / MT;
                if (fragment + 1 < KSUB * MT)
                    load_a(slot ^ 1, (fragment + 1) % MT, (fragment + 1) / MT);
#pragma unroll
                for (int ni = 0; ni < NT; ++ni)
                    common::mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                                     af[slot][0], af[slot][1], af[slot][2], af[slot][3],
                                     reg_b[ks][ni][0], reg_b[ks][ni][1]);
            }
        }
    };
#pragma unroll 1
    for (int kt = 0; kt < nkt; kt += 2) {
        compute.template operator()<0>(kt);
        compute.template operator()<1>(kt + 1);
    }

    const auto output_tile = linear_output_tile<kOutputRowsPerCta>(output, m0);
    if constexpr (requires {
                      epilogue.template finish_tile<Cfg, Full>(output_tile, scratch, acc, m0, n0, m,
                                                               n);
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

template <class Schedule, class Output, class Epilogue>
void launch_q4_a16_tma_prefetch(const Q4LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream) {
    validate_q4_operands(operands);
    if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
        throw std::invalid_argument(
            "Q4 A16 prefetched TMA requires matching padding-free static K");
    if (operands.rows % Schedule::kBlockRows != 0)
        throw std::invalid_argument("Q4 A16 prefetched TMA requires complete row tiles");
    Q4A16TmaDescriptors maps;
    maps.x = q4_tma_activation_map<Schedule>(operands);
    maps.codes =
        q4_a16_tensor_map(const_cast<std::uint8_t*>(operands.codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                          operands.k / 2, operands.rows, operands.k / 2, 32 * Schedule::kCodeGroup,
                          Schedule::kBlockRows, CU_TENSOR_MAP_SWIZZLE_NONE);
    common::for_each_token_slice(
        operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
            const dim3 grid(operands.rows / Schedule::kBlockRows,
                            common::div_up(count, Schedule::kBlockTokens));
            const auto launch = [&]<bool Full>() {
                constexpr auto kernel =
                    q4_a16_tma_prefetch_kernel<Schedule, Full, Output, Epilogue>;
                constexpr int bytes = linear_mma_shared_bytes<Schedule, Epilogue>;
                static_assert(bytes >= sizeof(Q4A16TmaPrefetchStorage<Schedule>) &&
                              bytes <= 99 * 1024);
                (void)common::prepare_shared<bytes, kernel>();
                kernel<<<grid, Schedule::kThreads, bytes, stream>>>(operands, maps, output,
                                                                    epilogue, offset, count);
                CUDA_CHECK(cudaGetLastError());
            };
            if (count % Schedule::kBlockTokens == 0)
                launch.template operator()<true>();
            else
                launch.template operator()<false>();
        });
}
} // namespace ninfer::ops::detail
