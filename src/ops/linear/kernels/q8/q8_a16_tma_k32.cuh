#pragma once

#include "ops/linear/kernels/q8/q8_a16_tma.cuh"

#include <type_traits>

namespace ninfer::ops::detail {

// 128x128 output tiles with token-tail predication, K32 computation, and three raw stages.
// K is a multiple of 32; incomplete final three-stage cycles consume only live slots.
// Stored Q8 codes and FP16 scales are consumed without packing or global scratch.
template <int RasterRows, int StaticK, bool BiasedDecode = false>
struct Q8A16TmaK32Schedule {
    static constexpr int kBlockRows = 128, kBlockTokens = 128, kBlockK = 32;
    static constexpr int kWarpRows = 64, kWarpTokens = 32;
    static constexpr int kWarpGridRows = 2, kWarpGridTokens = 4;
    static constexpr int kThreads = 256, kMinBlocksPerSm = 2, kStages = 3;
    static constexpr int kMmaRows = 4, kMmaTokens = 4, kMmaKSteps = 2;
    static constexpr int kRasterRows = RasterRows, kStaticK = StaticK;
    static constexpr bool kBiasedDecode  = BiasedDecode;
    static constexpr bool kPrefetchCodes = false;
    static constexpr int kCodeGroup = 1, kCodeStages = 3, kLookahead = 0;
    static constexpr int kSharedBytes = 45 * 1024;
    static_assert(RasterRows >= 0 && StaticK >= 96 && StaticK % 32 == 0);
};

// Group raw-code transfers independently of the three K32 activation stages.
// One K128 code tile retains two resident CTAs while amortizing TMA requests.
template <int RasterRows, int StaticK, bool BiasedDecode = false, int CodeGroup = 4,
          int CodeStages = 1, int Lookahead = 4>
struct Q8A16TmaK32PrefetchSchedule : Q8A16TmaK32Schedule<RasterRows, StaticK, BiasedDecode> {
    static_assert(CodeGroup == 1 || CodeGroup == 2 || CodeGroup == 4);
    static_assert(CodeStages >= 1 && CodeStages <= 3 && Lookahead > 0);
    static_assert(StaticK % (32 * CodeGroup) == 0 && StaticK / (32 * CodeGroup) >= CodeStages);
    static constexpr bool kPrefetchCodes = true;
    static constexpr int kCodeGroup = CodeGroup, kCodeStages = CodeStages, kLookahead = Lookahead;
    static constexpr int kSharedBytes =
        ((128 * 32 * 2 + 3 * 128 * 32 * 2 + CodeStages * 128 * 32 * CodeGroup + 24 +
          CodeStages * 8 + 1023) /
         1024) *
        1024;
    static_assert(kSharedBytes <= 99 * 1024);
};

template <class Cfg>
struct Q8A16TmaK32Storage {
    alignas(1024) __nv_bfloat16 weights[Cfg::kBlockRows * 32];
    alignas(1024) __nv_bfloat16 activations[3][Cfg::kBlockTokens * 32];
    alignas(128) std::uint8_t codes[3][Cfg::kBlockRows * 32];
    alignas(8) std::uint64_t ready[3];
};

template <class Cfg>
struct Q8A16TmaK32PrefetchStorage {
    alignas(1024) __nv_bfloat16 weights[Cfg::kBlockRows * 32];
    alignas(1024) __nv_bfloat16 activations[3][Cfg::kBlockTokens * 32];
    alignas(128) std::uint8_t codes[Cfg::kCodeStages][Cfg::kBlockRows * 32 * Cfg::kCodeGroup];
    alignas(8) std::uint64_t ready[3], codes_ready[Cfg::kCodeStages];
};

// 64-byte TMA swizzling uses address bits [8:7] to permute 16-byte chunks.
// Each BF16 row is 64 bytes; two rows form one 128-byte swizzle row.
__device__ __forceinline__ int q8_tma_swizzle_k32(int row, int column) {
    return column ^ (((row >> 1) & 3) << 3);
}

template <class Cfg, bool Full, class Output, class Epilogue, class RowPolicy>
__global__ __launch_bounds__(Cfg::kThreads, Cfg::kMinBlocksPerSm) void q8_a16_tma_k32_kernel(
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

    using Storage = std::conditional_t<Cfg::kPrefetchCodes, Q8A16TmaK32PrefetchStorage<Cfg>,
                                       Q8A16TmaK32Storage<Cfg>>;
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
    const int m0 = bx * kOutputRowsPerCta;
    const int n0 = token_begin + by * BN;
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
        if constexpr (Cfg::kPrefetchCodes)
            for (int stage = 0; stage < Cfg::kCodeStages; ++stage)
                common::cta_mbarrier_init(&storage.codes_ready[stage], 1);
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();
    auto issue = [&](int stage, int kt) {
        if (tid == 0) {
            constexpr int bytes = BN * BK * 2 + (Cfg::kPrefetchCodes ? 0 : BM * BK);
            auto* barrier       = &storage.ready[stage];
            common::cta_mbarrier_arrive_expect_tx(barrier, bytes);
            common::tma_load_2d(Bs[stage], &maps.x, kt * BK, n0, barrier);
            if constexpr (!Cfg::kPrefetchCodes) {
#pragma unroll
                for (int panel = 0; panel < BM / RowPolicy::kTmaRowsPerPanel; ++panel) {
                    constexpr int rows = RowPolicy::kTmaRowsPerPanel;
                    common::tma_load_2d(Cr[stage] + panel * rows * BK, &maps.codes, kt * BK,
                                        row_policy.weight_row(m0, panel * rows, m), barrier);
                }
            }
        }
    };

    constexpr int code_groups = Cfg::kStaticK / (BK * Cfg::kCodeGroup);
    const auto prefetch_codes = [&](int group) {
        if constexpr (Cfg::kPrefetchCodes) {
            if (tid == 0 && group < code_groups) {
#pragma unroll
                for (int panel = 0; panel < BM / RowPolicy::kTmaRowsPerPanel; ++panel)
                    asm volatile(
                        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(
                            &maps.codes),
                        "r"(group * Cfg::kCodeGroup * BK),
                        "r"(row_policy.weight_row(m0, panel * RowPolicy::kTmaRowsPerPanel, m))
                        : "memory");
            }
        }
    };
    const auto issue_codes = [&](int stage, int group) {
        if constexpr (Cfg::kPrefetchCodes) {
            prefetch_codes(group + Cfg::kLookahead);
            if (tid == 0) {
                auto* barrier = &storage.codes_ready[stage];
                common::cta_mbarrier_arrive_expect_tx(barrier, BM * BK * Cfg::kCodeGroup);
#pragma unroll
                for (int panel = 0; panel < BM / RowPolicy::kTmaRowsPerPanel; ++panel) {
                    constexpr int rows = RowPolicy::kTmaRowsPerPanel;
                    common::tma_load_2d(Cr[stage] + panel * rows * BK * Cfg::kCodeGroup,
                                        &maps.codes, group * Cfg::kCodeGroup * BK,
                                        row_policy.weight_row(m0, panel * rows, m), barrier);
                }
            }
        }
    };

    constexpr int ScaleVecs = (BM * (BK / 8) + Cfg::kThreads - 1) / Cfg::kThreads;
    std::uint16_t scale_regs[3][ScaleVecs];
    auto stage_scales = [&]<int Slot>(int kt) {
#pragma unroll
        for (int v = 0; v < ScaleVecs; ++v) {
            const int item = tid + v * Cfg::kThreads;
            if (item < BM * (BK / 8)) {
                const int row = item / (BK / 8), col = (item % (BK / 8)) * 8;
                const int grow      = row_policy.weight_row(m0, row, m);
                scale_regs[Slot][v] = common::load_ldg<std::uint16_t>(
                    scales + (std::int64_t(grow) * kg + kt * (BK / 32) + col / 32) * 2);
            }
        }
    };
    auto dequant_w = [&]<int Slot>(int kt) {
        constexpr int kChunksPerRow = BK / 8;
#pragma unroll
        for (int v = 0; v < ScaleVecs; ++v) {
            const int item = tid + v * Cfg::kThreads;
            if (item >= BM * kChunksPerRow) continue;
            const int row      = item / kChunksPerRow;
            const int chunk    = item - row * kChunksPerRow;
            const int col      = chunk * 8;
            const uint2 packed = [&] {
                if constexpr (Cfg::kPrefetchCodes)
                    return *reinterpret_cast<const uint2*>(
                        &Cr[(kt / Cfg::kCodeGroup) % Cfg::kCodeStages]
                           [row * BK * Cfg::kCodeGroup + (kt % Cfg::kCodeGroup) * BK + col]);
                else
                    return *reinterpret_cast<const uint2*>(&Cr[Slot][row * BK + col]);
            }();
            common::store_vec(&As[row * BK + q8_tma_swizzle_k32(row, col)],
                              q8_tma_decode_eight<Cfg::kBiasedDecode>(packed, scale_regs[Slot][v]));
        }
    };

    const int nkt = k / BK;
    stage_scales.template operator()<0>(0);
    stage_scales.template operator()<1>(1);
    stage_scales.template operator()<2>(2);
    issue(0, 0);
    issue(1, 1);
    issue(2, 2);
    if constexpr (Cfg::kPrefetchCodes) {
        for (int group = 0; group < Cfg::kLookahead; ++group) prefetch_codes(group);
        for (int stage = 0; stage < Cfg::kCodeStages; ++stage) issue_codes(stage, stage);
    }
    auto compute = [&]<int Slot>(int kt) {
        if (tid == 0) common::cta_mbarrier_wait(&storage.ready[Slot], (kt / 3) & 1);
        if constexpr (Cfg::kPrefetchCodes)
            if (kt % Cfg::kCodeGroup == 0 && tid == 0)
                common::cta_mbarrier_wait(
                    &storage.codes_ready[(kt / Cfg::kCodeGroup) % Cfg::kCodeStages],
                    (kt / (Cfg::kCodeGroup * Cfg::kCodeStages)) & 1);
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
                    common::smem_addr(&Bs[Slot][br * BK + q8_tma_swizzle_k32(br, bc)]));
            }
        }
        // Order generic Cr/Bs reads before their slot is reused by the TMA writer.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (kt + 3 < nkt) {
            stage_scales.template operator()<Slot>(kt + 3);
            issue(Slot, kt + 3);
        }
        if constexpr (Cfg::kPrefetchCodes)
            if (kt % Cfg::kCodeGroup == Cfg::kCodeGroup - 1 &&
                kt / Cfg::kCodeGroup + Cfg::kCodeStages < code_groups)
                issue_codes((kt / Cfg::kCodeGroup) % Cfg::kCodeStages,
                            kt / Cfg::kCodeGroup + Cfg::kCodeStages);

        // Keep only the current and next A fragment live. The larger output tile
        // needs 64 accumulator registers/thread, so retaining all A fragments
        // would constrain scheduling at the two-CTA register budget.
        unsigned af[2][4];
        const auto load_a = [&](int slot, int mi, int ks) {
            const int ar = wm * WM + mi * 16 + a_rowoff;
            const int ac = ks * 16 + a_coloff;
            common::ldmatrix_x4(af[slot][0], af[slot][1], af[slot][2], af[slot][3],
                                common::smem_addr(&As[ar * BK + q8_tma_swizzle_k32(ar, ac)]));
        };
        load_a(0, 0, 0);
#pragma unroll
        for (int fragment = 0; fragment < KSUB * MT; ++fragment) {
            const int slot = fragment & 1, mi = fragment % MT, ks = fragment / MT;
            if (fragment + 1 < KSUB * MT)
                load_a(slot ^ 1, (fragment + 1) % MT, (fragment + 1) / MT);
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                common::mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                                 af[slot][0], af[slot][1], af[slot][2], af[slot][3],
                                 reg_b[ks][ni][0], reg_b[ks][ni][1]);
            }
        }
    };
    constexpr int complete = (Cfg::kStaticK / BK) / 3 * 3;
#pragma unroll 1
    for (int kt = 0; kt < complete; kt += 3) {
        compute.template operator()<0>(kt);
        compute.template operator()<1>(kt + 1);
        compute.template operator()<2>(kt + 2);
    }

    if constexpr ((Cfg::kStaticK / BK) % 3 > 0) compute.template operator()<0>(complete);
    if constexpr ((Cfg::kStaticK / BK) % 3 > 1) compute.template operator()<1>(complete + 1);

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

template <class Schedule, class RowPolicy = Q8MmaIdentityRows<Schedule>, class Output,
          class Epilogue>
void launch_q8_a16_tma_k32(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                           cudaStream_t stream) {
    validate_q8_operands(operands);
    if (operands.k != Schedule::kStaticK || operands.padded_k != operands.k ||
        operands.rows % 128 != 0)
        throw std::invalid_argument("Q8 K32 TMA requires full 128-row tiles and matching static K");
    Q8A16TmaDescriptors maps;
    maps.x = q8_a16_tensor_map(const_cast<__nv_bfloat16*>(operands.x),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, operands.k, operands.tokens,
                               std::uint64_t(operands.k) * 2, 32, 128, CU_TENSOR_MAP_SWIZZLE_64B);
    maps.codes =
        q8_a16_tensor_map(const_cast<std::uint8_t*>(operands.codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                          operands.k, operands.rows, operands.k, 32 * Schedule::kCodeGroup,
                          RowPolicy::kTmaRowsPerPanel, CU_TENSOR_MAP_SWIZZLE_NONE);
    common::for_each_token_slice(operands.tokens, 128, [&](int offset, int count) {
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel =
                q8_a16_tma_k32_kernel<Schedule, Full, Output, Epilogue, RowPolicy>;
            constexpr int bytes = linear_mma_shared_bytes<Schedule, Epilogue>;
            using Storage =
                std::conditional_t<Schedule::kPrefetchCodes, Q8A16TmaK32PrefetchStorage<Schedule>,
                                   Q8A16TmaK32Storage<Schedule>>;
            static_assert(bytes >= sizeof(Storage) && bytes <= 99 * 1024);
            if constexpr (Schedule::kPrefetchCodes) (void)common::prepare_shared<bytes, kernel>();
            constexpr int ratio = Schedule::kBlockRows / RowPolicy::kOutputRowsPerCta;
            static_assert(Schedule::kBlockRows % RowPolicy::kOutputRowsPerCta == 0);
            static_assert(Schedule::kBlockRows % RowPolicy::kTmaRowsPerPanel == 0);
            kernel<<<dim3(operands.rows / ratio / RowPolicy::kOutputRowsPerCta,
                          common::div_up(count, 128)),
                     256, bytes, stream>>>(operands, maps, output, epilogue, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % 128 == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}
} // namespace ninfer::ops::detail
