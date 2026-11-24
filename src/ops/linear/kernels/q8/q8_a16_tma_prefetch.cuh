#pragma once

#include "ops/linear/kernels/q8/q8_a16_tma.cuh"

namespace ninfer::ops::detail {

// Independent activation/code barriers permit weight lookahead without another
// decoded tile. Grouped raw transfers amortize code staging; scale bits remain
// in registers and decode retains the stored FP16 scale semantics.
template <int Rows, int Tokens, int WarpRows, int WarpTokens, int MinBlocks, int StaticK,
          int CodeGroup = 1, int RasterRows = 8, int Lookahead = 4>
struct Q8A16TmaPrefetchSchedule : Q8A16TmaMmaSchedule<Rows, Tokens, WarpRows, WarpTokens, MinBlocks,
                                                      RasterRows, StaticK, 64, true, true, true> {
    static_assert(CodeGroup == 1 || CodeGroup == 2);
    static_assert(StaticK >= 256 && StaticK % (128 * CodeGroup) == 0);
    static_assert(Lookahead > 0);
    static constexpr int kCodeGroup = CodeGroup, kCodeStages = 2, kLookahead = Lookahead;
    static constexpr int kSharedBytes =
        ((Rows * 128 + Tokens * 256 + 2 * Rows * 64 * CodeGroup + 32 + 1023) / 1024) * 1024;
    static_assert(kSharedBytes <= 99 * 1024);
};

template <class S>
struct Q8A16TmaPrefetchStorage {
    alignas(1024) __nv_bfloat16 weights[S::kBlockRows * 64];
    alignas(1024) __nv_bfloat16 activations[2][S::kBlockTokens * 64];
    alignas(128) std::uint8_t codes[S::kCodeStages][S::kBlockRows * 64 * S::kCodeGroup];
    alignas(8) std::uint64_t ready[2], codes_ready[S::kCodeStages];
};

template <class Cfg, bool Full, class Output, class Epilogue>
__global__ __launch_bounds__(Cfg::kThreads, Cfg::kMinBlocksPerSm) void q8_a16_tma_prefetch_kernel(
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
    constexpr int kOutputRowsPerCta = BM;

    using Storage = Q8A16TmaPrefetchStorage<Cfg>;
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
    const int kg        = k / 32;
    constexpr int Group = Cfg::kCodeGroup, Stages = Cfg::kCodeStages;
    constexpr int part_groups = Cfg::kStaticK / (BK * Group);
    constexpr int part_tiles  = part_groups * Group;
    // The full contraction launches one z plane. Keep the transfer coordinates
    // and scale loads on the same CTA-uniform K origin.
    const int k_tile_begin = int(blockIdx.z) * part_tiles;

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
            common::tma_load_2d(Bs[stage], &maps.x, (k_tile_begin + kt) * BK, n0, barrier);
        }
    };
    auto prefetch_codes = [&](int group) {
        if (group < part_groups) {
            if (tid == 0)
                asm volatile(
                    "cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(
                        &maps.codes),
                    "r"((k_tile_begin + group * Group) * BK), "r"(m0)
                    : "memory");
        }
    };
    for (int group = 0; group < Cfg::kLookahead; ++group) prefetch_codes(group);
    auto issue_codes = [&](int stage, int group) {
        prefetch_codes(group + Cfg::kLookahead);
        if (tid == 0) {
            auto* barrier = &codes_ready[stage];
            common::cta_mbarrier_arrive_expect_tx(barrier, BM * BK * Group);
            common::tma_load_2d(Cr[stage], &maps.codes, (k_tile_begin + group * Group) * BK, m0,
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
                    k_tile_begin + kt < k / BK
                        ? common::load_ldg<std::uint16_t>(
                              scales + (std::int64_t(grow) * kg + (k_tile_begin + kt) * (BK / 32) +
                                        col / 32) *
                                           2)
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
            const int row      = item / kChunksPerRow;
            const int chunk    = item - row * kChunksPerRow;
            const int col      = chunk * 8;
            const uint2 packed = *reinterpret_cast<const uint2*>(
                &Cr[(kt / Group) % Stages][row * BK * Group + (kt % Group) * BK + col]);
            common::store_vec(&As[row * BK + common::swizzle_xor_64(row, col)],
                              q8_tma_decode_eight<Cfg::kBiasedDecode>(packed, scale_regs[Slot][v]));
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
                    common::smem_addr(&Bs[Slot][br * BK + q8_tma_activation_col<Cfg>(br, bc)]));
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
void launch_q8_a16_tma_prefetch(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream) {
    validate_q8_operands(operands);
    if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
        throw std::invalid_argument(
            "Q8 A16 prefetched TMA requires matching padding-free static K");
    if (operands.rows % Schedule::kBlockRows != 0)
        throw std::invalid_argument("Q8 A16 prefetched TMA requires complete row tiles");
    Q8A16TmaDescriptors maps;
    maps.x = q8_tma_activation_map<Schedule>(operands);
    maps.codes =
        q8_a16_tensor_map(const_cast<std::uint8_t*>(operands.codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                          operands.k, operands.rows, operands.k, 64 * Schedule::kCodeGroup,
                          Schedule::kBlockRows, CU_TENSOR_MAP_SWIZZLE_NONE);
    common::for_each_token_slice(
        operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
            const dim3 grid(operands.rows / Schedule::kBlockRows,
                            common::div_up(count, Schedule::kBlockTokens));
            const auto launch = [&]<bool Full>() {
                constexpr auto kernel =
                    q8_a16_tma_prefetch_kernel<Schedule, Full, Output, Epilogue>;
                constexpr int bytes = linear_mma_shared_bytes<Schedule, Epilogue>;
                static_assert(bytes >= sizeof(Q8A16TmaPrefetchStorage<Schedule>) &&
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
