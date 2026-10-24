#pragma once

#include "ops/linear/kernels/bf16/bf16_a16_tma_mma.cuh"

namespace ninfer::ops::detail {

// All warps compute. Two weight tiles overlap one activation tile; consumers
// retain that activation tile in registers before the next TMA overwrites it.
template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens,
          int MinBlocksPerSm = 2, Bf16MmaRaster Raster = Bf16MmaRaster::TokenFast,
          int RasterGroupRows = 1>
struct Bf16A16TmaRegisterBSchedule
    : Bf16A16MmaSchedule<BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens, 2, MinBlocksPerSm,
                         Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::Serial, Raster,
                         Bf16MmaSwizzle::Tma128, RasterGroupRows> {
    static constexpr int kProducerThreads = 0;
    static constexpr int kWarpGridTokens  = BlockTokens / WarpTokens;
    static constexpr int kTensorBytes     = (2 * BlockRows + BlockTokens) * BlockK * 2;
    static constexpr int kSharedElements  = kTensorBytes / 2;
    static constexpr int kBarrierBytes    = 3 * sizeof(std::uint64_t);
    static constexpr int kSharedBytes     = kTensorBytes + kBarrierBytes;
    static_assert(BlockRows <= 256 && BlockTokens <= 256 && kSharedBytes <= 99 * 1024);
};

template <class Schedule, bool FullTokens, class Output, class Epilogue,
          class WeightAccess = Bf16ContiguousTmaWeightAccess>
__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::kMinBlocksPerSm) void bf16_a16_tma_register_b_kernel(const __grid_constant__
                                                                       Bf16TmaDescriptors
                                                                           descriptors,
                                                                   Output output, Epilogue epilogue,
                                                                   int rows, int input_rows,
                                                                   int token_offset, int count) {
    constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens;
    constexpr int BK = Schedule::kBlockK, MT = Schedule::kMmaRows, NT = Schedule::kMmaTokens;
    constexpr int KS  = BK / 16;
    const int k       = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int tiles_k = k / BK + (bf16_predicated_k<Schedule> && k % BK != 0);
    const int tiles_r = bf16_predicated_rows<Schedule> ? (rows + BR - 1) / BR : rows / BR;
    int tile_r, tile_t;
    bf16_mma_tile_coordinates<Schedule>(blockIdx.x, tiles_r, (count + BT - 1) / BT, tile_r, tile_t);
    const int row_begin = tile_r * BR, token_begin = token_offset + tile_t * BT;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int wm = warp / Schedule::kWarpsTokens, wn = warp % Schedule::kWarpsTokens;
    extern __shared__ __align__(128) unsigned char storage[];
    auto* a = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* b = a + 2 * BR * BK;
    auto* a_ready =
        reinterpret_cast<std::uint64_t*>(storage + bf16_tma_scratch_bytes<Schedule, Epilogue>);
    auto* b_ready = a_ready + 2;
    if (threadIdx.x == 0) {
        common::cta_mbarrier_init(a_ready, 1);
        common::cta_mbarrier_init(a_ready + 1, 1);
        common::cta_mbarrier_init(b_ready, 1);
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();
    const auto load_a = [&](int kt) {
        const int slot = kt % 2;
        common::cta_mbarrier_arrive_expect_tx(a_ready + slot, BR * BK * 2);
        WeightAccess::template load<Schedule>(a + slot * BR * BK, &descriptors.weight,
                                              kt * (BK / 64), row_begin, a_ready + slot);
    };
    const auto load_b = [&](int kt) {
        common::cta_mbarrier_arrive_expect_tx(b_ready, BT * BK * 2);
        bf16_tma_load<Schedule>(b, &descriptors.activation, kt * (BK / 64), token_begin, b_ready);
    };
    if (threadIdx.x == 0) {
        load_a(0);
        if (tiles_k > 1) load_a(1);
        load_b(0);
    }
    float accum[MT][NT][4] = {};
#pragma unroll 1
    for (int kt = 0; kt < tiles_k; ++kt) {
        common::cta_mbarrier_wait(a_ready + kt % 2, (kt / 2) & 1);
        common::cta_mbarrier_wait(b_ready, kt & 1);
        unsigned reg_b[KS][NT][2];
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = wn * Schedule::kWarpTokens + ni * 8 + (lane & 7);
                const int col = ks * 16 + ((lane >> 3) & 1) * 8;
                common::ldmatrix_x2(
                    reg_b[ks][ni][0], reg_b[ks][ni][1],
                    common::smem_addr(b + row * BK + bf16_mma_shared_col<Schedule>(row, col)));
            }
        }
        // Retire every generic-proxy B read before its next async-proxy write.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (threadIdx.x == 0 && kt + 1 < tiles_k) load_b(kt + 1);
        const auto* current_a = a + (kt % 2) * BR * BK;
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row =
                    wm * Schedule::kWarpRows + mi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int col = ks * 16 + (lane >> 4) * 8;
                unsigned af[4];
                common::ldmatrix_x4(af[0], af[1], af[2], af[3],
                                    common::smem_addr(current_a + row * BK +
                                                      bf16_mma_shared_col<Schedule>(row, col)));
#pragma unroll
                for (int ni = 0; ni < NT; ++ni)
                    common::mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2],
                                     accum[mi][ni][3], af[0], af[1], af[2], af[3], reg_b[ks][ni][0],
                                     reg_b[ks][ni][1]);
            }
        }
        // The same CTA owns A reads and the next write into this two-tile ring.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (threadIdx.x == 0 && kt + 2 < tiles_k) load_a(kt + 2);
    }
    bf16_finish_mma_tile<Schedule, FullTokens>(output, epilogue, storage, accum, row_begin,
                                               token_begin, rows, token_offset + count, warp, lane);
}

template <class Schedule, class Output, class Epilogue,
          class WeightAccess = Bf16ContiguousTmaWeightAccess>
void launch_bf16_a16_tma_register_b(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                                    cudaStream_t stream) {
    const auto descriptors = make_bf16_tma_descriptors<Schedule, WeightAccess>(p);
    common::for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const auto blocks =
            static_cast<std::int64_t>(bf16_predicated_rows<Schedule>
                                          ? common::div_up(p.rows, Schedule::kBlockRows)
                                          : p.rows / Schedule::kBlockRows) *
            common::div_up(count, Schedule::kBlockTokens);
        if (blocks > 2147483647LL)
            throw std::invalid_argument("BF16 register-B TMA grid exceeds CUDA grid.x capacity");
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel =
                bf16_a16_tma_register_b_kernel<Schedule, Full, Output, Epilogue, WeightAccess>;
            constexpr int bytes =
                bf16_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            common::prepare_shared<bytes, kernel, true>();
            kernel<<<static_cast<unsigned>(blocks), Schedule::kThreads, bytes, stream>>>(
                descriptors, output, epilogue, p.rows, p.k, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

} // namespace ninfer::ops::detail
