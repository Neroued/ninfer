#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, 2304>;

using Gemv = Bf16KTailSchedule<
    StaticK<Bf16A16GemvSchedule<4, 4, 1, 16, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>>>;
using Sliced8 = StaticK<Bf16A16SlicedKMmaSchedule<32, 8, 4, 64, 2, Cache::cg, Cache::cg, 1, 1>>;
using Sliced16 =
    StaticK<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2, Cache::cg, Cache::cg, 1, 1,
                                      Bf16MmaRaster::TokenFast>>;
using Sliced32 =
    StaticK<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 2, Cache::cg, Cache::cg, 1, 1,
                                      Bf16MmaRaster::TokenFast>>;
using TmaR64T32  = StaticK<Bf16A16TmaMmaSchedule<64, 32, 128, 32, 16, 2, 1>>;
using TmaR128T64 =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 128, 32, 32, 2, 1, Bf16MmaRaster::TokenFast, 1, 2>>;
// 1,563 tiles of 128 rows leave a half tile when processing 256 rows per CTA.
using TmaR256T64 = Bf16RowTailSchedule<
    StaticK<Bf16A16TmaMmaSchedule<256, 64, 64, 64, 32, 2, 1, Bf16MmaRaster::Grouped, 4, 2>>>;
using TmaR128T256 =
    StaticK<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1, Bf16MmaRaster::Grouped, 4>>;

} // namespace

Bf16Launch select_bf16_n200064_k2304(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Sliced8>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Sliced16>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<Sliced32>;
    if (tokens <= 64) return launch_bf16_tma_mma<TmaR64T32>;
    if (tokens <= 128) return launch_bf16_tma_mma<TmaR128T64, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 640) return launch_bf16_tma_mma<TmaR256T64, Bf16TmaRowVectorEpilogue>;
    return launch_bf16_tma_mma<TmaR128T256, Bf16MmaStreamingWarpStoreEpilogue>;
}

} // namespace ninfer::ops::detail
