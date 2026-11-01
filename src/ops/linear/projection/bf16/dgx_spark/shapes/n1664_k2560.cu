#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// The geometry owns its CUDA entries, even where constants match another Op.
template <class Base>
struct Bf16N1664K2560Schedule : Base {};

using Gemv = Bf16N1664K2560Schedule<Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>>;
template <int Stages>
using Hot32  = Bf16N1664K2560Schedule<Bf16ScheduleInstance<
     Bf16A16MmaSchedule<32, 32, 128, 16, 8, Stages, 1, Cache::cg, Cache::cg,
                        Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
     2560>>;
using HotTma = Bf16N1664K2560Schedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 2560>>;
using Mid48 = Bf16N1664K2560Schedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2>, 2560>>;
using Bulk64 = Bf16N1664K2560Schedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2>, 2560>>;
using Bulk80 = Bf16N1664K2560Schedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 3>, 2560>>;
template <Bf16MmaRaster Raster>
using Bulk96 = Bf16N1664K2560Schedule<Bf16ScheduleInstance<
    Bf16RowTailSchedule<Bf16A16TmaMmaSchedule<96, 128, 64, 32, 32, 3, 1, Raster>>, 2560>>;
template <Bf16MmaRaster Raster>
using Bulk192 = Bf16N1664K2560Schedule<Bf16ScheduleInstance<
    Bf16RowTailSchedule<Bf16A16TmaMmaSchedule<192, 128, 64, 64, 32, 2, 1, Raster>>, 2560>>;

template <class S>
void launch_warp_projection(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n1664_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 32) return launch_bf16_mma<Hot32<3>>;
    if (tokens <= 64) return launch_bf16_mma<Hot32<2>>;
    if (tokens <= 96) return launch_bf16_mma<Hot32<3>>;
    if (tokens <= 128) return launch_bf16_tma_mma<HotTma>;
    if (tokens <= 160) return launch_warp_projection<Bulk80>;
    if (tokens <= 256) return launch_bf16_tma_mma<Bulk96<Bf16MmaRaster::TokenFast>>;
    if (tokens <= 336) return launch_warp_projection<Mid48>;
    if (tokens <= 448) return launch_warp_projection<Bulk64>;
    if (tokens <= 560) return launch_warp_projection<Bulk80>;
    // Keep the measured 80-token trough before its next CTA wave at 1441.
    if (tokens > 1408 && tokens <= 1440) return launch_warp_projection<Bulk80>;
    const std::int64_t t = tokens;
    const auto waves96   = (18 * ((t + 127) / 128) + 47) / 48;
    const auto waves192  = (9 * ((t + 127) / 128) + 47) / 48;
    const auto cost96    = waves96 * 96 * 128;
    const auto cost192   = waves192 * 192 * 128;
    if (tokens <= 2048) {
        const auto waves64 = (13 * ((t + 63) / 64) + 95) / 96;
        const auto cost64  = waves64 * 2 * 128 * 64;
        // Keep wider fragment reuse only when complete CTA waves add no work.
        if (cost192 <= cost96 && (cost192 < cost64 || (tokens > 1152 && cost192 == cost64)))
            return launch_bf16_tma_mma<Bulk192<Bf16MmaRaster::TokenFast>>;
        if (cost96 < cost64) return launch_bf16_tma_mma<Bulk96<Bf16MmaRaster::TokenFast>>;
        return launch_warp_projection<Bulk64>;
    }
    if (cost192 <= cost96) return launch_bf16_tma_mma<Bulk192<Bf16MmaRaster::RowFast>>;
    return launch_bf16_tma_mma<Bulk96<Bf16MmaRaster::RowFast>>;
}
} // namespace ninfer::ops::detail
