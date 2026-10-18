#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using Small = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<4, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 8>,
    2560>;
using Sliced = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>, 2560>;
using Hot256 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using Hot128 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 128, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using Medium  = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 2560>;
using Bulk    = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 2560>;
using Window  = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<160, 192, 64, 80, 48, 2, 1>, 2560>;
using Prefill = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1>, 2560>;
using Wide    = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1>, 2560>;
using Large   = Bf16ScheduleInstance<
      Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1, Bf16MmaRaster::Grouped, 6, 1>, 2560>;

template <class Schedule>
void warp_store(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(bf16_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                                      Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n124160_k2560(std::int32_t tokens) {
    if (tokens <= 8) return launch_bf16_simt<Small>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Sliced>;
    if (tokens <= 96) return launch_bf16_mma<Hot256>;
    if (tokens <= 128) return launch_bf16_mma<Hot128>;
    if (tokens <= 256) return launch_bf16_tma_mma<Medium>;
    if (tokens >= 3584) return warp_store<Large>;
    if ((tokens > 512 && tokens <= 576) || (tokens > 1024 && tokens <= 1152))
        return launch_bf16_tma_mma<Window>;
    const int tail = tokens % 256;
    if (tokens >= 385 && tokens <= 2560 && (tail == 0 || tail > 128)) return warp_store<Wide>;
    if (tokens < 512) return launch_bf16_tma_mma<Bulk>;
    return launch_bf16_tma_mma<Prefill>;
}
} // namespace ninfer::ops::detail
