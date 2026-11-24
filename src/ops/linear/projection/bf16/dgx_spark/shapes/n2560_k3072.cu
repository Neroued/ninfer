#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv =
    Bf16ScheduleInstance<Bf16KTailSchedule<Bf16A16GemvSchedule<
                             8, 4, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                             Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>,
                         3072>;
using Sliced = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>, 3072>;
using Medium = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 2>, 3072>;
using Hot128 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 128, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    3072>;
using HotTma = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 3072>;
using Bulk   = Bf16ScheduleInstance<
      Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 12, 3>, 3072>;
using Wide = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<160, 192, 64, 80, 48, 2, 1, Bf16MmaRaster::Grouped, 8>, 3072>;
using Large = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1, Bf16MmaRaster::Grouped, 6, 1>, 3072>;

void vector_store(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Bulk>(bf16_a16_operands(x, w),
                                  LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                                  Bf16TmaVectorEpilogue{}, stream);
}

template <class Schedule>
void warp_store(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(bf16_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                                      Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n2560_k3072(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Sliced>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<Medium>;
    if (tokens <= 64) return launch_bf16_mma<Hot128>;
    if (tokens <= 128) return launch_bf16_tma_mma<HotTma>;
    if (tokens >= 4096) return warp_store<Large>;
    if ((tokens > 256 && tokens <= 576) || (tokens > 896 && tokens <= 1152))
        return warp_store<Wide>;
    return vector_store;
}
} // namespace ninfer::ops::detail
