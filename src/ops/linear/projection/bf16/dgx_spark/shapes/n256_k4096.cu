#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 4096>;

using Gemv =
    Static<Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Short = Static<
    Bf16A16SimtSchedule<4, 4, 1, 8, 2, 4, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2, 4>>;
using Small  = Static<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 2>>;
using Medium = Static<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>>;
using Transition =
    Static<Bf16A16MmaSchedule<16, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::RowFast>>;
using Large = Static<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::RowFast>>;
using Bulk  = Static<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::RowFast>>;

template <class Schedule>
void tma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(
        bf16_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n256_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 4) return launch_bf16_simt<Short>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Small>;
    if (tokens <= 128) return launch_bf16_sliced_k_mma<Medium>;
    if (tokens <= 240) return launch_bf16_mma<Transition>;
    if (tokens <= 1024) return tma<Large>;
    return tma<Bulk>;
}
} // namespace ninfer::ops::detail
