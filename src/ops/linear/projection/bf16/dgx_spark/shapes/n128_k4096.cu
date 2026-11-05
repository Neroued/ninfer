#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 4096>;

template <class Schedule>
void tma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(
        bf16_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n128_k4096(std::int32_t tokens) {
    if (tokens <= 1)
        return launch_bf16_gemv<Static<
            Bf16A16GemvSchedule<4, 4, 1, 16, 4, Bf16ActivationAccess::Direct,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>>;
    if (tokens <= 4)
        return launch_bf16_simt<Static<Bf16A16SimtSchedule<
            4, 4, 1, 8, 2, 4, Bf16SimtActivationAccess::WarpPacked, Bf16WeightCache::Default,
            Bf16PhaseOrder::Sequential, 1, 2, 1, 2, 2>>>;
    if (tokens <= 8)
        return launch_bf16_simt<Static<Bf16A16SimtSchedule<
            4, 4, 1, 8, 2, 4, Bf16SimtActivationAccess::WarpPacked, Bf16WeightCache::Default,
            Bf16PhaseOrder::Sequential, 1, 2, 1, 2, 8>>>;
    if (tokens <= 48)
        return launch_bf16_sliced_k_mma<Static<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>>>;
    if (tokens <= 96)
        return launch_bf16_sliced_k_mma<Static<Bf16A16SlicedKMmaSchedule<16, 16, 4, 128, 2>>>;
    if (tokens <= 128)
        return launch_bf16_sliced_k_mma<Static<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>>>;
    if (tokens <= 192)
        return launch_bf16_sliced_k_mma<Static<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>>>;
    if (tokens <= 256)
        return launch_bf16_sliced_k_mma<Static<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 2>>>;
    if (tokens <= 768)
        return tma<
            Static<Bf16A16TmaMmaSchedule<32, 32, 128, 16, 16, 3, 2, Bf16MmaRaster::RowFast>>>;
    if (tokens <= 1152)
        return tma<Static<Bf16A16TmaMmaSchedule<32, 32, 64, 16, 16, 3, 1, Bf16MmaRaster::RowFast>>>;
    return tma<Static<Bf16A16TmaMmaSchedule<64, 24, 64, 32, 8, 2, 4, Bf16MmaRaster::RowFast>>>;
}
} // namespace ninfer::ops::detail
