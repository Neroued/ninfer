#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_tma_register_b.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 4096>;

using Gemv =
    Static<Bf16A16GemvSchedule<4, 4, 1, 16, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
template <int Tokens>
using Simt = Static<
    Bf16A16SimtSchedule<4, 4, 1, 8, 2, 4, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2, Tokens>>;
using Small  = Static<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 2>>;
using Medium = Static<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>>;
using Large  = Static<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::RowFast>>;
using Bulk   = Static<Bf16A16TmaRegisterBSchedule<64, 64, 64, 32, 32, 2, Bf16MmaRaster::RowFast>>;

void large(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Large>(
        bf16_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        Bf16MmaWarpStoreEpilogue{}, stream);
}

void bulk(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_register_b<Bulk>(
        bf16_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n128_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 4) return launch_bf16_simt<Simt<4>>;
    if (tokens <= 8) return launch_bf16_simt<Simt<8>>;
    if (tokens <= 64) return launch_bf16_sliced_k_mma<Small>;
    if (tokens <= 256) return launch_bf16_sliced_k_mma<Medium>;
    if (tokens <= 1024) return large;
    return bulk;
}
} // namespace ninfer::ops::detail
