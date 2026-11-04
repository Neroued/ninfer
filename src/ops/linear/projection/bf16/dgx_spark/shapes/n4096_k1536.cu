#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_tma_register_b.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 1536>;

using Gemv =
    Static<Bf16A16GemvSchedule<4, 1, 2, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Short = Static<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 1>>;
using Small = Static<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 1>>;
using Medium =
    Static<Bf16A16MmaSchedule<32, 32, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::RowFast>>;
using Large  = Static<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3, 2, Bf16MmaRaster::RowFast>>;
using Larger = Static<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::RowFast>>;
using Bulk   = Static<Bf16A16TmaRegisterBSchedule<128, 128, 64, 64, 32, 2, Bf16MmaRaster::RowFast>>;

template <class Schedule>
void tma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(
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

Bf16Launch select_bf16_n4096_k1536(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Short>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Small>;
    if (tokens <= 32) return launch_bf16_mma<Medium>;
    if (tokens <= 64) return tma<Large>;
    if (tokens <= 128) return tma<Larger>;
    return bulk;
}
} // namespace ninfer::ops::detail
