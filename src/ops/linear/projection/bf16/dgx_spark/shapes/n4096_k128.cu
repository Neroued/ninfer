#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct Bf16N4096K128Schedule : Base {};
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 128>;

using Small =
    Static<Bf16A16MmaSchedule<64, 32, 64, 16, 16, 2, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Medium = Bf16N4096K128Schedule<Static<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2>>>;
using Bulk   = Bf16N4096K128Schedule<
      Static<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 2, Bf16MmaRaster::Grouped, 8, 3>>>;

template <class Schedule>
void vector_store(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(
        bf16_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, Bf16TmaVectorEpilogue{},
        stream);
}
} // namespace

Bf16Launch select_bf16_n4096_k128(std::int32_t tokens) {
    if (tokens <= 128) return launch_bf16_mma<Small>;
    if (tokens < 4096) return vector_store<Medium>;
    return vector_store<Bulk>;
}
} // namespace ninfer::ops::detail
