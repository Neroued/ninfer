#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
template <class S, int Tag>
struct Bf16VisionFc1Schedule : S {};
template <class S, int Tag>
using Fc1         = Bf16VisionFc1Schedule<Bf16RowTailSchedule<Bf16ScheduleInstance<S, 1152>>, Tag>;
using Small       = Fc1<Bf16A16MmaSchedule<32, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                                           Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
                        0>;
using Hot         = Fc1<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3, 2>, 1>;
using Medium      = Fc1<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 2>;
using QuarterTail = Fc1<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 4>;
using Wide128     = Fc1<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 10>;
using Wide192     = Fc1<Bf16A16TmaMmaSchedule<192, 128, 64, 64, 32, 2, 1>, 11>;
using Large192 = Fc1<Bf16A16TmaMmaSchedule<192, 128, 64, 64, 32, 2, 1, Bf16MmaRaster::RowFast>, 12>;

template <class Schedule>
void launch_vector(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(bf16_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                      Bf16TmaRowVectorEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n4304_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("bf16 linear: P must be a multiple of 4 in [4,131072]");
    if (tokens <= 64) return launch_bf16_mma<Small>;
    if (tokens <= 128) return launch_bf16_tma_mma<Hot>;
    if (tokens <= 256) return launch_bf16_tma_mma<Medium>;
    if (tokens <= 320) return launch_bf16_tma_mma<QuarterTail>;
    if (tokens <= 384) return launch_vector<Wide128>;
    if (tokens > 4096) return launch_vector<Large192>;
    const auto tiles   = (static_cast<std::int64_t>(tokens) + 127) / 128;
    const auto work128 = 128 * ((34 * tiles + 47) / 48);
    const auto work192 = 192 * ((23 * tiles + 47) / 48);
    if (work192 <= work128) return launch_vector<Wide192>;
    return launch_vector<Wide128>;
}
} // namespace ninfer::ops::detail
