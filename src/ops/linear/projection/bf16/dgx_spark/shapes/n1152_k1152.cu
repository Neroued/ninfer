#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct Bf16VisionOutputSchedule : Base {};
template <class Base>
using Output = Bf16VisionOutputSchedule<Bf16ScheduleInstance<Base, 1152>>;
using Tiny   = Output<Bf16A16SlicedKMmaSchedule<16, 8, 3, 128, 2>>;
using Small  = Bf16VisionOutputSchedule<
     Bf16KTailSchedule<Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 2>, 1152>>>;
using Middle =
    Output<Bf16A16MmaSchedule<32, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Hot     = Output<Bf16A16MmaSchedule<32, 32, 64, 16, 8, 3, 2, Cache::cg, Cache::cg,
                                          Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Narrow  = Output<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>>;
using Medium  = Output<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>>;
using Wide96  = Output<Bf16A16TmaMmaSchedule<96, 128, 64, 32, 32, 3, 1>>;
using Wide128 = Output<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>>;
using Wide192 = Output<Bf16A16TmaMmaSchedule<192, 128, 64, 64, 32, 2, 1, Bf16MmaRaster::RowFast>>;

template <class Schedule>
void launch_vector(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(bf16_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                      Bf16TmaVectorEpilogue{}, stream);
}

template <class S>
void launch_warp_projection(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n1152_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("bf16 linear: P must be a multiple of 4 in [4,131072]");
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Tiny>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Small>;
    if (tokens <= 64) return launch_bf16_mma<Middle>;
    if (tokens <= 128) return launch_bf16_mma<Hot>;
    if (tokens <= 192) return launch_warp_projection<Narrow>;
    if (tokens <= 320) return launch_warp_projection<Medium>;
    if (tokens > 4096) return launch_warp_projection<Wide128>;
    const auto tiles   = (static_cast<std::int64_t>(tokens) + 127) / 128;
    const auto work96  = 96 * ((12 * tiles + 47) / 48);
    const auto work128 = 128 * ((9 * tiles + 47) / 48);
    const auto work192 = 192 * ((6 * tiles + 47) / 48);
    if (work192 <= work96 && work192 <= work128) return launch_vector<Wide192>;
    if (work128 < work96) return launch_warp_projection<Wide128>;
    return launch_warp_projection<Wide96>;
}
} // namespace ninfer::ops::detail
