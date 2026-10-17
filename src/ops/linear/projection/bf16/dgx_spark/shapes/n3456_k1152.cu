#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct Bf16VisionQkvSchedule : Base {};
template <class Base>
using Qkv     = Bf16VisionQkvSchedule<Bf16ScheduleInstance<Base, 1152>>;
using Tiny    = Qkv<Bf16A16SlicedKMmaSchedule<16, 8, 6, 64, 2>>;
using Small   = Qkv<Bf16A16MmaSchedule<32, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Hot     = Qkv<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>>;
using Wide96  = Qkv<Bf16A16TmaMmaSchedule<96, 128, 64, 32, 32, 3, 1>>;
using Wide144 = Qkv<Bf16A16TmaMmaSchedule<144, 128, 64, 48, 32, 2, 1>>;
using Wide192 = Qkv<Bf16A16TmaMmaSchedule<192, 128, 64, 64, 32, 2, 1>>;
using Large   = Qkv<Bf16A16TmaMmaSchedule<192, 128, 64, 64, 32, 2, 1, Bf16MmaRaster::RowFast>>;

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

Bf16Launch select_bf16_n3456_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("bf16 linear: P must be a multiple of 4 in [4,131072]");
    if (tokens <= 4) return launch_bf16_sliced_k_mma<Tiny>;
    if (tokens <= 64) return launch_bf16_mma<Small>;
    if (tokens <= 96) return launch_bf16_tma_mma<Hot>;
    if (tokens <= 128) return launch_vector<Wide96>;
    if (tokens > 4096) return launch_warp_projection<Large>;
    // Compare full 48-SM waves; retain the deeper R96 pipeline when it ties R144.
    const auto tiles   = (static_cast<std::int64_t>(tokens) + 127) / 128;
    const auto work96  = 96 * ((36 * tiles + 47) / 48);
    const auto work144 = 144 * ((24 * tiles + 47) / 48);
    const auto work192 = 192 * ((18 * tiles + 47) / 48);
    if (work192 <= work96 && work192 <= work144) return launch_vector<Wide192>;
    if (work144 < work96) return launch_warp_projection<Wide144>;
    // The fragment store also covers the former scalar window after T128.
    return launch_warp_projection<Wide96>;
}
} // namespace ninfer::ops::detail
