#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N5120K25600;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C8 =
    Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C16 =
    Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24 =
    Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C32 =
    Q8A16SlicedKMmaSchedule<32, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C40 =
    Q8A16SlicedKMmaSchedule<40, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C48 =
    Q8A16SlicedKMmaSchedule<48, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C56 =
    Q8A16SlicedKMmaSchedule<56, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;

template <int Rows>
void launch_tiled(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // Retain the predicated variant on complete tiles: the Full variant regresses T=64.
    using Schedule = Q8A16MmaSchedule<Rows, 64, 128, 16, 16, 1, 1, Q8MmaFragmentPipeline::PingPong,
                                      Cache::cg, Cache::cg, Cache::ca, true>;
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                                stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

Q8Launch select_q8_n5120_k25600(std::int32_t tokens) {
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_a16_sliced<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_a16_sliced<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_a16_sliced<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_a16_sliced<Geometry, 48, C48>;
    if (tokens <= 56) return launch_q8_a16_sliced<Geometry, 56, C56>;
    if (tokens <= 64) return launch_tiled<16>;
    if (tokens <= 128) return launch_tiled<32>;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
