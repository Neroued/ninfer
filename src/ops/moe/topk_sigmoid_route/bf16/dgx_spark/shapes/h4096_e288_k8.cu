#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_instances.cuh"
#include "ops/moe/topk_sigmoid_route/dispatch.h"
#include "ops/moe/topk_sigmoid_route/topk.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 4096>;
template <class Schedule>
using Tail = Static<Bf16KTailSchedule<Schedule>>;

using Gemv =
    Tail<Bf16A16GemvSchedule<4, 4, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                             Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Short  = Tail<Bf16A16SlicedKMmaSchedule<16, 8, 16, 64, 2>>;
using Hot    = Tail<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>>;
using Medium = Tail<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 2>>;
using Mma16 =
    Static<Bf16A16MmaSchedule<48, 16, 128, 16, 8, 3, 1, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Mma32 =
    Static<Bf16A16MmaSchedule<48, 32, 256, 16, 8, 2, 1, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Tma32 = Static<Bf16A16TmaMmaSchedule<48, 32, 128, 16, 16, 2, 2, Bf16MmaRaster::RowFast>>;
using Tma64 = Static<Bf16A16TmaMmaSchedule<48, 64, 64, 16, 32, 2, 2, Bf16MmaRaster::RowFast>>;

struct RouterLogitsOutput {
    float* data;

    __device__ void store(int row, int token, float value) const {
        data[std::int64_t(token) * 288 + row] = value;
    }
};
} // namespace

void moe_sigmoid::launch(const Tensor& x, const Weight& w, const Tensor& bias, ExpertRoutes& routes,
                         const SigmoidRouteProfile& profile, float* scores, cudaStream_t stream) {
    const auto p = bf16_a16_operands(x, w);
    const RouterLogitsOutput out{scores};
    const LinearIdentityEpilogue epi{};
    if (p.tokens == 1)
        launch_bf16_a16_gemv<Gemv>(p, out, epi, stream);
    else if (p.tokens <= 16)
        launch_bf16_a16_sliced_k_mma<Short>(p, out, epi, stream);
    else if (p.tokens <= 32)
        launch_bf16_a16_sliced_k_mma<Hot>(p, out, epi, stream);
    else if (p.tokens <= 64)
        launch_bf16_a16_sliced_k_mma<Medium>(p, out, epi, stream);
    else if (p.tokens <= 128)
        launch_bf16_a16_mma<Mma16>(p, out, epi, stream);
    else if (p.tokens <= 256)
        launch_bf16_a16_mma<Mma32>(p, out, epi, stream);
    else if (p.tokens <= 512)
        launch_bf16_a16_tma_mma<Tma32>(p, out, epi, stream);
    else
        launch_bf16_a16_tma_mma<Tma64>(p, out, epi, stream);
    const auto select = [&]<int Warps>() {
        moe_cuda::routing::sigmoid_correction_top8<Warps>
            <<<(p.tokens + Warps - 1) / Warps, Warps * 32, 0, stream>>>(
                scores, static_cast<const float*>(bias.data), static_cast<int*>(routes.ids.data),
                static_cast<float*>(routes.coefficients.data), p.tokens, profile.scaling_factor,
                profile.normalization_epsilon);
    };
    if (p.tokens < 512)
        select.template operator()<4>();
    else
        select.template operator()<8>();
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
