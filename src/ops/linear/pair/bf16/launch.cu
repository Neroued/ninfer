#include "ops/linear/pair/dispatch.h"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_instances.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using Schedule = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 32, 64, 16, 16, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    128>;

template <bool Full>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void pair_kernel(
    Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya, LinearBf16Output yb) {
    const auto& p  = blockIdx.y ? b : a;
    const auto out = blockIdx.y ? yb : ya;
    bf16_a16_mma_body<Schedule, Full>(p.x, p.weight, out, LinearIdentityEpilogue{}, p.rows, p.k, 0,
                                      p.tokens);
}
} // namespace

void launch_bf16_pair(const Tensor& x0, const Tensor& x1, const Weight& w0, const Weight& w1,
                      Tensor& y0, Tensor& y1, cudaStream_t stream) {
    const auto a = bf16_a16_operands(x0, w0), b = bf16_a16_operands(x1, w1);
    if (a.tokens > 128) {
        const auto run = [&]<class S>() {
            launch_bf16_a16_tma_mma<S>(a,
                                       LinearBf16Output{static_cast<__nv_bfloat16*>(y0.data), w0.n},
                                       Bf16TmaVectorEpilogue{}, stream);
            launch_bf16_a16_tma_mma<S>(b,
                                       LinearBf16Output{static_cast<__nv_bfloat16*>(y1.data), w1.n},
                                       Bf16TmaVectorEpilogue{}, stream);
        };
        if (a.tokens < 4096)
            run.template operator()<
                Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2>, 128>>();
        else
            run.template operator()<Bf16ScheduleInstance<
                Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 2, Bf16MmaRaster::Grouped, 8, 3>,
                128>>();
        return;
    }
    const dim3 grid(
        a.rows / Schedule::kBlockRows * common::div_up(a.tokens, Schedule::kBlockTokens), 2);
    const auto launch = [&]<bool Full>() {
        constexpr auto kernel = pair_kernel<Full>;
        const auto bytes =
            common::prepare_shared<linear_mma_shared_bytes<Schedule, LinearIdentityEpilogue>,
                                   kernel, true>();
        kernel<<<grid, Schedule::kThreads, bytes, stream>>>(
            a, b, {static_cast<__nv_bfloat16*>(y0.data), w0.n},
            {static_cast<__nv_bfloat16*>(y1.data), w1.n});
        CUDA_CHECK(cudaGetLastError());
    };
    if (a.tokens % Schedule::kBlockTokens == 0)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}
} // namespace ninfer::ops::detail
