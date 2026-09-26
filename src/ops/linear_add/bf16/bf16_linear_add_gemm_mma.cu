#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_template_launch.cuh"
#include "ops/linear_add/bf16/bf16_linear_add_plan.h"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
void launch(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    launch_bf16_a16_mma<Bf16ScheduleInstance<Schedule, 6144>>(
        bf16_a16_operands(x, weight), LinearBf16Output{data, weight.n},
        LinearResidualAddEpilogue{{data, weight.n}}, stream);
}
} // namespace

void bf16_linear_add_aggregate_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                          cudaStream_t stream) {
    using UpTo32 = Bf16A16MmaR32T32K256S3;
    using UpTo48 = Bf16A16MmaR32T32K192S2;

    if (x.ne[1] <= 32)
        launch<UpTo32>(x, weight, residual, stream);
    else
        launch<UpTo48>(x, weight, residual, stream);
}

void bf16_linear_add_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                cudaStream_t stream) {
    auto* data     = static_cast<__nv_bfloat16*>(residual.data);
    const auto p   = bf16_a16_operands(x, weight);
    const auto tma = [&]<class S>() {
        launch_bf16_a16_tma_mma<Bf16ScheduleInstance<S, 6144>>(
            p, LinearBf16Output{data, weight.n}, LinearResidualAddEpilogue{{data, weight.n}},
            stream);
    };
    if (p.tokens <= 96)
        tma.template operator()<Bf16A16TmaR64T32K64S3>();
    else if (p.tokens <= 128)
        tma.template operator()<Bf16A16TmaR64T64K128S2>();
    else if (p.tokens <= 256)
        tma.template operator()<Bf16A16TmaR64T64K64S3>();
    else
        tma.template operator()<Bf16A16TmaR64T128K64S2>();
}
} // namespace ninfer::ops::detail
