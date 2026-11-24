#include "ops/linear/add/dispatch.h"
#include "ops/linear/kernels/bf16/bf16_instances.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"

namespace ninfer::ops::detail {
namespace {
void launch_decode(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule = Bf16ScheduleInstance<
        Bf16A16GemvSchedule<8, 2, 2, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                            Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>,
        6144>;
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    launch_bf16_a16_gemv<Schedule>(bf16_a16_operands(x, weight), LinearBf16Output{data, weight.n},
                                   LinearResidualAddEpilogue{{data, weight.n}}, stream);
}

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    static_assert(ActiveTokens >= 2 && ActiveTokens <= 4);
    using Schedule =
        Bf16A16SimtSchedule<4, 1, ActiveTokens == 4 ? 2 : 4, ActiveTokens == 3 ? 16 : 8, 1, 4,
                            Bf16SimtActivationAccess::WarpPacked, Bf16WeightCache::Default,
                            Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    launch_bf16_a16_simt<Bf16ScheduleInstance<Schedule, 6144, ActiveTokens, true>>(
        bf16_a16_operands(x, weight), LinearBf16Output{data, weight.n},
        LinearResidualAddEpilogue{{data, weight.n}}, stream);
}

template <class Schedule>
void launch_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(out.data);
    launch_bf16_a16_mma<Bf16ScheduleInstance<Schedule, 6144>>(
        bf16_a16_operands(x, w), LinearBf16Output{data, w.n},
        LinearResidualAddEpilogue{{data, w.n}}, stream);
}

template <class Schedule>
void launch_tma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(out.data);
    launch_bf16_a16_tma_mma<Bf16ScheduleInstance<Schedule, 6144>>(
        bf16_a16_operands(x, w), LinearBf16Output{data, w.n},
        LinearResidualAddEpilogue{{data, w.n}}, stream);
}
} // namespace

LinearAddLaunch select_bf16_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    if (rows != 5120 || k != 6144 || tokens <= 0)
        throw std::invalid_argument("bf16 linear_add: unsupported shape or token extent");
    if (tokens == 1) return launch_decode;
    if (tokens == 2) return launch_exact<2>;
    if (tokens == 3) return launch_exact<3>;
    if (tokens == 4) return launch_exact<4>;
    if (tokens <= 32) return launch_mma<Bf16A16MmaR32T32K256S3>;
    if (tokens <= 64) return launch_mma<Bf16A16MmaR32T32K192S2>;
    if (tokens <= 96) return launch_tma<Bf16A16TmaR64T32K64S3>;
    if (tokens <= 128) return launch_tma<Bf16A16TmaR64T64K128S2>;
    if (tokens <= 256) return launch_tma<Bf16A16TmaR64T64K64S3>;
    return launch_tma<Bf16A16TmaR64T128K64S2>;
}
} // namespace ninfer::ops::detail
