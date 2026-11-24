#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "core/weight.h"
#include "core/device.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include "ops/linear/kernels/bf16/bf16_instances.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {

void launch_decode(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                   cudaStream_t stream) {
    using Schedule = Bf16ScheduleInstance<
        Bf16A16GemvSchedule<4, 1, 8, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                            Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>,
        5120>;
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        {static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data),
         static_cast<__nv_bfloat16*>(outputs[2].data),
         static_cast<__nv_bfloat16*>(outputs[3].data)}};
    launch_bf16_a16_gemv<Schedule>(bf16_a16_operands(x, weight), output, LinearIdentityEpilogue{},
                                   stream);
}

using Launch = void (*)(const Tensor&, const Weight&, std::span<const Tensor>, cudaStream_t);

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                  cudaStream_t stream) {
    static_assert(ActiveTokens >= 2 && ActiveTokens <= 4);
    using Schedule = Bf16A16SimtSchedule<4, 1, 8, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                         Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1,
                                         ActiveTokens == 4 ? 2 : 1, 1, 2>;
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        {static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data),
         static_cast<__nv_bfloat16*>(outputs[2].data),
         static_cast<__nv_bfloat16*>(outputs[3].data)}};
    launch_bf16_a16_simt<Bf16ScheduleInstance<Schedule, 5120, ActiveTokens, true>>(
        bf16_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{&launch_exact<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<3>{});

void launch_small(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                  cudaStream_t stream) {
    kLaunchers[x.ne[1] - 2](x, weight, outputs, stream);
}

void launch_matrix(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                   cudaStream_t stream) {
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        {static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data),
         static_cast<__nv_bfloat16*>(outputs[2].data),
         static_cast<__nv_bfloat16*>(outputs[3].data)}};

    const auto p   = bf16_a16_operands(x, weight);
    const auto mma = [&]<class S>() {
        launch_bf16_a16_mma<Bf16ScheduleInstance<S, 5120>>(p, output, LinearIdentityEpilogue{},
                                                           stream);
    };
    const auto tma = [&]<class S>() {
        launch_bf16_a16_tma_mma<Bf16ScheduleInstance<S, 5120>>(p, output, LinearIdentityEpilogue{},
                                                               stream);
    };
    if (p.tokens <= 16)
        launch_bf16_a16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, 5120>>(
            p, output, LinearIdentityEpilogue{}, stream);
    else if (p.tokens <= 32)
        mma.template operator()<Bf16A16MmaR32T32K128S3>();
    else if (p.tokens <= 64)
        tma.template operator()<Bf16A16TmaR64T64K64S3>();
    else if (p.tokens <= 96)
        mma.template operator()<Bf16A16MmaR64T32K64S3>();
    else if (p.tokens <= 128)
        tma.template operator()<Bf16A16TmaR64T128K64S2>();
    else if (p.tokens <= 192)
        tma.template operator()<Bf16A16TmaR64T64K64S3>();
    else
        tma.template operator()<Bf16A16TmaR64T128K64S2>();
}

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> outputs, MemoryArena*,
                  cudaStream_t stream) {
    if (x.ne[1] == 1)
        launch_decode(x, weights.first, outputs, stream);
    else if (x.ne[1] <= 4)
        launch_small(x, weights.first, outputs, stream);
    else
        launch_matrix(x, weights.first, outputs, stream);
}
} // namespace

const SplitShape kBf16SplitN14336K5120{14336, 5120, kSplitRows<6144, 1024, 6144, 1024>,
                                       launch_split};
} // namespace ninfer::ops::detail::linear_split
