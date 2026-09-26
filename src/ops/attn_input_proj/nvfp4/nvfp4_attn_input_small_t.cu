#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);

// The four-output epilogue shifts the measured low-T warp crossover relative to contiguous Linear,
// so Attention owns this production mapping even though both routes share the compute body.
template <int ActiveTokens>
struct Nvfp4AttentionSmallTProductionSchedule {
    static_assert(ActiveTokens >= 2);
    static_assert(ActiveTokens <= 32);
    static constexpr int kWarpsPerCta       = ActiveTokens >= 17 ? 4 : (ActiveTokens >= 8 ? 16 : 8);
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SimtActivationAccess::SharedPhase
                                                  : Nvfp4SimtActivationAccess::TokenPacked;
    using Type =
        Nvfp4A16SimtSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                             Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                             Nvfp4SimtBlockOrder::RowsContiguous, 1>;
};

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                  Tensor& v, cudaStream_t stream) {
    using Geometry = Nvfp4N14336K5120;
    using Schedule = typename Nvfp4AttentionSmallTProductionSchedule<ActiveTokens>::Type;
    launch_nvfp4_a16_simt<
        Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        nvfp4_a16_operands(x, weight),
        Nvfp4AttentionInputOutput{
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)},
        LinearIdentityEpilogue{}, stream);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{&launch_exact<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<2 - 2 + 1>{});

} // namespace

void nvfp4_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    kLaunchers[x.ne[1] - 2](x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
