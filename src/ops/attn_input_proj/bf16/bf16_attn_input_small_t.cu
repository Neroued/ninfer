#include "core/weight.h"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_template_launch.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                  Tensor& v, cudaStream_t stream) {
    static_assert(ActiveTokens >= 2 && ActiveTokens <= 4);
    using Schedule = Bf16A16SimtSchedule<4, 1, 8, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                         Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1,
                                         ActiveTokens == 4 ? 2 : 1, 1, 2>;
    const LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output{
        {static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
         static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)}};
    launch_bf16_a16_simt<Bf16ScheduleInstance<Schedule, 5120, ActiveTokens, true>>(
        bf16_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kBf16AttnInputSmallTMinTokens + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(
    std::make_index_sequence<kBf16AttnInputSmallTMaxTokens - kBf16AttnInputSmallTMinTokens + 1>{});

} // namespace

void bf16_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kBf16AttnInputSmallTMinTokens || x.ne[1] > kBf16AttnInputSmallTMaxTokens) {
        throw std::invalid_argument("bf16 attn_input_proj small-T requires T in [2,4]");
    }
    kLaunchers[x.ne[1] - kBf16AttnInputSmallTMinTokens](x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
