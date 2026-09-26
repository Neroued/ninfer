#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

#include <array>
#include <cstddef>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                  cudaStream_t stream) {
    using Geometry = Nvfp4N16384K5120;
    using Schedule =
        Nvfp4A16SimtSchedule<4, 1, 2, (ActiveTokens >= 17 && ActiveTokens <= 20) ? 8 : 16,
                             ActiveTokens, 1,
                             ActiveTokens == 2 ? Nvfp4SimtActivationAccess::SharedPhase
                                               : Nvfp4SimtActivationAccess::TokenPacked,
                             Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                             Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    launch_nvfp4_a16_simt<
        Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        nvfp4_a16_operands(x, weight),
        Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(qkv.data),
                            static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{&launch_exact<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<2 - 2 + 1>{});

} // namespace

void nvfp4_gdn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream) {
    kLaunchers[x.ne[1] - 2](x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
