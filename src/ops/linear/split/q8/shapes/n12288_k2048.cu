#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/q8/q8_output_launch.cuh"
#include <utility>

namespace ninfer::ops::detail::linear_split {
namespace {
using Output = LinearBf16SegmentedOutput<8192, 4096>;
using Routes = Q8A16OutputLaunch<Output>;

template <int Columns>
void launch_exact(const Routes& routes) {
    constexpr int capacity   = (Columns + 7) / 8 * 8;
    constexpr int min_blocks = capacity == 8 ? 5 : capacity == 16 ? 4 : capacity == 24 ? 3 : 2;
    constexpr auto scales    = Columns > 4 ? Q8ScaleAccess::Shared : Q8ScaleAccess::Direct;
    using Schedule           = Q8A16SlicedKMmaSchedule<capacity, 8, 1, min_blocks, scales>;
    routes.template sliced<typename Schedule::template with_problem<2048, Columns, true>>();
}

template <std::size_t... I>
constexpr auto exact_launchers(std::index_sequence<I...>) {
    return std::array{&launch_exact<2 + static_cast<int>(I)>...};
}

constexpr auto kExact = exact_launchers(std::make_index_sequence<31>{});

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    const Output output{static_cast<__nv_bfloat16*>(out[0].data),
                        static_cast<__nv_bfloat16*>(out[1].data)};
    const Routes routes{q8_linear_operands(x, weights.first), output, stream};
    const int t = x.ne[1];
    if (t == 1) return routes.template gemv<Q8A16GemvSchedule<8, 1, 2, 2048>>();
    if (t <= 32) return kExact[t - 2](routes);

    if (t <= 48)
        return routes.template grouped<Q8A16GroupedSlicedKMmaSchedule<48, 4, 2, 1, 3, 2048>>();
    if (t <= 64)
        return routes.template grouped<Q8A16GroupedSlicedKMmaSchedule<64, 4, 2, 1, 2, 2048>>();
    if (t <= 96)
        return routes.template grouped<Q8A16GroupedSlicedKMmaSchedule<96, 2, 4, 1, 3, 2048>>();

    routes.template mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>();
}
} // namespace

const SplitShape kQ8SplitN12288K2048{12288, 2048, kSplitRows<8192, 4096>, launch_split};
} // namespace ninfer::ops::detail::linear_split
