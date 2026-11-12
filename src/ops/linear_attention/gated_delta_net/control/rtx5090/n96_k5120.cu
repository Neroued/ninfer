#include "ops/linear_attention/gated_delta_net/control/kernels/mma_launch.cuh"

#include <array>

namespace ninfer::ops::detail::gdn_control {
namespace {
constexpr Plan mma(int columns, int split, RowMode rows = RowMode::Paired) {
    return {Kernel::Mma, columns, split, rows};
}

constexpr std::array kRoutes{
    Route{1, 1, {Kernel::Gemv}},
    Route{2, 16, mma(16, 20, RowMode::Merged)},
    Route{17, 64, mma(32, 20, RowMode::Merged)},
    Route{65, 128, mma(32, 20)},
    Route{129, 1024, mma(128, 8)},
    Route{1025, 2048, mma(128, 4)},
    Route{2049, 4096, mma(128, 2)},
    Route{4097, std::numeric_limits<int>::max(), mma(128, 1)},
};
static_assert(closed(kRoutes));

template <int Columns, bool Paired>
void launch_small(const Inputs& in, GdnControlOutputs out, Scratch scratch,
                  CudaExecutionView execution) {
    using Geometry = MmaGeometry<48, 5120, Columns, Paired>;
    if (!launch_mma<Geometry, 20, Columns / 8>(in, out, scratch, execution))
        throw std::logic_error("gdn_control: RTX5090 control grid exceeds residency");
}
} // namespace

std::span<const Route> routes_n96_k5120() { return kRoutes; }

void launch_small_n96_k5120(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch scratch,
                            CudaExecutionView execution) {
    if (plan.tile_cols == 16)
        launch_small<16, false>(in, out, scratch, execution);
    else if (plan.rows == RowMode::Merged)
        launch_small<32, false>(in, out, scratch, execution);
    else
        launch_small<32, true>(in, out, scratch, execution);
}
} // namespace ninfer::ops::detail::gdn_control
