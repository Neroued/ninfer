#include "ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_plan.h"
#include "ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_kernels.h"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct Plan {
    W8DynamicConvAddSchedule schedule;
    std::size_t workspace_bytes;
};

Plan resolve_plan(int input_rows, int width, int batch) {
    if (input_rows != 4096 && input_rows != 17408)
        throw std::invalid_argument("linear dynamic grouped conv add: C must be 4096 or 17408");
    if (width < 2 || width > 16 || batch < 1 || batch > 8)
        throw std::invalid_argument("linear dynamic grouped conv add: invalid W/B profile");
    const int columns = width * batch;
    using S           = W8DynamicConvAddSchedule;
    const S schedule = columns <= 88 ? S::TiledMma : S::MmaK128;
    return {schedule, static_cast<std::size_t>(5120) * columns * sizeof(std::uint16_t)};
}
} // namespace

std::size_t w8_linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    int input_rows, int min_width, int max_width, int min_batch, int max_batch) {
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch)
        throw std::invalid_argument(
            "linear dynamic grouped conv add workspace: invalid W/B interval");
    // Every route shares one projected BF16 matrix; its capacity is monotonic in W and B.
    return resolve_plan(input_rows, max_width, max_batch).workspace_bytes;
}

const char* w8_linear_dynamic_grouped_conv_add_route_name(int input_rows, int width, int batch) {
    switch (resolve_plan(input_rows, width, batch).schedule) {
    case W8DynamicConvAddSchedule::TiledMma:
        return "dynamic_grouped_conv_add.w8.tiled_mma.materialized_bf16";
    case W8DynamicConvAddSchedule::MmaK128:
        return "dynamic_grouped_conv_add.w8.r64_c64_k128.materialized_bf16";
    }
    throw std::logic_error("linear dynamic grouped conv add: invalid production schedule");
}

void w8_linear_dynamic_grouped_conv_add_dispatch(const Tensor& x, const Weight& weight,
                                                 const Tensor& base, const Tensor& delta,
                                                 Tensor& residual, WorkspaceArena& workspace,
                                                 cudaStream_t stream) {
    const Plan plan          = resolve_plan(x.ne[0], x.ne[1], x.ne[2]);
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(plan.workspace_bytes);
    Tensor projected(storage.data, DType::BF16, {5120, x.ne[1], x.ne[2]});
    w8_dynamic_grouped_conv_add_materialized_launch(plan.schedule, x, weight, base, delta, residual,
                                                    projected, stream);
}
} // namespace ninfer::ops::detail
