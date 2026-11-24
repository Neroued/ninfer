#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ops/dynamic_grouped_conv/add/launch.h"

namespace ninfer::ops::detail::dynamic_grouped_conv::add {
namespace {
Plan select_plan(int width, int batch) {
    const int columns = width * batch;
    return {columns <= 88 ? Projection::TiledMma : Projection::MmaK128, (columns - 1) / 8,
            std::size_t(kHidden) * columns * sizeof(std::uint16_t)};
}
} // namespace
} // namespace ninfer::ops::detail::dynamic_grouped_conv::add

namespace ninfer::ops {
namespace dc = detail::dynamic_grouped_conv;

std::size_t linear_dynamic_grouped_conv_add_workspace_capacity_bytes(int input_rows, int min_width,
                                                                     int max_width, int min_batch,
                                                                     int max_batch) {
    dc::validate_interval(min_width, max_width, min_batch, max_batch, dc::kAddOp);
    dc::validate_input_rows(input_rows);
    // Both routes store the same BF16 projection; capacity is monotonic in W and B.
    return dc::add::select_plan(max_width, max_batch).workspace_bytes;
}

void linear_dynamic_grouped_conv_add(const Tensor& x, const Weight& projection,
                                     const Tensor& base_kernel, const Tensor& finish_delta,
                                     Tensor& residual, MemoryArena& workspace,
                                     CudaExecutionView execution) {
    const auto stream = execution.stream();
    dc::validate_add(x, projection, base_kernel, finish_delta, residual, workspace, execution);
    const auto plan    = dc::add::select_plan(x.ne[1], x.ne[2]);
    auto scope         = workspace.scope();
    const auto storage = workspace.alloc_bytes(plan.workspace_bytes);
    Tensor projected(storage, DType::BF16, {dc::kHidden, x.ne[1], x.ne[2]});
    dc::add::launch(plan, x, projection, base_kernel, finish_delta, residual, projected, stream);
}
} // namespace ninfer::ops
