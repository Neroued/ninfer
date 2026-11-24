#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/dynamic_grouped_conv/prepare/launch.h"

#include <algorithm>

namespace ninfer::ops::detail::dynamic_grouped_conv::prepare {
namespace {
Plan select_plan(int width, int batch) {
    const int columns = width * batch;
    // Split-8 R16 wins through 48 columns; R32 uses half the split scratch above it.
    const int tile_columns      = columns <= 8 ? 8 : columns <= 16 ? 16 : columns <= 32 ? 32 : 48;
    const Projection projection = columns <= 48 ? Projection{16, tile_columns, 8}
                                                : Projection{32, columns <= 96 ? 32 : 64, 4};
    return {projection, width <= 8 ? 8 : 16,
            std::size_t(kCoefficientRows) * columns * projection.splits * sizeof(float)};
}
} // namespace
} // namespace ninfer::ops::detail::dynamic_grouped_conv::prepare

namespace ninfer::ops {
namespace dc = detail::dynamic_grouped_conv;

std::size_t rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(int min_width,
                                                                          int max_width,
                                                                          int min_batch,
                                                                          int max_batch) {
    dc::validate_interval(min_width, max_width, min_batch, max_batch, dc::kPrepareOp);
    std::size_t maximum = 0;
    for (int width = min_width; width <= max_width; ++width)
        for (int batch = min_batch; batch <= max_batch; ++batch)
            maximum = std::max(maximum, dc::prepare::select_plan(width, batch).workspace_bytes);
    return maximum;
}

void rmsnorm_dynamic_grouped_conv_prepare(const Tensor& residual, const Tensor& norm_weight,
                                          float epsilon, const Tensor& base_kernel,
                                          const Weight& projection, Tensor& prepared,
                                          Tensor& finish_delta, MemoryArena& workspace,
                                          CudaExecutionView execution) {
    const auto stream = execution.stream();
    dc::validate_prepare(residual, norm_weight, epsilon, base_kernel, projection, prepared,
                         finish_delta, workspace, execution);
    const auto plan = dc::prepare::select_plan(residual.ne[1], residual.ne[2]);
    auto scope      = workspace.scope();
    auto* partials  = static_cast<float*>(workspace.alloc_bytes(plan.workspace_bytes).data);
    rmsnorm(residual, norm_weight, epsilon, false, prepared, execution);
    dc::prepare::launch(plan, base_kernel, projection, partials, prepared, finish_delta, stream);
}
} // namespace ninfer::ops
