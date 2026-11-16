#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/dynamic_grouped_conv/prepare/launch.h"
#include "core/platform.h"

#include <algorithm>

namespace ninfer::ops::detail::dynamic_grouped_conv::prepare {
namespace {
constexpr int kFusedColumns = 256;

Plan select_plan(PrepareGeometry geometry, int width, int batch) {
    const int columns        = width * batch;
    const auto inverse_bytes = std::size_t(columns) * sizeof(float);
#if NINFER_TARGET_SPARK
    if (geometry.hidden == 4096) {
        const auto bytes = std::size_t(geometry.hidden) * columns * sizeof(std::uint16_t);
        return {{}, 0, bytes + inverse_bytes, bytes, true};
    }
#else
    if (columns >= kFusedColumns) {
        const auto bytes = std::size_t(geometry.hidden) * columns * sizeof(std::uint16_t);
        return {{}, 0, bytes + inverse_bytes, bytes, true};
    }
#endif
    // Split-8 R16 wins through 48 columns; R32 uses half the split scratch above it.
    const int tile_columns      = columns <= 8 ? 8 : columns <= 16 ? 16 : columns <= 32 ? 32 : 48;
    const Projection projection = columns <= 48 ? Projection{16, tile_columns, 8}
                                                : Projection{32, columns <= 96 ? 32 : 64, 4};
    const auto bytes =
        std::size_t(geometry.coefficient_rows()) * columns * projection.splits * sizeof(float);
    return {projection, width <= 8 ? 8 : 16, bytes + inverse_bytes, bytes, false};
}
} // namespace
} // namespace ninfer::ops::detail::dynamic_grouped_conv::prepare

namespace ninfer::ops {
namespace dc = detail::dynamic_grouped_conv;

std::size_t rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(int hidden, int min_width,
                                                                          int max_width,
                                                                          int min_batch,
                                                                          int max_batch) {
    dc::validate_interval(min_width, max_width, min_batch, max_batch, dc::kPrepareOp);
    const auto geometry = dc::prepare_geometry(hidden);
    std::size_t maximum = 0;
    for (int width = min_width; width <= max_width; ++width) {
        maximum =
            std::max(maximum, dc::prepare::select_plan(geometry, width, max_batch).workspace_bytes);
        // Check the final point before every scratch-reducing route change.
        for (const int last_column : {48, dc::prepare::kFusedColumns - 1}) {
            const int last_batch = std::min(max_batch, last_column / width);
            if (last_batch >= min_batch)
                maximum = std::max(
                    maximum, dc::prepare::select_plan(geometry, width, last_batch).workspace_bytes);
        }
    }
    return maximum;
}

void rmsnorm_dynamic_grouped_conv_prepare(const Tensor& residual, const Tensor& norm_weight,
                                          float epsilon, const Tensor& base_kernel,
                                          const Weight& projection, Tensor& prepared,
                                          Tensor& finish_delta, MemoryArena& workspace,
                                          CudaExecutionView execution) {
    const auto stream = execution.stream();
    const auto geometry =
        dc::validate_prepare(residual, norm_weight, epsilon, base_kernel, projection, prepared,
                             finish_delta, workspace, execution);
    const auto plan    = dc::prepare::select_plan(geometry, residual.ne[1], residual.ne[2]);
    auto scope         = workspace.scope();
    const auto storage = workspace.alloc_bytes(plan.workspace_bytes);
    Tensor inverse(storage.subspan(plan.inverse_offset), DType::FP32,
                   {residual.ne[1], residual.ne[2]});
    const dc::prepare::NormSource source{residual, norm_weight, inverse};
    if (plan.fused) {
        Tensor normalized(storage.subspan(0, plan.inverse_offset), DType::BF16,
                          {geometry.hidden, residual.ne[1], residual.ne[2]});
        rmsnorm(residual, norm_weight, epsilon, false, normalized, inverse, execution);
        dc::prepare::launch_fused(source, normalized, base_kernel, projection, prepared,
                                  finish_delta, stream);
        return;
    }
    auto* partials = static_cast<float*>(storage.data);
    rmsnorm(residual, norm_weight, epsilon, false, prepared, inverse, execution);
    dc::prepare::launch(plan, source, base_kernel, projection, partials, prepared, finish_delta,
                        stream);
}
} // namespace ninfer::ops
