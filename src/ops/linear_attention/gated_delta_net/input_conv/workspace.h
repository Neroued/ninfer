#pragma once

#include "ops/linear_attention/gated_delta_net/input_conv/fp8/launch.h"
#include "ops/linear_attention/gated_delta_net/input_conv/plan.h"
#include "ops/linear/kernels/fp8/fp8_a8_plan.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"

namespace ninfer::ops::detail::gdn_input {

struct Scratch {
    Tensor projected;
    Fp8A8Workspace fp8;
    Nvfp4A4Workspace nvfp4;
};

// The same allocation recipe runs on WorkspaceLayoutBuilder and on the call-local arena.
template <class Allocator>
Scratch allocate_workspace(Allocator& allocator, const WeightProfile& weights, ProjectionPlan plan,
                           std::int32_t columns, bool materialized_snapshot = false) {
    Scratch out;
    if (materialized_snapshot)
        out.projected = allocator.alloc(DType::BF16, {weights.geometry.channels(), columns}, 256);
    if (plan.compute == Compute::A8)
        out.fp8 = allocate_fp8_a8_workspace(allocator, columns, weights.geometry.input_rows,
                                            fp8_gdn_input_partial_capacity_bytes(columns));
    if (plan.compute == Compute::A4)
        out.nvfp4 = allocate_nvfp4_a4_workspace(allocator, columns, weights.geometry.input_rows);
    return out;
}

inline std::size_t workspace_bytes(const WeightProfile& weights, ProjectionPlan plan,
                                   std::int32_t columns, bool materialized_snapshot = false) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, weights, plan, columns, materialized_snapshot);
    return layout.peak_bytes(1);
}

} // namespace ninfer::ops::detail::gdn_input
