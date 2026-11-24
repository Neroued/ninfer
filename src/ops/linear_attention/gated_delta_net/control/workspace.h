#pragma once

#include "ops/common/host/operands.h"
#include "core/layout.h"
#include "ops/linear_attention/gated_delta_net/control/plan.h"

namespace ninfer::ops::detail::gdn_control {

struct Scratch {
    MemorySpan partial;
};

// Query and execution use the same call-local, aligned allocation recipe.
template <class Allocator>
Scratch allocate_workspace(Allocator& allocator, Geometry geometry, Plan plan, std::int32_t columns,
                           bool norm_partial = false) {
    if (plan.split_k == 1) return {};
    const auto rows  = std::uint64_t(2 * geometry.heads + (norm_partial ? 1 : 0));
    const auto count = rows * std::uint64_t(columns) * plan.split_k;
    return {
        allocator.alloc_bytes(common::operand::bytes(count, sizeof(float), "gdn_control"), 256)};
}

inline std::size_t workspace_bytes(Geometry geometry, Plan plan, std::int32_t columns,
                                   bool norm_partial = false) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, geometry, plan, columns, norm_partial);
    return layout.peak_bytes(1);
}

} // namespace ninfer::ops::detail::gdn_control
