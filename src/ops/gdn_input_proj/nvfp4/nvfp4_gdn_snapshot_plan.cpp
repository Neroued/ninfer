#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include "core/layout.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct Nvfp4GdnProjectedWorkspace {
    Tensor projected;
    DeviceSpan projection;
};

template <class Allocator>
Nvfp4GdnProjectedWorkspace allocate_workspace(Allocator& allocator, std::int32_t tokens) {
    Nvfp4GdnProjectedWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {10240, tokens}, 256);
    // Floored for the same reason as the linear_swiglu baseline: the A16 sub-projection declares
    // no transient, and an arena allocation must be nonzero.
    const std::size_t projection_bytes = std::max<std::size_t>(
        nvfp4_gdn_input_workspace_capacity_bytes(kNvfp4InternalPolicy, tokens, tokens), 256);
    out.projection = allocator.alloc_bytes(projection_bytes, 256);
    return out;
}

} // namespace

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy policy, std::int32_t tokens,
                                             std::int32_t batch_size) {
    if (tokens <= 0 || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument("nvfp4 gdn conv: invalid B/T domain");
    }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 gdn conv admits only A16 or A4");
    }
    if (batch_size > 1) { return {Nvfp4GdnConvScheduleId::Materialized}; }
    if (policy == LinearPolicy::A16Only) {
        if (tokens == 1) { return {Nvfp4GdnConvScheduleId::DecodeFusedA16}; }
        if (tokens <= 16) { return {Nvfp4GdnConvScheduleId::SmallTFusedA16}; }
        // Same reasoning as the A4 branch below: past the fused registration, materialize the
        // projection and post-process. The sub-projection inherits kNvfp4InternalPolicy, which is
        // A16 on Volta and chunks internally, so this covers prefill width.
        return {Nvfp4GdnConvScheduleId::Materialized};
    }
    if (tokens == 1) { return {Nvfp4GdnConvScheduleId::DecodeFusedA16}; }
    if (tokens <= 3) { return {Nvfp4GdnConvScheduleId::SmallTFusedA16}; }
    return {Nvfp4GdnConvScheduleId::Materialized};
}

std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                        std::int32_t min_tokens,
                                                        std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn snapshot workspace: invalid token interval");
    }
    (void)nvfp4_gdn_conv_resolve_plan(policy, min_tokens, 1);
    const Nvfp4GdnConvPlan maximum_plan = nvfp4_gdn_conv_resolve_plan(policy, max_tokens, 1);
    if (maximum_plan.schedule != Nvfp4GdnConvScheduleId::Materialized) { return 0; }

    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, max_tokens);
    return layout.peak_bytes(1);
}

void nvfp4_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                 Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                 LinearPolicy policy, WorkspaceArena& workspace,
                                 cudaStream_t stream) {
    switch (nvfp4_gdn_conv_resolve_plan(policy, x.ne[1], 1).schedule) {
    case Nvfp4GdnConvScheduleId::DecodeFusedA16:
        nvfp4_gdn_snapshot_decode_launch(x, weight, conv_weight, conv_states, valid_columns,
                                         initial_slot, snapshot_base_slot, query, key, value, z,
                                         stream);
        return;
    case Nvfp4GdnConvScheduleId::SmallTFusedA16:
        nvfp4_gdn_snapshot_small_t_launch(x, weight, conv_weight, conv_states, valid_columns,
                                          initial_slot, snapshot_base_slot, query, key, value, z,
                                          stream);
        return;
    case Nvfp4GdnConvScheduleId::Materialized:
        break;
    }

    auto scope                         = workspace.scope();
    Nvfp4GdnProjectedWorkspace scratch = allocate_workspace(workspace, x.ne[1]);
    WorkspaceArena projection_workspace(scratch.projection);
    nvfp4_gdn_input_dispatch(x, weight, scratch.projected, z, kNvfp4InternalPolicy,
                             &projection_workspace, stream);
    nvfp4_gdn_snapshot_post_launch(scratch.projected, conv_weight, conv_states, valid_columns,
                                   initial_slot, snapshot_base_slot, query, key, value, stream);
}

} // namespace ninfer::ops::detail
