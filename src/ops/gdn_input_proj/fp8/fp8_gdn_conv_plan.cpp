#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"

#include "core/layout.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_config.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kChannels = 10240;
constexpr std::int32_t kZRows    = 6144;

enum class Fp8GdnConvScheduleId : std::uint8_t {
    FusedA16,
    MaterializedA16,
    MaterializedA8,
};

struct Fp8GdnConvPlan {
    Fp8GdnConvScheduleId schedule;
};

struct Fp8GdnProjectedWorkspace {
    Tensor projected;
};

template <class Allocator>
Fp8GdnProjectedWorkspace allocate_projected(Allocator& allocator, std::int32_t columns) {
    return {allocator.alloc(DType::BF16, {kChannels, columns}, 256)};
}

void require_policy(LinearPolicy policy, const char* operation) {
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument(std::string(operation) + " admits only A16 or A8");
    }
}

Fp8GdnConvPlan b1_a16_plan(std::int32_t width) {
    const bool fused = width <= 3 || (width >= 7 && width <= 10);
    return {fused ? Fp8GdnConvScheduleId::FusedA16 : Fp8GdnConvScheduleId::MaterializedA16};
}

bool materialized(Fp8GdnConvPlan plan) { return plan.schedule != Fp8GdnConvScheduleId::FusedA16; }

std::size_t snapshot_capacity(Fp8GdnConvPlan maximum_plan, std::int32_t materialized_columns,
                              std::int32_t maximum_columns) {
    if (materialized_columns == 0) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected(layout, materialized_columns);
    if (maximum_plan.schedule == Fp8GdnConvScheduleId::MaterializedA8) {
        (void)allocate_fp8_a8_workspace(layout, maximum_columns, Fp8GdnInputGeometry::kInputRows);
    }
    return layout.peak_bytes(1);
}

std::size_t record_capacity(Fp8GdnConvPlan plan, std::int32_t aggregate_columns) {
    if (plan.schedule != Fp8GdnConvScheduleId::MaterializedA8) { return 0; }
    return fp8_a8_workspace_capacity_bytes(aggregate_columns, Fp8GdnInputGeometry::kInputRows);
}

void launch_projection(const Tensor& x, const Weight& weight, Tensor& projected, Tensor& z,
                       Fp8GdnConvScheduleId schedule, WorkspaceArena& workspace,
                       cudaStream_t stream) {
    if (schedule == Fp8GdnConvScheduleId::MaterializedA16) {
        fp8_gdn_input_a16_dispatch(x, weight, projected, z, stream);
        return;
    }
    if (schedule == Fp8GdnConvScheduleId::MaterializedA8) {
        fp8_gdn_input_a8_dispatch(x, weight, projected, z, workspace, stream);
        return;
    }
    throw std::logic_error("fp8 GDN materialized projection received a fused plan");
}

Fp8GdnConvPlan fp8_gdn_snapshot_resolve_plan(LinearPolicy policy, std::int32_t width,
                                             std::int32_t batch_size) {
    require_policy(policy, "fp8 GDN snapshot");
    if (width <= 0 || batch_size <= 0 || batch_size > 8 || (batch_size > 1 && width > 16)) {
        throw std::invalid_argument("fp8 GDN snapshot: invalid B/W domain");
    }
    if (batch_size == 1) {
        if (policy == LinearPolicy::AllowA8 && width >= 10) {
            return {Fp8GdnConvScheduleId::MaterializedA8};
        }
        return b1_a16_plan(width);
    }
    if (policy == LinearPolicy::AllowA8 && width * batch_size >= 9) {
        return {Fp8GdnConvScheduleId::MaterializedA8};
    }
    return {Fp8GdnConvScheduleId::MaterializedA16};
}

Fp8GdnConvPlan fp8_gdn_record_resolve_plan(LinearPolicy policy, std::int32_t width,
                                           std::int32_t batch_size) {
    require_policy(policy, "fp8 GDN record");
    if (width < 2 || width > 16 || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument("fp8 GDN record: invalid B/W domain");
    }
    // Record and snapshot must choose the same arithmetic for the same physical block.
    return fp8_gdn_snapshot_resolve_plan(policy, width, batch_size);
}

} // namespace

std::size_t fp8_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy, std::int32_t batch_size,
                                                      std::int32_t min_width,
                                                      std::int32_t max_width) {
    if (min_width <= 0 || max_width < min_width) {
        throw std::invalid_argument("fp8 GDN snapshot workspace: invalid width interval");
    }
    (void)fp8_gdn_snapshot_resolve_plan(policy, min_width, batch_size);
    const Fp8GdnConvPlan maximum = fp8_gdn_snapshot_resolve_plan(policy, max_width, batch_size);
    std::int32_t largest_materialized_width = 0;
    if (batch_size > 1 || max_width > kFp8LinearSmallTMax<Fp8GdnInputGeometry>) {
        largest_materialized_width = max_width;
    } else {
        for (std::int32_t width = min_width; width <= max_width; ++width) {
            if (materialized(fp8_gdn_snapshot_resolve_plan(policy, width, batch_size))) {
                largest_materialized_width = width;
            }
        }
    }
    return snapshot_capacity(maximum, batch_size * largest_materialized_width,
                             batch_size * max_width);
}

std::size_t fp8_gdn_record_workspace_capacity_bytes(LinearPolicy policy, std::int32_t batch_size,
                                                    std::int32_t min_width,
                                                    std::int32_t max_width) {
    if (min_width < 2 || max_width < min_width) {
        throw std::invalid_argument("fp8 GDN record workspace: invalid width interval");
    }
    (void)fp8_gdn_record_resolve_plan(policy, min_width, batch_size);
    const Fp8GdnConvPlan maximum = fp8_gdn_record_resolve_plan(policy, max_width, batch_size);
    return record_capacity(maximum, batch_size * max_width);
}

namespace {

void launch_snapshot_plan(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                          Tensor& conv_states, const Tensor& valid_columns,
                          const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                          Tensor& query, Tensor& key, Tensor& value, Tensor& z, Fp8GdnConvPlan plan,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (plan.schedule == Fp8GdnConvScheduleId::FusedA16) {
        fp8_gdn_snapshot_fused_launch(x, weight, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value, z,
                                      stream);
        return;
    }

    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    const std::int32_t aggregate_columns = width * batch;
    auto scope                           = workspace.scope();
    Fp8GdnProjectedWorkspace scratch     = allocate_projected(workspace, aggregate_columns);
    Tensor x_flat(x.data, DType::BF16, {Fp8GdnInputGeometry::kInputRows, aggregate_columns});
    Tensor z_flat(z.data, DType::BF16, {kZRows, aggregate_columns});
    launch_projection(x_flat, weight, scratch.projected, z_flat, plan.schedule, workspace, stream);

    Tensor projected(scratch.projected.data, DType::BF16, {kChannels, width, batch});
    gdn_projected_conv_snapshot_launch(projected, conv_weight, conv_states, valid_columns,
                                       initial_slot, snapshot_base_slot, query, key, value, stream);
}

void launch_record_plan(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                        const Tensor& conv_states, const Tensor& valid_columns,
                        const Tensor& initial_slot, Tensor& conv_record, Tensor& query, Tensor& key,
                        Tensor& value, Tensor& z, Fp8GdnConvPlan plan, WorkspaceArena& workspace,
                        cudaStream_t stream) {
    if (plan.schedule == Fp8GdnConvScheduleId::FusedA16) {
        fp8_gdn_record_fused_launch(x, weight, conv_weight, conv_states, valid_columns,
                                    initial_slot, conv_record, query, key, value, z, stream);
        return;
    }

    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    const std::int32_t aggregate_columns = width * batch;
    auto scope                           = workspace.scope();
    Tensor x_flat(x.data, DType::BF16, {Fp8GdnInputGeometry::kInputRows, aggregate_columns});
    Tensor record_flat(conv_record.data, DType::BF16, {kChannels, aggregate_columns});
    Tensor z_flat(z.data, DType::BF16, {kZRows, aggregate_columns});
    launch_projection(x_flat, weight, record_flat, z_flat, plan.schedule, workspace, stream);
    gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states, valid_columns,
                                     initial_slot, query, key, value, stream);
}

} // namespace

void fp8_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                               Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                               Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                               LinearPolicy policy, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    launch_snapshot_plan(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, snapshot_base_slot, query,
        key, value, z, fp8_gdn_snapshot_resolve_plan(policy, x.ne[1], x.ne[2]), workspace, stream);
}

void fp8_gdn_record_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                             const Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                             Tensor& key, Tensor& value, Tensor& z, LinearPolicy policy,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    launch_record_plan(x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                       conv_record, query, key, value, z,
                       fp8_gdn_record_resolve_plan(policy, x.ne[1], x.ne[2]), workspace, stream);
}

} // namespace ninfer::ops::detail
