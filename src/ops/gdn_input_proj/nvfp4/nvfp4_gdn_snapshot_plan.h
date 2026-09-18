#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Nvfp4GdnConvScheduleId {
    DecodeFusedA16,
    SmallTFusedA16,
    Materialized,
};

struct Nvfp4GdnConvPlan {
    Nvfp4GdnConvScheduleId schedule;
};

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy policy, std::int32_t tokens,
                                             std::int32_t batch_size);

[[nodiscard]] std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                                      std::int32_t min_tokens,
                                                                      std::int32_t max_tokens);

void nvfp4_gdn_snapshot_decode_launch(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, cudaStream_t stream);

void nvfp4_gdn_snapshot_small_t_launch(const Tensor& x, const Weight& weight,
                                       const Tensor& conv_weight, Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                       Tensor& value, Tensor& z, cudaStream_t stream);

void nvfp4_gdn_record_small_t_launch(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, const Tensor& conv_states,
                                     const Tensor& valid_columns, const Tensor& initial_slot,
                                     Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                     Tensor& z, cudaStream_t stream);

void nvfp4_gdn_snapshot_post_launch(const Tensor& projected, const Tensor& conv_weight,
                                    Tensor& conv_states, const Tensor& valid_columns,
                                    const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                    Tensor& query, Tensor& key, Tensor& value, cudaStream_t stream);

void nvfp4_gdn_record_post_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                  const Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_slot, Tensor& query, Tensor& key,
                                  Tensor& value, cudaStream_t stream);

void nvfp4_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                 Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                 LinearPolicy policy, WorkspaceArena& workspace,
                                 cudaStream_t stream);

} // namespace ninfer::ops::detail
