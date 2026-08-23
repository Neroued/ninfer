#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                                    std::int32_t batch_size,
                                                                    std::int32_t min_width,
                                                                    std::int32_t max_width);

[[nodiscard]] std::size_t fp8_gdn_record_workspace_capacity_bytes(LinearPolicy policy,
                                                                  std::int32_t batch_size,
                                                                  std::int32_t min_width,
                                                                  std::int32_t max_width);

void fp8_gdn_snapshot_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   cudaStream_t stream);

void fp8_gdn_record_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);

void fp8_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                               Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                               Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

void fp8_gdn_record_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                             const Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                             Tensor& key, Tensor& value, Tensor& z, LinearPolicy policy,
                             WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
