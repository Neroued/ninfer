#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "core/device.h"
#include "core/record_view.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t gdn_projection_workspace_bytes(const GdnProfile& parameters,
                                                         std::int32_t first, std::int32_t last);
[[nodiscard]] std::size_t gdn_snapshot_workspace_bytes(const GdnProfile& parameters,
                                                       std::int32_t batch, std::int32_t first_width,
                                                       std::int32_t last_width);
[[nodiscard]] std::size_t gdn_record_workspace_bytes(const GdnProfile& parameters,
                                                     std::int32_t batch, std::int32_t first_width,
                                                     std::int32_t last_width);
void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    MemoryArena& workspace, CudaExecutionView execution);
void gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                      const GdnParameters& parameters, Tensor& hidden, Tensor& g, Tensor& beta,
                      MemoryArena& workspace, CudaExecutionView execution);
void gdn_projection_update(const Tensor& hidden, const GdnParameters& parameters,
                           StateFieldView source, StateFieldView destination,
                           const Tensor& valid_columns, Tensor& query, Tensor& key, Tensor& value,
                           Tensor& z, MemoryArena& workspace, CudaExecutionView execution);
void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           StateFieldView source, const Tensor& valid_columns, Tensor& conv_record,
                           Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                           MemoryArena& workspace, CudaExecutionView execution);

} // namespace ninfer::models::qwen3_5::execution
