#include "models/qwen3_5/execution/gdn.h"
#include "ninfer/ops/gdn_control.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_split.h"

namespace ninfer::models::qwen3_5::execution {

std::size_t gdn_projection_workspace_bytes(const GdnProfile& parameters, std::int32_t first,
                                           std::int32_t last) {
    return ops::linear_split_workspace_capacity_bytes(parameters.projection,
                                                      parameters.projection_rows, first, last);
}

std::size_t gdn_snapshot_workspace_bytes(const GdnProfile& parameters, std::int32_t batch,
                                         std::int32_t first_width, std::int32_t last_width) {
    return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(parameters.projection, batch,
                                                                      first_width, last_width);
}

std::size_t gdn_record_workspace_bytes(const GdnProfile& parameters, std::int32_t batch,
                                       std::int32_t first_width, std::int32_t last_width) {
    return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(parameters.projection, batch,
                                                                    first_width, last_width);
}

void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    MemoryArena& workspace, CudaExecutionView execution) {
    std::array outputs{qkv, z};
    ops::linear_split(hidden, parameters.projection, outputs, workspace, execution);
}

void gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                      const GdnParameters& parameters, Tensor& hidden, Tensor& g, Tensor& beta,
                      MemoryArena& workspace, CudaExecutionView execution) {
    ops::gdn_norm_control_proj(residual, norm, epsilon, parameters.control, parameters.a_log,
                               parameters.dt_bias, {hidden, {g, beta}}, workspace, execution);
}

void gdn_projection_update(const Tensor& hidden, const GdnParameters& parameters,
                           StateFieldView source, StateFieldView destination,
                           const Tensor& valid_columns, Tensor& query, Tensor& key, Tensor& value,
                           Tensor& z, MemoryArena& workspace, CudaExecutionView execution) {
    ops::gdn_input_proj_conv_update(hidden, parameters.projection, parameters.convolution, source,
                                    destination, valid_columns, {query, key, value, z}, workspace,
                                    execution);
}

void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           StateFieldView source, const Tensor& valid_columns, Tensor& conv_record,
                           Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                           MemoryArena& workspace, CudaExecutionView execution) {
    ops::gdn_input_proj_conv_record(hidden, parameters.projection, parameters.convolution, source,
                                    valid_columns, {query, key, value, z}, conv_record, workspace,
                                    execution);
}

} // namespace ninfer::models::qwen3_5::execution
