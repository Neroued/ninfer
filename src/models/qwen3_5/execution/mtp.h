#pragma once

#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t mtp_projection_workspace_bytes(const MtpProjectionParameters& parameters,
                                                         std::int32_t first, std::int32_t last);
[[nodiscard]] std::size_t mtp_kv_workspace_bytes(const MtpProjectionParameters& parameters,
                                                 const AttentionConfig& config, std::int32_t first,
                                                 std::int32_t last);
[[nodiscard]] std::size_t mtp_query_gate_workspace_bytes(const MtpProjectionParameters& parameters,
                                                         const AttentionConfig& config,
                                                         std::int32_t first, std::int32_t last);
void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, WorkspaceArena& workspace, cudaStream_t stream);
void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                       const AttentionConfig& config, Tensor& key, Tensor& value,
                       WorkspaceArena& workspace, cudaStream_t stream);
void mtp_query_gate_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                               const AttentionConfig& config, Tensor& query, Tensor& gate,
                               WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
