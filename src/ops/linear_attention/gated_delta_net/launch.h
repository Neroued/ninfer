#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

struct alignas(16) GdnReplayFoldKernelRow {
    std::int32_t source_state_slot;
    std::int32_t destination_state_slot;
    std::int32_t commit_columns;
    std::int32_t reserved = 0;
};

struct alignas(16) GdnReplayFoldKernelRows {
    GdnReplayFoldKernelRow row[8];
};

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream);

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream);

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, Tensor& ssm_states,
                                   const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, Tensor& out,
                                   cudaStream_t stream);

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& out, cudaStream_t stream);

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        cudaStream_t stream);

std::size_t chunked_workspace_bytes(std::int32_t value_heads, std::int32_t tokens);

void launch_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, float scale, const Tensor& ssm_state_in,
                    Tensor& ssm_state_out, Tensor& out, void* workspace,
                    std::size_t workspace_bytes, cudaStream_t stream);

} // namespace ninfer::ops::detail::gated_delta_net
