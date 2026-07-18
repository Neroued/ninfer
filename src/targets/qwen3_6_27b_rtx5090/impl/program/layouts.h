#pragma once

#include "core/dtype.h"
#include "core/layout.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6_27b_rtx5090/package.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {

using TensorLayout = TensorRegion;

struct PersistentLayout {
    qwen3_6::DecoderStateLayout decoder;
    qwen3_6::RoundStateLayout round;
    TensorLayout prefill_hidden;
    TensorLayout token_counts;
    TensorLayout sampling_config;
    TensorLayout tail_hidden;
    TensorLayout boundary_hidden;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

struct SequencePlan::Impl {
    std::uint32_t capacity      = 0;
    std::uint32_t prefill_chunk = 0;
    std::uint32_t mtp_k         = 0;
    DType kv_dtype              = DType::BF16;
    std::int32_t kv_quant_group = 0;
    ProposalHead proposal_head  = ProposalHead::Full;
    bool use_cuda_graph         = true;
    int device                  = 0;
    PersistentLayout persistent;
    std::size_t workspace_bytes       = 0;
    std::size_t graph_allowance_bytes = 0;
};

void validate_target_options(DeviceContext& device, const EngineOptions& options);
[[nodiscard]] std::unique_ptr<SequencePlan::Impl> plan_sequence_impl(DeviceContext& device,
                                                                     const EngineOptions& options);

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
