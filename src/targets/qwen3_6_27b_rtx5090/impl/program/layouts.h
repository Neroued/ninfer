#pragma once

#include "core/layout.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "core/kv_cache.h"
#include "targets/qwen3_6_27b_rtx5090/impl/state/state_store.h"
#include <ninfer/targets/qwen3_6_27b_rtx5090/package.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {

using TensorLayout = TensorRegion;

struct StepStateLayout {
    TensorLayout token;
    TensorLayout pos;
    TensorLayout rope_pos;
    TensorLayout rope_delta;
    TensorLayout logits;
    TensorLayout verify_hidden;
    TensorLayout prefill_hidden;
    TensorLayout target_tokens;
    TensorLayout drafts;
    TensorLayout sampled_out;
    TensorLayout num_sampled;
    TensorLayout verify_ids;
    TensorLayout shifted_ids;
    TensorLayout positions;
    TensorLayout window_base;
    TensorLayout accepted;
    TensorLayout gdn_initial_slot;
    TensorLayout ar_pos;
    TensorLayout mtp_ar_hidden;
    TensorLayout stats;
};

struct PersistentLayout {
    KVCacheLayout text_kv;
    std::optional<KVCacheLayout> mtp_kv;
    GdnStateLayout gdn;
    StepStateLayout io;
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
