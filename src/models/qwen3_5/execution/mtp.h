#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/types.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t mtp_projection_workspace_bytes(const MtpProjectionProfile& parameters,
                                                         std::int32_t first, std::int32_t last);
[[nodiscard]] std::size_t mtp_kv_workspace_bytes(const MtpProjectionProfile& parameters,
                                                 const AttentionConfig& config, std::int32_t first,
                                                 std::int32_t last);
void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, MemoryArena& workspace, CudaExecutionView execution);
void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                       const AttentionConfig& config, Tensor& key, Tensor& value,
                       MemoryArena& workspace, CudaExecutionView execution);

struct MtpExecutionView {
    const Parameters& parameters;
    MemoryArena& workspace;
    CudaExecutionView execution;
    PagedKVBatchLayerView cache;
    ProposalHead proposal_head = ProposalHead::Full;
};

// Each live column pairs target conditioning H[p] with token x[p+1], writes cache position p,
// and applies the explicitly supplied RoPE position. The caller owns shifting and frontiers.
// Inputs are contiguous: ids/cache_positions I32 [W,B], conditioning BF16 [H,W,B],
// valid_columns/kv_table_rows I32 [B], and rope_positions I32 [W*B] or [W*B,3].
// Optional embeddings are BF16 [H,W,B]. All physical columns contain safe inputs; only each
// row's valid prefix writes KV and has defined output hidden values.
struct MtpInputs {
    Tensor ids;
    Tensor conditioning;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor valid_columns;
    Tensor kv_table_rows;
    const Tensor* input_embeddings = nullptr;
};

void mtp_forward(MtpExecutionView state, const MtpInputs& inputs,
                 ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden);

// Materialize K/V from true target conditioning without running query attention or the MLP.
void mtp_history(MtpExecutionView state, const MtpInputs& inputs,
                 ops::KVCacheAppendPrefixExecutionEnvelope envelope);

// hidden is BF16 [H,C] and ids is I32 [C]. Full uses caller-owned BF16 [V,C] logits;
// Optimized uses its selected proposal head and optional token-ID map, leaving full_logits alone.
void mtp_propose(MtpExecutionView state, const Tensor& hidden, Tensor& full_logits, Tensor& ids);

} // namespace ninfer::models::qwen3_5::execution
