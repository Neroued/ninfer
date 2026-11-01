#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"

namespace ninfer::models::qwen3_5::execution {

struct DraftExecutionView {
    const Parameters& parameters;
    MemoryArena& work;
    CudaExecutionView cuda;
    const CyclicKVCacheView& local;
    const PagedKVCache* full   = nullptr;
    ProposalHead proposal_head = ProposalHead::Full;

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer,
                                                     const Tensor& offsets) const;
    [[nodiscard]] PagedKVBatchLayerView full_layer(std::uint32_t layer) const;
};

struct DraftInputs {
    Tensor anchors;
    Tensor frontiers;
    Tensor valid_columns;
    Tensor state_offsets;
    Tensor full_kv_rows;
    const ops::SamplingConfig* sampling = nullptr;
};

struct DraftOutputs {
    Tensor ids;
    Tensor positions;
    Tensor drafts;
    Tensor candidate_ids;
    Tensor proposal_q;
};

struct DraftEnvelopes {
    ops::SlidingWindowAttentionExecutionEnvelope local;
    ops::ContextAttentionExecutionEnvelope full;
};

// Appends target features to cyclic State records and full-attention history. Byte offsets
// select State records; I32 table rows select paged histories.
void draft_append_context(DraftExecutionView, const Tensor& features, const Tensor& positions,
                          const Tensor& counts, const Tensor& offsets, const Tensor& table_rows,
                          ops::KVCacheAppendPrefixExecutionEnvelope);

// Produces this round's proposal for each row. ids/positions are [K+1,B], drafts [K,B].
// DFlash2 additionally writes its sparse proposal distribution; ordinary DFlash leaves it absent.
void draft_propose(DraftExecutionView, const DraftInputs&, DraftOutputs&, DraftEnvelopes);

} // namespace ninfer::models::qwen3_5::execution
