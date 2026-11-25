#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/ragged.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"

#include <functional>

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
    Tensor live_rows;
    const ops::SamplingConfig* sampling = nullptr;
};

struct DraftOutputs {
    Tensor ids;
    Tensor positions;
    Tensor drafts;
    Tensor candidate_ids;
    Tensor proposal_q;
    Tensor confidences;
};

struct DraftBuckets {
    ops::ContextBucket local;
    ops::ContextBucket full;
};

struct SlidingWindowReadout {
    Tensor query, key, value, positions, valid_columns, output;
    ConstCyclicKVCacheLayerView context;
    ops::AttentionHeadGeometry geometry;
    float scale;
    MemorySpan scratch;

    [[nodiscard]] ops::OpGraphProfile profile(ops::ContextBucket) const;
    [[nodiscard]] std::size_t scratch_bytes(ops::ContextBucketRange) const;
    void execute(ops::ContextBucket, CudaExecutionView) const;
    void execute(ops::ContextBucket, MemoryArena&, CudaExecutionView) const;
};

struct ContextAttentionReadout {
    Tensor query, key, value, frontiers, valid_columns, table_rows, output;
    ConstPagedKVBatchLayerView context;
    ops::AttentionHeadGeometry geometry;
    float scale;
    MemorySpan scratch;

    [[nodiscard]] ops::OpGraphProfile profile(ops::ContextBucket) const;
    [[nodiscard]] std::size_t scratch_bytes(ops::ContextBucketRange) const;
    void execute(ops::ContextBucket, CudaExecutionView) const;
    void execute(ops::ContextBucket, MemoryArena&, CudaExecutionView) const;
};

// Each supplied callback submits the current layer's readout on the draft stream.
// The proposal workspace owns all operand and scratch backing.
struct DraftReadoutSubmit {
    ops::ContextBucketRange buckets;
    std::function<void(const SlidingWindowReadout&)> local;
    std::function<void(const ContextAttentionReadout&)> full;
};

// Appends target features to cyclic State records and full-attention history. Byte offsets
// select State records; I32 table rows select paged histories.
void draft_append_context(DraftExecutionView, const Tensor& features, const Tensor& positions,
                          const Tensor& counts, const Tensor& offsets, const Tensor& table_rows,
                          ops::KVCacheAppendPrefixExecutionEnvelope);

// Appends packed ragged target features to cyclic State records and (for DFlash1) paged
// full-attention history in a single pass. local_ragged holds each segment truncated to the
// cyclic sliding window; when local_ragged.total_tokens() < ragged.total_tokens(), local_indices
// selects the retained columns from features/positions.
void draft_append_context(DraftExecutionView, const Tensor& features, const Tensor& positions,
                          const ops::RaggedSequenceView& ragged,
                          const ops::RaggedSequenceView& local_ragged, const Tensor& local_indices,
                          const Tensor& offsets, const Tensor& table_rows);

// Produces this round's proposal for each row. ids/positions are [K+1,B], drafts [g,B] for the
// first 1 <= g <= K mask columns. DFlash2 additionally writes its sparse proposal distribution;
// ordinary DFlash leaves it absent.
void draft_propose(DraftExecutionView, const DraftInputs&, DraftOutputs&, DraftBuckets,
                   const DraftReadoutSubmit& = {});
void bind_draft_readouts(DraftExecutionView, const DraftInputs&, const DraftOutputs&,
                         const DraftReadoutSubmit&);

} // namespace ninfer::models::qwen3_5::execution
