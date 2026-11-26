#pragma once

#include "models/inkling2/execution/batch.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/state/context.h"
#include "ninfer/ops/context_bucket.h"

#include <functional>

namespace ninfer::models::inkling2::execution {

struct DSparkReadout {
    ops::AttentionHeadGeometry geometry;
    std::uint32_t window;
    Tensor q, k, v, valid_columns, lengths, table_rows, out;
    ConstCyclicKVCacheLayerView swa;
    ConstPagedKVBatchLayerView full;
    MemorySpan scratch;

    [[nodiscard]] ops::OpGraphProfile profile(ops::ContextBucket) const;
    void execute(ops::ContextBucket, CudaExecutionView) const;
};

using DSparkSubmit = std::function<void(const DSparkReadout&)>;

struct DSparkProposal {
    Tensor anchors, base_positions, rows, valid_columns, stem_indices, tail_indices;
    Tensor drafts, probabilities; // [B, steps]
};

struct DSparkExecution {
    const StateViews& states;
    const CacheViews& cache;
    MemoryArena& arena;
};

void propose_dspark(const Parameters&, DSparkExecution, const BatchView&, const DSparkProposal&,
                    ops::ContextBucketRange, ops::ContextBucket, CudaExecutionView,
                    const DSparkSubmit& = {});

[[nodiscard]] std::vector<DSparkReadout>
plan_dspark_readouts(const Parameters&, DSparkExecution, const BatchView&, const DSparkProposal&,
                     ops::ContextBucketRange, CudaExecutionView);

struct DSparkRaggedContext {
    ops::RaggedSequenceView full, swa;
    Tensor swa_indices, last_indices;
};

void dspark_append_context(const Parameters&, DSparkExecution, const BatchView&, const Tensor& aux,
                           const Tensor& counts, CudaExecutionView);
void dspark_append_context(const Parameters&, DSparkExecution, const BatchView&, const Tensor& aux,
                           const DSparkRaggedContext&, CudaExecutionView);
void dspark_mix_frnn(const Parameters&, const Tensor& aux, Tensor& out, CudaExecutionView);

[[nodiscard]] std::size_t dspark_proposal_bytes(const ModelProfile&, int batch, int steps,
                                                ops::ContextBucketRange);
[[nodiscard]] std::size_t dspark_context_bytes(const ModelProfile&, int columns, int segments);

} // namespace ninfer::models::inkling2::execution
