#pragma once

#include "models/inkling2/execution/batch.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/execution/workspace.h"
#include "models/inkling2/state/context.h"

#include <functional>

namespace ninfer::models::inkling2::execution {

// All bindings, including scratch, borrow stable backing. The submitter enqueues or
// captures the readout on the execution stream before its workspace can be reused.
struct FullAttentionReadout {
    ops::AttentionHeadGeometry geometry;
    float scale = 0.0F;
    ops::RelativeAttentionQueryInputs inputs;
    Tensor valid_columns;
    Tensor kv_table_rows;
    ConstPagedKVBatchLayerView cache;
    MemorySpan scratch;
    Tensor out;

    [[nodiscard]] ops::OpGraphProfile profile(ops::ContextBucket bucket, std::int32_t sms) const;
    [[nodiscard]] std::size_t scratch_bytes(ops::ContextBucketRange buckets,
                                            std::int32_t sms) const;
    void execute(ops::ContextBucket bucket, CudaExecutionView cuda) const;
};

using FullAttentionSubmit = std::function<void(const FullAttentionReadout&)>;

struct AttentionExecution {
    const StateViews& states;
    const CacheViews& cache;
};

class RelativeAttentionRun {
public:
    RelativeAttentionRun(const Parameters&, const LogicalLayerConfig&, const BatchView&,
                         AttentionExecution, const Tensor& input, Tensor output, MemoryArena&,
                         CudaExecutionView, const RecordWindow* = nullptr);

    [[nodiscard]] FullAttentionReadout readout(ops::ContextBucketRange buckets);
    void execute(const FullAttentionSubmit& submit = {});

private:
    void project();
    void prepare();
    void attend_swa();
    void finish();

    const TextConfig& config_;
    const LogicalLayerConfig& logical_;
    const PhysicalLayerParameters& layer_;
    const AttentionPassParameters& pass_;
    ops::AttentionHeadGeometry geometry_;
    const BatchView& batch_;
    const StateViews& states_;
    Tensor input_, output_;
    PagedKVBatchLayerView full_cache_;
    CyclicKVCacheLayerView swa_cache_;
    MemoryArena& arena_;
    CudaExecutionView cuda_;
    MemoryArena::Scope scope_;
    workspace::AttentionBuffers buffers_;
    ops::RelativeAttentionQueryInputs query_;
    Tensor attended_;
    const RecordWindow* records_;
};

void relative_attention_ragged(const Parameters&, const LogicalLayerConfig&, const BatchView&,
                               const ops::RaggedSequenceView&, AttentionExecution,
                               const Tensor& input, Tensor& output, MemoryArena&,
                               CudaExecutionView);

} // namespace ninfer::models::inkling2::execution
