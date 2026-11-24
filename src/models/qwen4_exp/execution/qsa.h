#pragma once

#include "models/qwen4_exp/execution/batch.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/execution/workspace.h"

#include <functional>

namespace ninfer::models::qwen4_exp::execution {

struct QsaReadout {
    Tensor query, gate, index_query, output;
    ops::ConstQsaIndexCacheView index;
    ConstPagedKVBatchLayerView cache;
    ops::QsaBatchView batch;
    ops::QsaBlockSelection selection;
    MemorySpan scratch;
    int multiprocessor_count;

    [[nodiscard]] ops::OpGraphProfile index_profile(ops::ContextBucket) const;
    [[nodiscard]] ops::OpGraphProfile attention_profile(ops::ContextBucket, int sms) const;
    void indexer(ops::ContextBucket, CudaExecutionView) const;
    void attention(ops::ContextBucket, CudaExecutionView) const;
    void execute(ops::ContextBucket, MemoryArena&, CudaExecutionView) const;

private:
    void indexer(ops::ContextBucket, MemoryArena&, CudaExecutionView) const;
    void attention(ops::ContextBucket, MemoryArena&, CudaExecutionView) const;
};

using QsaSubmit = std::function<void(const QsaReadout&)>;

class QsaRun {
public:
    QsaRun(const Tensor&, const QsaConfig&, const QsaParameters&, BatchView,
           ops::QsaTailStateView before, std::optional<ops::QsaTailStateView> after,
           ops::QsaCacheView, ops::QsaRawRecordWriteView, Tensor output, MemoryArena&,
           CudaExecutionView, const PackedPrefillState* = nullptr);
    void prepare();
    [[nodiscard]] QsaReadout readout(ops::ContextBucketRange);
    void finish();
    void execute(const QsaSubmit& = {});

private:
    [[nodiscard]] QsaReadout bind_readout() const;

    Tensor input_, output_;
    const QsaConfig& config_;
    const QsaParameters& parameters_;
    BatchView batch_;
    ops::QsaTailStateView before_;
    std::optional<ops::QsaTailStateView> after_;
    ops::QsaCacheView cache_;
    ops::QsaRawRecordWriteView records_;
    MemoryArena& arena_;
    CudaExecutionView cuda_;
    const PackedPrefillState* packed_;
    MemoryArena::Scope scope_;
    workspace::QsaBuffers buffers_;
};

void qsa_history(const Tensor& input, const QsaConfig&, const QsaHistoryParameters&,
                 const BatchView&, const ops::QsaTailStateView& before, ops::QsaTailStateView after,
                 ops::QsaCacheView, MemoryArena&, CudaExecutionView,
                 const ops::RaggedSequenceView* = nullptr);

} // namespace ninfer::models::qwen4_exp::execution
