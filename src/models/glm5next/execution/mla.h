#pragma once
#include "models/glm5next/execution/batch.h"
#include "models/glm5next/execution/parameters.h"
#include "models/glm5next/execution/workspace.h"

#include <functional>

namespace ninfer::models::glm5next::execution {
struct MlaSelection {
    ops::KpoolBlockSelection value;
    bool reuse = false;
};

// Borrowed, fully bound readout. Its scratch starts at the same cursor for direct execution
// and independently captured Indexer/MLA nodes; the backing owner outlives all consumers.
struct MlaReadout {
    Tensor query, index_query, coefficients, output, selection_rows;
    ops::KpoolMlaFactors factors;
    ops::KpoolMlaCacheView cache;
    ops::KpoolMlaBatchView batch;
    ops::KpoolBlockSelection selection;
    MemorySpan scratch;
    float scale;
    ops::ContextBucket bucket;
    bool index;

    [[nodiscard]] ops::OpGraphProfile index_profile(ops::ContextBucket, int sms) const;
    [[nodiscard]] ops::OpGraphProfile attention_profile(ops::ContextBucket, int sms) const;
    [[nodiscard]] std::size_t scratch_bytes(ops::ContextBucketRange, int sms) const;
    void indexer(ops::ContextBucket, CudaExecutionView) const;
    void attention(ops::ContextBucket, CudaExecutionView) const;
};

using MlaSubmit = std::function<void(const MlaReadout&)>;

class MlaRun {
public:
    MlaRun(const Tensor&, const TextConfig&, const MlaParameters&, ops::KpoolMlaBatchView,
           ops::ConstKpoolTailStateView, std::optional<ops::KpoolTailStateView>,
           ops::KpoolMlaCacheView, ops::KpoolRawRecordWriteView, const Tensor& selection_rows,
           ops::ContextBucket bucket, Tensor output, MemoryArena&, CudaExecutionView,
           std::optional<MlaSelection> = {});
    void prepare();
    [[nodiscard]] MlaReadout readout(ops::ContextBucketRange);
    void execute(const MlaSubmit& = {});
    void finish();
private:
    const TextConfig& config_;
    const MlaParameters& parameters_;
    Tensor input_, output_, selection_rows_;
    ops::KpoolMlaBatchView batch_;
    ops::ConstKpoolTailStateView before_;
    std::optional<ops::KpoolTailStateView> after_;
    ops::KpoolMlaCacheView cache_;
    ops::KpoolRawRecordWriteView records_;
    ops::ContextBucket bucket_;
    MemoryArena& arena_;
    CudaExecutionView cuda_;
    MemoryArena::Scope scope_;
    workspace::MlaBuffers buffers_;
    ops::KpoolBlockSelection selection_;
    bool reuse_;
};

void mla_history(const Tensor&, const MlaParameters&, const ops::KpoolMlaBatchView&,
                 ops::ConstKpoolTailStateView, ops::KpoolTailStateView, ops::KpoolMlaCacheView,
                 MemoryArena&, CudaExecutionView);
void mla_history(const Tensor&, const MlaParameters&, const ops::KpoolMlaRaggedBatchView&,
                 ops::ConstKpoolTailStateView, ops::KpoolTailStateView, ops::KpoolMlaCacheView,
                 MemoryArena&, CudaExecutionView);

void mla_prefill(const Tensor&, const TextConfig&, const MlaParameters&,
                 const ops::KpoolMlaRaggedBatchView&, ops::ConstKpoolTailStateView,
                 ops::KpoolTailStateView, ops::KpoolMlaCacheView, const Tensor& selection_rows,
                 ops::ContextBucket, Tensor&, MemoryArena&, CudaExecutionView);
} // namespace ninfer::models::glm5next::execution
