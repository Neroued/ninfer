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

    void execute(std::uint32_t visible, CudaExecutionView) const;
    void execute(std::uint32_t visible, MemoryArena&, CudaExecutionView) const;
};

using QsaSubmit = std::function<void(const QsaReadout&)>;

class QsaRun {
public:
    QsaRun(const Tensor&, const QsaConfig&, const QsaParameters&, BatchView,
           ops::QsaTailStateView before, std::optional<ops::QsaTailStateView> after,
           ops::QsaCacheView, ops::QsaRawRecordWriteView, Tensor output, MemoryArena&,
           CudaExecutionView);
    void prepare();
    [[nodiscard]] QsaReadout readout(std::span<const std::uint32_t> bounds = {});
    void finish();
    void execute(const QsaSubmit& = {});

private:
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
    MemoryArena::Scope scope_;
    workspace::QsaBuffers buffers_;
};

void qsa_history(const Tensor& input, const QsaConfig&, const QsaHistoryParameters&,
                 const BatchView&, const ops::QsaTailStateView& before, ops::QsaTailStateView after,
                 ops::QsaCacheView, MemoryArena&, CudaExecutionView);

} // namespace ninfer::models::qwen4_exp::execution
