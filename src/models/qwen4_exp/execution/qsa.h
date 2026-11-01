#pragma once

#include "models/qwen4_exp/execution/batch.h"
#include "models/qwen4_exp/execution/parameters.h"

namespace ninfer::models::qwen4_exp::execution {

void qsa(const Tensor& input, const QsaConfig& c, const QsaParameters&, const BatchView&,
         const ops::QsaTailStateView& before, std::optional<ops::QsaTailStateView> after,
         ops::QsaCacheView, ops::QsaRawRecordWriteView records, Tensor& output, MemoryArena&,
         CudaExecutionView);

void qsa_history(const Tensor& input, const QsaConfig&, const QsaHistoryParameters&,
                 const BatchView&, const ops::QsaTailStateView& before, ops::QsaTailStateView after,
                 ops::QsaCacheView, MemoryArena&, CudaExecutionView);

} // namespace ninfer::models::qwen4_exp::execution
