#pragma once

#include "models/qwen4_exp/execution/batch.h"
#include "models/qwen4_exp/execution/parameters.h"

namespace ninfer::models::qwen4_exp::execution {

void ple(const Tensor& embedding, const TextConfig& c, const PleParameters&, const BatchView&,
         const ops::PleConvStateView& before, std::optional<ops::PleConvStateView> after,
         ops::PleConvRecordWindowView records, const HcParameters& next, Tensor& residual,
         Tensor& normalized, MemoryArena&, CudaExecutionView);

} // namespace ninfer::models::qwen4_exp::execution
