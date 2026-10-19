#pragma once

#include "models/qwen4_exp/execution/batch.h"
#include "models/qwen4_exp/execution/parameters.h"

namespace ninfer::models::qwen4_exp::execution {

// A backend position pairs the preceding Main/draft conditioning with the next input embedding.
// Cache positions and RoPE coordinates belong to the conditioning position.
void mtp_forward(const TextConfig&, const QsaConfig&, const MtpParameters&,
                 const Tensor& conditioning, const BatchView&, const ops::QsaTailStateView& before,
                 ops::QsaTailStateView after, ops::QsaCacheView, TextOutputs, MemoryArena&,
                 CudaExecutionView);

// Builds target-aligned history without attention readout, MoE or final mixing.
void mtp_history(const TextConfig&, const QsaConfig&, const MtpParameters&,
                 const Tensor& conditioning, const BatchView&, const ops::QsaTailStateView& before,
                 ops::QsaTailStateView after, ops::QsaCacheView, std::optional<VisualEmbeddingView>,
                 MemoryArena&, CudaExecutionView);

} // namespace ninfer::models::qwen4_exp::execution
