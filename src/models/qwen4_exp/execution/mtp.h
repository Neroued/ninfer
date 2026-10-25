#pragma once

#include "models/qwen4_exp/execution/batch.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/common/parallel_execution.h"
#include "models/qwen4_exp/execution/workspace.h"

namespace ninfer::models::qwen4_exp::execution {

class MtpRun {
public:
    MtpRun(const TextConfig&, const QsaConfig&, const MtpParameters&, const Tensor& conditioning,
           BatchView, ops::QsaTailStateView before, ops::QsaTailStateView after, ops::QsaCacheView,
           MemoryArena&, CudaExecutionView, ParallelExecution* = nullptr);
    void stem();
    void local_mixer();
    void finish_mixer();
    void feed_forward();
    void finish(TextOutputs);

private:
    const TextConfig& config_;
    const QsaConfig& qsa_;
    const MtpParameters& parameters_;
    Tensor conditioning_;
    BatchView batch_;
    ops::QsaTailStateView before_, after_;
    ops::QsaCacheView cache_;
    MemoryArena& arena_;
    CudaExecutionView cuda_;
    ParallelExecution* parallel_;
    MemoryArena::Scope scope_;
    workspace::Roots roots_;
};

// A backend position pairs the preceding Main/draft conditioning with the next input embedding.
// Cache positions and RoPE coordinates belong to the conditioning position.
void mtp_forward(const TextConfig&, const QsaConfig&, const MtpParameters&,
                 const Tensor& conditioning, const BatchView&, const ops::QsaTailStateView& before,
                 ops::QsaTailStateView after, ops::QsaCacheView, TextOutputs, MemoryArena&,
                 CudaExecutionView, ParallelExecution* = nullptr);

// Builds target-aligned history without attention readout, MoE or final mixing.
void mtp_history(const TextConfig&, const QsaConfig&, const MtpParameters&,
                 const Tensor& conditioning, const BatchView&, const ops::QsaTailStateView& before,
                 ops::QsaTailStateView after, ops::QsaCacheView, std::optional<VisualEmbeddingView>,
                 MemoryArena&, CudaExecutionView);

} // namespace ninfer::models::qwen4_exp::execution
