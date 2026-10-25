#pragma once

#include "models/common/parallel_execution.h"
#include "models/glm5next/execution/mla.h"
#include "models/glm5next/execution/workspace.h"
#include "models/glm5next/state/context.h"

namespace ninfer::models::glm5next::execution {

struct MtpExecution {
    MemoryArena& workspace;
    const CacheViews& cache;
    ops::ConstKpoolTailStateView before;
    ops::KpoolTailStateView after;
    ParallelExecution* parallel = nullptr;
};

class MtpRun {
public:
    MtpRun(const Parameters&, MtpExecution, BatchView, CudaExecutionView);
    void stem(const Tensor& conditioning);
    void local_attention(MlaSelection, const MlaSubmit& = {});
    [[nodiscard]] MlaRun local_mla(MlaSelection);
    void finish_attention();
    void ffn();
    void finish(Tensor& hidden);

private:
    const Parameters& parameters_;
    MtpExecution execution_;
    BatchView batch_;
    CudaExecutionView cuda_;
    MemoryArena::Scope scope_;
    workspace::MtpRoots roots_;
    workspace::AttentionBuffers attention_;
};

void mtp_history(const Parameters&, MtpExecution, const BatchView&, const Tensor& conditioning,
                 CudaExecutionView);

} // namespace ninfer::models::glm5next::execution
