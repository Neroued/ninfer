#pragma once
#include "models/glm5next/execution/batch.h"
#include "models/glm5next/execution/parameters.h"
#include "models/glm5next/execution/workspace.h"
#include "models/common/parallel_execution.h"
#include "distributed/communication.h"

namespace ninfer::models::glm5next::execution {
struct TextExecution {
    MemoryArena& workspace;
    const StateViews& states;
    const CacheViews& cache;
    Tensor raw_keys, raw_gates;
    ParallelExecution* parallel = nullptr;
};

class TextRun {
public:
    TextRun(const Parameters&, TextExecution, BatchView, StateUpdate, CudaExecutionView);
    void stem(std::optional<models::vision::EmbeddingView> = {});
    void local_mixer(std::uint32_t layer);
    void finish_mixer(std::uint32_t layer);
    void ffn(std::uint32_t layer);
    void finish_layer(std::uint32_t layer);
    void finish(Tensor& hidden);
private:
    void next_pre(const ops::MhcPreWeights&);
    const Parameters& parameters_;
    TextExecution execution_;
    BatchView batch_;
    StateUpdate update_;
    CudaExecutionView cuda_;
    MemoryArena::Scope scope_;
    workspace::Roots roots_;
    std::uint32_t coefficients_ = 0;
    std::vector<distributed::PeerTransfer> mla_exchange_;
};
} // namespace ninfer::models::glm5next::execution
