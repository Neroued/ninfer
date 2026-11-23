#pragma once

#include "core/decode_graph.h"
#include "models/common/parallel_execution.h"
#include "models/common/graph_preparation.h"

namespace ninfer::models {

struct GraphSegment {
    CommunicationGraphWork communication;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;

    void prepare(CudaExecutionView, const std::function<void()>&, ParallelExecution* = nullptr);
    void launch(CudaExecutionView, ParallelExecution* = nullptr,
                const CommunicationGraphWork* replay_work = nullptr);
};

// One active prefix of serial MTP steps in a fused round.
class MtpGraphSteps {
public:
    void capture_step(cudaStream_t, ParallelExecution*, const std::function<void()>&);
    void finish(const CommunicationGraphWork& full);

    [[nodiscard]] bool empty() const noexcept { return work_.empty(); }

    [[nodiscard]] const CommunicationGraphWork& set_depth(const DecodeGraphExecutable&, unsigned);

private:
    std::vector<DecodeGraphSection> steps_;
    std::vector<CommunicationGraphWork> work_;
    unsigned installed_depth_ = 0;
};

} // namespace ninfer::models
