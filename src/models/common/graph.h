#pragma once

#include "core/decode_graph.h"
#include "models/common/parallel_execution.h"

namespace ninfer::models {

struct GraphSegment {
    CommunicationWork communication;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;

    void prepare(CudaExecutionView, const std::function<void()>&, ParallelExecution* = nullptr);
    void launch(CudaExecutionView, ParallelExecution* = nullptr);
};

} // namespace ninfer::models
