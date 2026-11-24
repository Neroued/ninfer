#include "models/common/graph.h"

namespace ninfer::models {

void GraphSegment::prepare(CudaExecutionView cuda, const std::function<void()>& body,
                           ParallelExecution* parallel) {
    const auto capture = [&] { definition.capture(cuda.stream(), body); };
    if (parallel)
        parallel->record_graph(communication, capture);
    else
        capture();
    executable.instantiate(definition);
    executable.upload(cuda.stream());
}

void GraphSegment::launch(CudaExecutionView cuda, ParallelExecution* parallel) {
    executable.launch(cuda.stream());
    if (parallel) parallel->replayed(communication);
}

} // namespace ninfer::models
