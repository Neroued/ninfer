#include "models/common/graph.h"

#include <algorithm>

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

void GraphSegment::launch(CudaExecutionView cuda, ParallelExecution* parallel,
                          const CommunicationGraphWork* replay_work) {
    executable.launch(cuda.stream());
    if (parallel) parallel->replayed(replay_work ? *replay_work : communication);
}

void MtpGraphSteps::capture_step(cudaStream_t stream, ParallelExecution* parallel,
                                 const std::function<void()>& body) {
    if (work_.empty())
        body();
    else
        steps_.emplace_back().capture(stream, body);
    work_.push_back(parallel ? parallel->captured_work() : CommunicationGraphWork{});
}

void MtpGraphSteps::finish(const CommunicationGraphWork& full) {
    if (work_.empty()) throw std::logic_error("no MTP steps in fused round");
    // All snapshots come from one recording with a stable backend order.
    const auto& last = work_.back();
    for (std::size_t i = 0; i < full.size(); ++i) {
        const auto& total = full[i].work;
        const auto end    = last.at(i).work;
        for (auto& prefix : work_) {
            auto& b = prefix.at(i);
            if (b.backend != full[i].backend) throw std::logic_error("MTP graph backend changed");
            for (std::size_t op = 0; op < total.calls.size(); ++op) {
                b.work.calls[op] += total.calls[op] - end.calls[op];
                b.work.payload_bytes[op] += total.payload_bytes[op] - end.payload_bytes[op];
            }
        }
    }
    installed_depth_ = static_cast<unsigned>(work_.size());
}

const CommunicationGraphWork& MtpGraphSteps::set_depth(const DecodeGraphExecutable& exec,
                                                       unsigned depth) {
    if (!installed_depth_ || !depth || depth > work_.size())
        throw std::logic_error("invalid MTP graph depth");
    for (unsigned i = std::min(depth, installed_depth_); i < std::max(depth, installed_depth_); ++i)
        steps_[i - 1].set_enabled(exec, i < depth);
    installed_depth_ = depth;
    return work_[depth - 1];
}

} // namespace ninfer::models
