#pragma once

#include "ninfer_bench_common.h"

namespace ninfer::bench::moe {

struct Stats {
    double median_us = 0.0;
    double min_us    = 0.0;
    double p95_us    = 0.0;
};

class BodyTimedGraph {
public:
    BodyTimedGraph() {
        CUDA_CHECK(cudaEventCreate(&body_start_));
        CUDA_CHECK(cudaEventCreate(&body_stop_));
        CUDA_CHECK(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming));
    }

    ~BodyTimedGraph() {
        if (exec_ != nullptr) { cudaGraphExecDestroy(exec_); }
        if (graph_ != nullptr) { cudaGraphDestroy(graph_); }
        if (body_start_ != nullptr) { cudaEventDestroy(body_start_); }
        if (body_stop_ != nullptr) { cudaEventDestroy(body_stop_); }
        if (completion_ != nullptr) { cudaEventDestroy(completion_); }
    }

    BodyTimedGraph(const BodyTimedGraph&)            = delete;
    BodyTimedGraph& operator=(const BodyTimedGraph&) = delete;

    template <class Launch>
    void capture(CudaExecutionView cuda, Launch&& launch) {
        const auto stream = cuda.stream();
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        CUDA_CHECK(cudaEventRecordWithFlags(body_start_, stream, cudaEventRecordExternal));
        launch(cuda);
        CUDA_CHECK(cudaEventRecordWithFlags(body_stop_, stream, cudaEventRecordExternal));
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
        CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, 0));
        std::size_t nodes = 0;
        CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &nodes));
        if (nodes < 3) { throw std::runtime_error("MoE capture produced an empty graph"); }
    }

    void launch(CudaExecutionView cuda) const {
        const auto stream = cuda.stream();
        CUDA_CHECK(cudaGraphLaunch(exec_, stream));
    }

    double launch_body_timed(CudaExecutionView cuda) const {
        const auto stream = cuda.stream();
        launch(cuda);
        CUDA_CHECK(cudaEventRecord(completion_, stream));
        CUDA_CHECK(cudaEventSynchronize(completion_));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, body_start_, body_stop_));
        return static_cast<double>(milliseconds) * 1000.0;
    }

private:
    cudaGraph_t graph_      = nullptr;
    cudaGraphExec_t exec_   = nullptr;
    cudaEvent_t body_start_ = nullptr;
    cudaEvent_t body_stop_  = nullptr;
    cudaEvent_t completion_ = nullptr;
};

inline Stats summarize(std::vector<double> samples) {
    if (samples.empty()) { throw std::invalid_argument("cannot summarize an empty sample set"); }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double fraction) {
        const std::size_t index =
            std::min(samples.size() - 1,
                     static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1)));
        return samples[index];
    };
    return {percentile(0.50), samples.front(), percentile(0.95)};
}

} // namespace ninfer::bench::moe
