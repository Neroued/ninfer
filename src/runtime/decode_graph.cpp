#include "ninfer/runtime/decode_graph.h"

#include "ninfer/core/device.h"

#include <cstdio>

namespace ninfer {
namespace {

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_graph_exec(cudaGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_cuda_error("cudaGraphExecDestroy", cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_cuda_error("cudaGraphDestroy", cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

void discard_capture(cudaStream_t stream) noexcept {
    cudaGraph_t discard = nullptr;
    log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &discard));
    destroy_graph(discard);
}

} // namespace

DecodeGraph::~DecodeGraph() { reset(); }

DecodeGraph::DecodeGraph(DecodeGraph&& other) noexcept
    : graph_(other.graph_), exec_(other.exec_) {
    other.graph_ = nullptr;
    other.exec_  = nullptr;
}

DecodeGraph& DecodeGraph::operator=(DecodeGraph&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    graph_ = other.graph_;
    exec_  = other.exec_;

    other.graph_ = nullptr;
    other.exec_  = nullptr;
    return *this;
}

void DecodeGraph::capture(cudaStream_t stream, const std::function<void()>& body) {
    reset();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    try {
        body();
    } catch (...) {
        discard_capture(stream);
        throw;
    }

    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        destroy_graph(graph);
        CUDA_CHECK(err);
    }

    err = cudaGraphInstantiate(&exec, graph, 0);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        destroy_graph(graph);
        CUDA_CHECK(err);
    }

    graph_ = graph;
    exec_  = exec;
}

void DecodeGraph::launch(cudaStream_t stream) { CUDA_CHECK(cudaGraphLaunch(exec_, stream)); }

bool DecodeGraph::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraph::reset() noexcept {
    destroy_graph_exec(exec_);
    destroy_graph(graph_);
}

} // namespace ninfer
