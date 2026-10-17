#include "core/decode_graph.h"

#include "core/device.h"
#include "core/nvtx.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void check_graph(cudaError_t error, const char* operation) {
    if (error == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
        throw DeviceAllocationError(operation);
    }
    cuda_check(error, operation, __FILE__, __LINE__);
}

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

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_) {
    other.graph_ = nullptr;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    graph_ = other.graph_;

    other.graph_ = nullptr;
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    nvtx::ScopedRange capture_range(nvtx::Name::CudaGraphCapture, nvtx::Category::Graph);
    reset();

    DecodeGraphCapture capture(stream);
    body();
    *this = capture.finish();
}

DecodeGraphChild DecodeGraphDefinition::append_to(cudaStream_t stream) const {
    cudaStreamCaptureStatus status;
    cudaGraph_t parent                  = nullptr;
    const cudaGraphNode_t* dependencies = nullptr;
    std::size_t count                   = 0;
    check_graph(
        cudaStreamGetCaptureInfo(stream, &status, nullptr, &parent, &dependencies, nullptr, &count),
        "cudaStreamGetCaptureInfo");
    if (!ready() || status != cudaStreamCaptureStatusActive)
        throw std::logic_error("appending a child requires a definition and active capture");
    DecodeGraphChild child;
    check_graph(cudaGraphAddChildGraphNode(&child.node, parent, dependencies, count, graph_),
                "cudaGraphAddChildGraphNode");
    check_graph(cudaStreamUpdateCaptureDependencies(stream, &child.node, nullptr, 1,
                                                    cudaStreamSetCaptureDependencies),
                "cudaStreamUpdateCaptureDependencies");
    return child;
}

DecodeGraphCapture::DecodeGraphCapture(cudaStream_t stream) : stream_(stream) {
    check_graph(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
                "cudaStreamBeginCapture");
}

DecodeGraphCapture::~DecodeGraphCapture() {
    if (stream_) discard_capture(stream_);
}

DecodeGraphDefinition DecodeGraphCapture::finish() {
    if (!stream_) throw std::logic_error("CUDA Graph capture is already finished");
    DecodeGraphDefinition definition;
    const auto error = cudaStreamEndCapture(stream_, &definition.graph_);
    stream_          = nullptr;
    check_graph(error, "cudaStreamEndCapture");
    return definition;
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

void DecodeGraphDefinition::reset() noexcept { destroy_graph(graph_); }

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_) {
    other.exec_ = nullptr;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    exec_       = other.exec_;
    other.exec_ = nullptr;
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange instantiate_range(nvtx::Name::CudaGraphInstantiate, nvtx::Category::Graph);
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    reset();

    cudaGraphExec_t exec  = nullptr;
    const cudaError_t err = cudaGraphInstantiate(&exec, definition.graph_, 0);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        check_graph(err, "cudaGraphInstantiate");
    }
    exec_ = exec;
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange update_range(nvtx::Name::CudaGraphUpdate, nvtx::Category::Graph);
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = cudaGraphExecUpdate(exec_, definition.graph_, &result);
    if (err == cudaErrorMemoryAllocation) check_graph(err, "cudaGraphExecUpdate");
    if (err != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
        throw std::runtime_error(
            "CUDA Graph executable update failed: " + std::string(cudaGetErrorName(err)) +
            " (update result " + std::to_string(static_cast<int>(result.result)) + ")");
    }
}

void DecodeGraphExecutable::update_child(DecodeGraphChild child,
                                         const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange update_range(nvtx::Name::CudaGraphUpdate, nvtx::Category::Graph);
    if (!ready() || !child.node || !definition.ready())
        throw std::logic_error("CUDA Graph child update requires a node and definition");
    check_graph(cudaGraphExecChildGraphNodeSetParams(exec_, child.node, definition.graph_),
                "cudaGraphExecChildGraphNodeSetParams");
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    nvtx::ScopedRange upload_range(nvtx::Name::CudaGraphUpload, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    check_graph(cudaGraphUpload(exec_, stream), "cudaGraphUpload");
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    // This range executes for every replay; ranges in the captured body execute only at capture.
    nvtx::ScopedRange launch_range(nvtx::Name::CudaGraphLaunch, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    check_graph(cudaGraphLaunch(exec_, stream), "cudaGraphLaunch");
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraphExecutable::reset() noexcept { destroy_graph_exec(exec_); }

} // namespace ninfer
