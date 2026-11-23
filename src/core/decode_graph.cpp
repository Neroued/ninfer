#include "core/decode_graph.h"

#include "core/device.h"
#include "core/nvtx.h"

#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

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

cudaGraph_t active_capture(cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    cudaGraph_t graph = nullptr;
    check_graph(
        cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, nullptr, nullptr, nullptr),
        "cudaStreamGetCaptureInfo");
    if (status != cudaStreamCaptureStatusActive)
        throw std::logic_error("inline graph recording requires an active parent capture");
    return graph;
}

std::vector<cudaGraphNode_t> captured_nodes(cudaGraph_t graph) {
    std::size_t count = 0;
    check_graph(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes");
    std::vector<cudaGraphNode_t> nodes(count);
    check_graph(cudaGraphGetNodes(graph, nodes.data(), &count), "cudaGraphGetNodes");
    return nodes;
}

std::vector<cudaGraphNode_t> added_nodes(cudaGraph_t graph, std::vector<cudaGraphNode_t> before) {
    std::ranges::sort(before);
    auto nodes = captured_nodes(graph);
    std::erase_if(nodes, [&](auto node) { return std::ranges::binary_search(before, node); });
    return nodes;
}

bool cooperative_kernel(cudaGraphNode_t node) {
    cudaKernelNodeAttrValue attribute{};
    check_graph(
        cudaGraphKernelNodeGetAttribute(node, cudaKernelNodeAttributeCooperative, &attribute),
        "cudaGraphKernelNodeGetAttribute");
    return attribute.cooperative;
}

std::vector<cudaGraphNode_t> dependent_nodes(cudaGraphNode_t node) {
    std::size_t count = 0;
    check_graph(cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &count),
                "cudaGraphNodeGetDependentNodes");
    std::vector<cudaGraphNode_t> nodes(count);
    if (count)
        check_graph(cudaGraphNodeGetDependentNodes(node, nodes.data(), nullptr, &count),
                    "cudaGraphNodeGetDependentNodes");
    return nodes;
}

[[noreturn]] void invalid_kernel_chain() {
    throw std::logic_error("graph kernel slice requires a non-empty single dependency chain");
}

template <class Contains>
std::vector<cudaGraphNode_t> kernel_chain(cudaGraphNode_t root, std::size_t size,
                                          Contains contains) {
    std::vector<cudaGraphNode_t> nodes;
    nodes.reserve(size);
    for (auto node = root; node;) {
        cudaGraphNodeType type;
        check_graph(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
        if (type != cudaGraphNodeTypeKernel || cooperative_kernel(node))
            throw std::logic_error("graph kernel slice requires ordinary kernel nodes");
        nodes.push_back(node);
        cudaGraphNode_t next = nullptr;
        for (auto dependent : dependent_nodes(node)) {
            if (!contains(dependent)) continue;
            if (next) invalid_kernel_chain();
            next = dependent;
        }
        node = next;
    }
    if (nodes.empty() || nodes.size() != size) invalid_kernel_chain();
    return nodes;
}

} // namespace

void DecodeGraphSection::begin(cudaStream_t stream) {
    nodes_   = captured_nodes(active_capture(stream));
    enabled_ = true;
}

void DecodeGraphSection::end(cudaStream_t stream) {
    auto added = added_nodes(active_capture(stream), std::move(nodes_));
    nodes_.clear();
    cooperative_.clear();
    for (const auto node : added) {
        cudaGraphNodeType type;
        check_graph(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
        if (type != cudaGraphNodeTypeKernel && type != cudaGraphNodeTypeMemcpy &&
            type != cudaGraphNodeTypeMemset)
            throw std::logic_error(
                "a switchable graph section must contain only kernels or copies");
        if (type == cudaGraphNodeTypeKernel && cooperative_kernel(node)) {
            auto& saved = cooperative_.emplace_back(node, cudaKernelNodeParams{});
            check_graph(cudaGraphKernelNodeGetParams(node, &saved.original),
                        "cudaGraphKernelNodeGetParams");
            continue;
        }
        nodes_.push_back(node);
    }
}

void DecodeGraphSection::capture(cudaStream_t stream, const std::function<void()>& body) {
    begin(stream);
    body();
    end(stream);
}

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

void DecodeGraphKernelPatch::capture(cudaStream_t stream, const std::function<void()>& body) {
    reset();
    DecodeGraphDefinition definition;
    definition.capture(stream, body);
    std::size_t size = 0, roots = 0;
    check_graph(cudaGraphGetNodes(definition.graph_, nullptr, &size), "cudaGraphGetNodes");
    check_graph(cudaGraphGetRootNodes(definition.graph_, nullptr, &roots), "cudaGraphGetRootNodes");
    if (roots != 1) invalid_kernel_chain();
    cudaGraphNode_t root = nullptr;
    check_graph(cudaGraphGetRootNodes(definition.graph_, &root, &roots), "cudaGraphGetRootNodes");
    const auto nodes = kernel_chain(root, size, [](auto) { return true; });
    std::vector<cudaKernelNodeParams> params;
    params.reserve(size);
    for (auto node : nodes)
        check_graph(cudaGraphKernelNodeGetParams(node, &params.emplace_back()),
                    "cudaGraphKernelNodeGetParams");
    definition_ = std::move(definition);
    params_     = std::move(params);
}

void DecodeGraphKernelPatch::reset() noexcept {
    params_.clear();
    definition_.reset();
}

void DecodeGraphKernelSlice::begin(cudaStream_t stream) {
    if (capture_) throw std::logic_error("a graph kernel slice is already recording");
    const auto graph = active_capture(stream);
    before_          = captured_nodes(graph);
    nodes_.clear();
    capture_ = graph;
}

void DecodeGraphKernelSlice::end(cudaStream_t stream) {
    const auto graph = std::exchange(capture_, nullptr);
    auto before      = std::move(before_);
    if (!graph || graph != active_capture(stream))
        throw std::logic_error("a graph kernel slice must end in its parent capture");
    auto added = added_nodes(graph, std::move(before));
    if (added.empty()) invalid_kernel_chain();
    std::ranges::sort(added);
    const auto contains = [&](auto node) { return std::ranges::binary_search(added, node); };
    // External incoming edges are irrelevant. The root has no predecessor in the slice.
    auto roots = added;
    for (auto node : added)
        for (auto dependent : dependent_nodes(node))
            if (const auto it = std::ranges::lower_bound(added, dependent);
                it != added.end() && *it == dependent)
                roots[it - added.begin()] = nullptr;
    std::erase(roots, nullptr);
    if (roots.size() != 1) invalid_kernel_chain();
    nodes_ = kernel_chain(roots.front(), added.size(), contains);
}

void DecodeGraphKernelSlice::capture(cudaStream_t stream, const std::function<void()>& body) {
    begin(stream);
    try {
        body();
    } catch (...) {
        capture_ = nullptr;
        before_.clear();
        throw;
    }
    end(stream);
}

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

void DecodeGraphExecutable::update_slice(const DecodeGraphKernelSlice& slice,
                                         const DecodeGraphKernelPatch& patch) {
    if (!ready() || !patch.ready() || slice.size() != patch.params().size())
        throw std::logic_error("CUDA Graph kernel update requires a matching slice and patch");
    nvtx::ScopedRange update_range(nvtx::Name::CudaGraphUpdate, nvtx::Category::Graph);
    const auto params = patch.params();
    for (std::size_t i = 0; i < params.size(); ++i)
        check_graph(cudaGraphExecKernelNodeSetParams(exec_, slice.nodes_[i], &params[i]),
                    "cudaGraphExecKernelNodeSetParams");
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
