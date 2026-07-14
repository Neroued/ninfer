#pragma once

#include <cuda_runtime.h>

#include <functional>

namespace ninfer {

class DecodeGraph {
public:
    DecodeGraph() = default;
    ~DecodeGraph();

    DecodeGraph(const DecodeGraph&)            = delete;
    DecodeGraph& operator=(const DecodeGraph&) = delete;
    DecodeGraph(DecodeGraph&& other) noexcept;
    DecodeGraph& operator=(DecodeGraph&& other) noexcept;

    void capture(cudaStream_t stream, const std::function<void()>& body);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    cudaGraph_t graph_     = nullptr;
    cudaGraphExec_t exec_  = nullptr;
};

} // namespace ninfer
