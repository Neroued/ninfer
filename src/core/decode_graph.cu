#include "core/decode_graph.h"
#include "core/device.h"

#include <stdexcept>

namespace ninfer {
namespace {
__global__ void idle_graph_kernel() {}
} // namespace

void DecodeGraphSection::set_enabled(const DecodeGraphExecutable& executable, bool enabled) {
    if (!executable.ready()) throw std::logic_error("graph section has no executable");
    if (enabled == enabled_) return;
    for (const auto node : nodes_)
        CUDA_CHECK(cudaGraphNodeSetEnabled(executable.exec_, node, enabled));

    // Disabling cooperative launches faults on GB10. Preserve the cooperative node attribute
    // and restore its captured arguments from the enclosing graph definition when re-enabled.
    cudaKernelNodeParams idle{};
    idle.func     = reinterpret_cast<void*>(idle_graph_kernel);
    idle.gridDim  = dim3(1);
    idle.blockDim = dim3(1);
    for (const auto& kernel : cooperative_)
        CUDA_CHECK(cudaGraphExecKernelNodeSetParams(executable.exec_, kernel.node,
                                                    enabled ? &kernel.original : &idle));
    enabled_ = enabled;
}

} // namespace ninfer
