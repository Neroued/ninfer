#pragma once

#include "core/decode_graph.h"
#include "core/device.h"

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::test {

// Capture a single chain so kernel slots are paired by dependency order.
class CapturedKernelSequence {
public:
    template <class LaunchFn>
    CapturedKernelSequence(CudaExecutionView execution, LaunchFn&& launch) {
        auto capture = std::make_shared<Capture>();
        capture->definition.capture(execution.stream(), [&] {
            std::forward<LaunchFn>(launch)(execution);
            cudaStreamCaptureStatus status{};
            CUDA_CHECK(
                cudaStreamGetCaptureInfo(execution.stream(), &status, nullptr, &capture->graph));
            if (status != cudaStreamCaptureStatusActive) {
                throw std::runtime_error("kernel sequence requires an active CUDA graph capture");
            }
        });
        capture->record_nodes();
        capture_ = std::move(capture);
    }

    [[nodiscard]] std::span<const cudaGraphNode_t> nodes() const noexcept {
        return capture_->nodes;
    }

    // CUDA owns the argument arrays and values in these parameters through the captured nodes.
    [[nodiscard]] std::span<const cudaKernelNodeParams> params() const noexcept {
        return capture_->params;
    }

private:
    friend class FlatGraphPatchHarness;

    struct Capture {
        DecodeGraphDefinition definition;
        cudaGraph_t graph = nullptr; // Owned by definition after capture.
        std::vector<cudaGraphNode_t> nodes;
        std::vector<cudaKernelNodeParams> params;

        void record_nodes() {
            std::size_t node_count = 0;
            CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));
            nodes.reserve(node_count);
            params.reserve(node_count);

            std::array<cudaGraphNode_t, 2> next{};
            auto count = next.size();
            CUDA_CHECK(cudaGraphGetRootNodes(graph, next.data(), &count));
            while (count == 1) {
                const auto node = next.front();
                cudaGraphNodeType type{};
                CUDA_CHECK(cudaGraphNodeGetType(node, &type));
                if (type != cudaGraphNodeTypeKernel) {
                    throw std::invalid_argument("kernel sequence contains a non-kernel node");
                }
                nodes.push_back(node);
                CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &params.emplace_back()));

                count = next.size();
                CUDA_CHECK(cudaGraphNodeGetDependentNodes(node, next.data(), nullptr, &count));
            }
            if (count != 0 || nodes.size() != node_count) {
                throw std::invalid_argument("kernel sequence must be a single dependency chain");
            }
        }
    };

    std::shared_ptr<const Capture> capture_;
};

// Keeps the original graph nodes alive across both update methods. Launches are asynchronous.
class FlatGraphPatchHarness {
public:
    explicit FlatGraphPatchHarness(const CapturedKernelSequence& initial) : initial_(initial) {
        cudaGraphExec_t exec = nullptr;
        const auto error     = cudaGraphInstantiate(&exec, initial_.capture_->graph, 0);
        exec_.reset(exec);
        CUDA_CHECK(error);
    }

    void patch_and_launch(const CapturedKernelSequence& target, CudaExecutionView execution) {
        const auto nodes  = initial_.nodes();
        const auto params = target.params();
        if (params.size() != nodes.size()) {
            throw std::invalid_argument("graph patch requires a constant kernel node count");
        }
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            CUDA_CHECK(cudaGraphExecKernelNodeSetParams(exec_.get(), nodes[i], &params[i]));
        }
        upload_and_launch(execution.stream());
    }

    void update_whole_graph_and_launch(const CapturedKernelSequence& target,
                                       CudaExecutionView execution) {
        cudaGraphExecUpdateResultInfo result{};
        const auto error = cudaGraphExecUpdate(exec_.get(), target.capture_->graph, &result);
        if (error != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
            throw std::runtime_error(
                "cudaGraphExecUpdate failed: " + std::string(cudaGetErrorString(error)) +
                " (update result " + std::to_string(static_cast<int>(result.result)) + ")");
        }
        upload_and_launch(execution.stream());
    }

private:
    struct DestroyExec {
        void operator()(cudaGraphExec_t exec) const noexcept { (void)cudaGraphExecDestroy(exec); }
    };

    void upload_and_launch(cudaStream_t stream) {
        CUDA_CHECK(cudaGraphUpload(exec_.get(), stream));
        CUDA_CHECK(cudaGraphLaunch(exec_.get(), stream));
    }

    CapturedKernelSequence initial_;
    std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, DestroyExec> exec_;
};

} // namespace ninfer::test
