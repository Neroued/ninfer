#include "ninfer/ops/linear_split.h"
#include "ops/linear/split/dispatch.h"

#include <stdexcept>

namespace ninfer::ops {
namespace {
namespace split = detail::linear_split;

void execute(const Tensor& x, split::Weights weights, std::span<const Tensor> outputs,
             MemoryArena* workspace, CudaExecutionView execution) {
    const auto& shape = split::select_split_shape(
        {linear_profile({weights.first, weights.policy}),
         weights.second ? std::optional{linear_profile({*weights.second})} : std::nullopt},
        outputs);
    split::validate_operands(x, weights, outputs, shape, execution);
    const auto bytes = shape.capacity ? shape.capacity(weights.policy, x.ne[1], x.ne[1]) : 0;
    if (!bytes) {
        shape.launch(x, weights, outputs, nullptr, execution.stream());
        return;
    }
    if (!workspace) throw std::logic_error("linear_split: A16 route requested workspace");
    auto scope         = workspace->scope();
    const auto backing = workspace->alloc_bytes(bytes, 256);
    split::validate_scratch(x, weights, outputs, backing);
    MemoryArena local(backing);
    shape.launch(x, weights, outputs, &local, execution.stream());
}
} // namespace

void linear_split(const Tensor& x, const ProjectionWeights& weights,
                  std::span<const Tensor> outputs, MemoryArena& workspace,
                  CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("linear_split: workspace belongs to another device");
    }
    execute(x, split::describe_weights(weights), outputs, &workspace, execution);
}

void linear_split(const Tensor& x, const Weight& weight, std::span<const Tensor> outputs,
                  CudaExecutionView execution) {
    execute(x, {weight}, outputs, nullptr, execution);
}

std::size_t linear_split_workspace_capacity_bytes(const ProjectionProfile& weights,
                                                  std::span<const std::int32_t> output_rows,
                                                  std::int32_t first, std::int32_t last) {
    const auto& shape = split::select_split_shape(weights, output_rows);
    split::validate_interval(first, last);
    return shape.capacity ? shape.capacity(weights.first.policy, first, last) : 0;
}
} // namespace ninfer::ops
