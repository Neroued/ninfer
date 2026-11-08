#include "ninfer/ops/linear_split.h"
#include "ops/linear/split/dispatch.h"

namespace ninfer::ops {
namespace {
namespace split = detail::linear_split;

void execute_bias(const Tensor& x, const Weight& weight, const Tensor& bias,
                  std::span<const Tensor> outputs, LinearPolicy policy, MemoryArena* workspace,
                  CudaExecutionView execution) {
    const auto& shape =
        split::select_split_bias_shape(weight.qtype, weight.n, weight.k, policy, outputs);
    split::validate_operands(x, {weight, nullptr, policy}, outputs, shape.segments, execution);
    split::validate_bias(bias, weight.n, outputs, execution);
    const auto bytes = shape.capacity ? shape.capacity(policy, x.ne[1], x.ne[1]) : 0;
    if (!bytes) {
        shape.launch(x, weight, bias, outputs, policy, nullptr, execution.stream());
        return;
    }
    if (!workspace) throw std::logic_error("linear_split bias: requested route requires workspace");
    auto scope         = workspace->scope();
    const auto backing = workspace->alloc_bytes(bytes, 256);
    split::validate_scratch(x, {weight, nullptr, policy}, outputs, backing);
    split::validate_bias_scratch(bias, backing);
    MemoryArena local(backing);
    shape.launch(x, weight, bias, outputs, policy, &local, execution.stream());
}
} // namespace

void linear_split(const Tensor& x, const Weight& weight, const Tensor& bias,
                  std::span<const Tensor> outputs, LinearPolicy policy, MemoryArena& workspace,
                  CudaExecutionView execution) {
    if (workspace.device() != execution.device())
        throw std::invalid_argument("linear_split bias: workspace belongs to another device");
    execute_bias(x, weight, bias, outputs, policy, &workspace, execution);
}

void linear_split(const Tensor& x, const Weight& weight, const Tensor& bias,
                  std::span<const Tensor> outputs, CudaExecutionView execution) {
    if (weight.qtype != QType::BF16)
        throw std::invalid_argument("linear_split bias: workspace-free form requires BF16 weights");
    execute_bias(x, weight, bias, outputs, LinearPolicy::A16Only, nullptr, execution);
}

std::size_t linear_split_bias_workspace_capacity_bytes(QType type, std::int32_t n, std::int32_t k,
                                                       std::span<const std::int32_t> output_rows,
                                                       LinearPolicy policy, std::int32_t first,
                                                       std::int32_t last) {
    const auto& shape = split::select_split_bias_shape(type, n, k, policy, output_rows);
    split::validate_interval(first, last);
    return shape.capacity ? shape.capacity(policy, first, last) : 0;
}
} // namespace ninfer::ops
