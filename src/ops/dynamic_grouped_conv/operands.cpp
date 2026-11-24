#include "ops/dynamic_grouped_conv/operands.h"
#include "ops/common/host/operands.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::dynamic_grouped_conv {
namespace {
void require_tensor(const Tensor& tensor, DType dtype, std::int32_t d0, std::int32_t d1,
                    std::int32_t d2, std::int32_t d3, const char* op, const char* label,
                    Device device) {
    common::operand::device(tensor.device, device, op, label);
    if (tensor.dtype != dtype || tensor.ne[0] != d0 || tensor.ne[1] != d1 || tensor.ne[2] != d2 ||
        tensor.ne[3] != d3 || !tensor.is_contiguous() ||
        !common::operand::aligned(tensor.data, 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
}

void validate_prepare_weight(const Weight& weight) {
    constexpr std::uint64_t kPayloadBytes =
        static_cast<std::uint64_t>(kCoefficientRows) * kHidden * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < kPayloadBytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != kCoefficientRows || weight.k != kHidden ||
        weight.shape[0] != kCoefficientRows || weight.shape[1] != kHidden ||
        weight.padded_shape[0] != kCoefficientRows || weight.padded_shape[1] != kHidden ||
        weight.qhigh != nullptr || weight.scales != nullptr || weight.group_size != 0 ||
        weight.group != 0 || !common::operand::aligned(weight.qdata, 16)) {
        throw std::invalid_argument(
            "dynamic grouped conv prepare: invalid kernel_projection_weight");
    }
}

std::uint64_t required_q8_payload_bytes(std::int32_t input_rows) {
    const std::uint64_t rows       = kHidden;
    const std::uint64_t columns    = static_cast<std::uint64_t>(input_rows);
    const std::uint64_t code_bytes = rows * columns;
    const std::uint64_t groups     = columns / 32U;
    return code_bytes + rows * groups * sizeof(std::uint16_t);
}

void validate_add_weight(const Weight& weight, std::int32_t input_rows) {
    const std::uint64_t payload_bytes = required_q8_payload_bytes(input_rows);
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != kHidden || weight.k != input_rows ||
        weight.shape[0] != kHidden || weight.shape[1] != input_rows || weight.shape[2] != 1 ||
        weight.shape[3] != 1 || weight.padded_shape[0] != kHidden ||
        weight.padded_shape[1] != input_rows || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1 || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < payload_bytes || !common::operand::aligned(weight.qdata, 16) ||
        !common::operand::aligned(weight.scales, 16)) {
        throw std::invalid_argument("linear dynamic grouped conv add: invalid projection_weight");
    }
}

void require_add_nonoverlap(const Tensor& x, const Weight& projection_weight,
                            const Tensor& base_kernel, const Tensor& finish_delta,
                            const Tensor& residual, const MemoryArena& workspace) {
    const std::size_t code_bytes =
        static_cast<std::size_t>(kHidden) * static_cast<std::size_t>(x.ne[0]);
    const std::size_t scale_bytes = static_cast<std::size_t>(kHidden) *
                                    static_cast<std::size_t>(x.ne[0] / 32) * sizeof(std::uint16_t);
    const std::array<common::operand::Range, 7> ranges{{
        common::operand::range(x.data, x.bytes(), "dynamic grouped conv"),
        common::operand::range(projection_weight.qdata, code_bytes, "dynamic grouped conv"),
        common::operand::range(projection_weight.scales, scale_bytes, "dynamic grouped conv"),
        common::operand::range(base_kernel.data, base_kernel.bytes(), "dynamic grouped conv"),
        common::operand::range(finish_delta.data, finish_delta.bytes(), "dynamic grouped conv"),
        common::operand::range(residual.data, residual.bytes(), "dynamic grouped conv"),
        common::operand::range(workspace.storage().data, workspace.capacity(),
                               "dynamic grouped conv"),
    }};
    common::operand::disjoint(ranges, kAddOp);
}

void require_prepare_nonoverlap(const Tensor& residual, const Tensor& norm_weight,
                                const Tensor& base_kernel, const Weight& kernel_projection_weight,
                                const Tensor& prepared, const Tensor& finish_delta,
                                const MemoryArena& workspace) {
    constexpr std::size_t kWeightBytes =
        static_cast<std::size_t>(kCoefficientRows) * kHidden * sizeof(std::uint16_t);
    const std::array<common::operand::Range, 7> ranges{{
        common::operand::range(residual.data, residual.bytes(), "dynamic grouped conv"),
        common::operand::range(norm_weight.data, norm_weight.bytes(), "dynamic grouped conv"),
        common::operand::range(base_kernel.data, base_kernel.bytes(), "dynamic grouped conv"),
        common::operand::range(kernel_projection_weight.qdata, kWeightBytes,
                               "dynamic grouped conv"),
        common::operand::range(prepared.data, prepared.bytes(), "dynamic grouped conv"),
        common::operand::range(finish_delta.data, finish_delta.bytes(), "dynamic grouped conv"),
        common::operand::range(workspace.storage().data, workspace.capacity(),
                               "dynamic grouped conv"),
    }};
    common::operand::disjoint(ranges, kPrepareOp);
}

} // namespace

void validate_interval(int min_width, int max_width, int min_batch, int max_batch, const char* op) {
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch)
        throw std::invalid_argument(std::string(op) + ": invalid W/B interval");
}

void validate_input_rows(int input_rows) {
    if (input_rows != 4096 && input_rows != 17408)
        throw std::invalid_argument("linear dynamic grouped conv add: C must be 4096 or 17408");
}

void validate_prepare(const Tensor& residual, const Tensor& norm_weight, float eps,
                      const Tensor& base_kernel, const Weight& kernel_projection_weight,
                      const Tensor& prepared, const Tensor& finish_delta,
                      const MemoryArena& workspace, CudaExecutionView execution) {
    const std::int32_t batch_size = residual.ne[2];
    const std::int32_t width      = residual.ne[1];
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument(
            "dynamic grouped conv prepare: eps must be positive and finite");
    }
    validate_interval(width, width, batch_size, batch_size, kPrepareOp);
    require_tensor(residual, DType::BF16, kHidden, width, batch_size, 1, kPrepareOp, "residual",
                   execution.device());
    require_tensor(norm_weight, DType::BF16, kHidden, 1, 1, 1, kPrepareOp, "norm_weight",
                   execution.device());
    require_tensor(base_kernel, DType::BF16, kHidden, kTaps, kSides, 1, kPrepareOp, "base_kernel",
                   execution.device());
    require_tensor(prepared, DType::BF16, kHidden, width, batch_size, 1, kPrepareOp, "prepared",
                   execution.device());
    require_tensor(finish_delta, DType::BF16, kGroups, kTaps, width, batch_size, kPrepareOp,
                   "finish_delta", execution.device());
    common::operand::device(kernel_projection_weight.device, execution.device(), kPrepareOp,
                            "weight");
    common::operand::device(workspace.device(), execution.device(), kPrepareOp, "workspace");
    validate_prepare_weight(kernel_projection_weight);
    require_prepare_nonoverlap(residual, norm_weight, base_kernel, kernel_projection_weight,
                               prepared, finish_delta, workspace);
}

void validate_add(const Tensor& x, const Weight& projection_weight, const Tensor& base_kernel,
                  const Tensor& finish_delta, const Tensor& residual, const MemoryArena& workspace,
                  CudaExecutionView execution) {
    const std::int32_t input_rows = x.ne[0];
    const std::int32_t batch_size = x.ne[2];
    const std::int32_t width      = x.ne[1];
    validate_interval(width, width, batch_size, batch_size, kAddOp);
    validate_input_rows(input_rows);
    require_tensor(x, DType::BF16, input_rows, width, batch_size, 1, kAddOp, "x",
                   execution.device());
    require_tensor(base_kernel, DType::BF16, kHidden, kTaps, kSides, 1, kAddOp, "base_kernel",
                   execution.device());
    require_tensor(finish_delta, DType::BF16, kGroups, kTaps, width, batch_size, kAddOp,
                   "finish_delta", execution.device());
    require_tensor(residual, DType::BF16, kHidden, width, batch_size, 1, kAddOp, "residual",
                   execution.device());
    common::operand::device(projection_weight.device, execution.device(), kAddOp, "weight");
    common::operand::device(workspace.device(), execution.device(), kAddOp, "workspace");
    validate_add_weight(projection_weight, input_rows);
    require_add_nonoverlap(x, projection_weight, base_kernel, finish_delta, residual, workspace);
}

} // namespace ninfer::ops::detail::dynamic_grouped_conv
