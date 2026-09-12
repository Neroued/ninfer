#include "ninfer/ops/dynamic_grouped_conv.h"

#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.h"
#include "ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_plan.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHidden          = 5120;
constexpr std::int32_t kGroups          = 320;
constexpr std::int32_t kTaps            = 2;
constexpr std::int32_t kSides           = 2;
constexpr std::int32_t kCoefficientRows = kGroups * kTaps * kSides;
constexpr const char* kPrepareOp        = "dynamic grouped conv prepare";
constexpr const char* kAddOp            = "linear dynamic grouped conv add";

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t d0, std::int32_t d1,
                    std::int32_t d2, std::int32_t d3, const char* op, const char* label) {
    if (tensor.dtype != dtype || tensor.ne[0] != d0 || tensor.ne[1] != d1 || tensor.ne[2] != d2 ||
        tensor.ne[3] != d3 || !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
}

void require_kernel_projection_weight(const Weight& weight) {
    constexpr std::uint64_t kPayloadBytes =
        static_cast<std::uint64_t>(kCoefficientRows) * kHidden * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < kPayloadBytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != kCoefficientRows || weight.k != kHidden ||
        weight.shape[0] != kCoefficientRows || weight.shape[1] != kHidden ||
        weight.padded_shape[0] != kCoefficientRows || weight.padded_shape[1] != kHidden ||
        weight.qhigh != nullptr || weight.scales != nullptr || weight.group_size != 0 ||
        weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(
            "dynamic grouped conv prepare: invalid kernel_projection_weight");
    }
}

std::uint64_t required_w8_payload_bytes(std::int32_t input_rows) {
    const std::uint64_t rows       = kHidden;
    const std::uint64_t columns    = static_cast<std::uint64_t>(input_rows);
    const std::uint64_t code_bytes = rows * columns;
    const std::uint64_t groups     = columns / 32U;
    return code_bytes + rows * groups * sizeof(std::uint16_t);
}

void require_finish_projection_weight(const Weight& weight, std::int32_t input_rows) {
    const std::uint64_t payload_bytes = required_w8_payload_bytes(input_rows);
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != kHidden || weight.k != input_rows ||
        weight.shape[0] != kHidden || weight.shape[1] != input_rows || weight.shape[2] != 1 ||
        weight.shape[3] != 1 || weight.padded_shape[0] != kHidden ||
        weight.padded_shape[1] != input_rows || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1 || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < payload_bytes || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument("linear dynamic grouped conv add: invalid projection_weight");
    }
}

struct Range {
    const void* pointer;
    std::size_t bytes;
    const char* label;
};

bool overlaps(const Range& lhs, const Range& rhs) {
    if (lhs.bytes == 0 || rhs.bytes == 0 || lhs.pointer == nullptr || rhs.pointer == nullptr) {
        return false;
    }
    const std::uintptr_t lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.pointer);
    const std::uintptr_t rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.pointer);
    if (lhs.bytes > std::numeric_limits<std::uintptr_t>::max() - lhs_begin ||
        rhs.bytes > std::numeric_limits<std::uintptr_t>::max() - rhs_begin) {
        throw std::overflow_error("dynamic grouped conv: operand range overflows");
    }
    return lhs_begin < rhs_begin + rhs.bytes && rhs_begin < lhs_begin + lhs.bytes;
}

void require_finish_nonoverlap(const Tensor& x, const Weight& projection_weight,
                               const Tensor& base_kernel, const Tensor& finish_delta,
                               const Tensor& residual, const WorkspaceArena& workspace) {
    const std::size_t code_bytes =
        static_cast<std::size_t>(kHidden) * static_cast<std::size_t>(x.ne[0]);
    const std::size_t scale_bytes = static_cast<std::size_t>(kHidden) *
                                    static_cast<std::size_t>(x.ne[0] / 32) * sizeof(std::uint16_t);
    const std::array<Range, 7> ranges{{
        {x.data, x.bytes(), "x"},
        {projection_weight.qdata, code_bytes, "projection codes"},
        {projection_weight.scales, scale_bytes, "projection scales"},
        {base_kernel.data, base_kernel.bytes(), "base_kernel"},
        {finish_delta.data, finish_delta.bytes(), "finish_delta"},
        {residual.data, residual.bytes(), "residual"},
        {workspace.base(), workspace.capacity(), "workspace"},
    }};
    for (std::size_t first = 0; first < ranges.size(); ++first) {
        for (std::size_t second = first + 1; second < ranges.size(); ++second) {
            if (overlaps(ranges[first], ranges[second])) {
                throw std::invalid_argument(std::string(kAddOp) + ": " + ranges[first].label +
                                            " overlaps " + ranges[second].label);
            }
        }
    }
}

void require_nonoverlap(const Tensor& residual, const Tensor& norm_weight,
                        const Tensor& base_kernel, const Weight& kernel_projection_weight,
                        const Tensor& prepared, const Tensor& finish_delta,
                        const WorkspaceArena& workspace) {
    constexpr std::size_t kWeightBytes =
        static_cast<std::size_t>(kCoefficientRows) * kHidden * sizeof(std::uint16_t);
    const std::array<Range, 7> ranges{{
        {residual.data, residual.bytes(), "residual"},
        {norm_weight.data, norm_weight.bytes(), "norm_weight"},
        {base_kernel.data, base_kernel.bytes(), "base_kernel"},
        {kernel_projection_weight.qdata, kWeightBytes, "kernel_projection_weight"},
        {prepared.data, prepared.bytes(), "prepared"},
        {finish_delta.data, finish_delta.bytes(), "finish_delta"},
        {workspace.base(), workspace.capacity(), "workspace"},
    }};
    for (std::size_t first = 0; first < ranges.size(); ++first) {
        for (std::size_t second = first + 1; second < ranges.size(); ++second) {
            if (overlaps(ranges[first], ranges[second])) {
                throw std::invalid_argument(std::string("dynamic grouped conv prepare: ") +
                                            ranges[first].label + " overlaps " +
                                            ranges[second].label);
            }
        }
    }
}

} // namespace

std::size_t rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
    std::int32_t min_width, std::int32_t max_width, std::int32_t min_batch_size,
    std::int32_t max_batch_size) {
    return detail::bf16_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
        min_width, max_width, min_batch_size, max_batch_size);
}

void rmsnorm_dynamic_grouped_conv_prepare(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const Tensor& base_kernel,
                                          const Weight& kernel_projection_weight, Tensor& prepared,
                                          Tensor& finish_delta, WorkspaceArena& workspace,
                                          cudaStream_t stream) {
    const std::int32_t batch_size = residual.ne[2];
    const std::int32_t width      = residual.ne[1];
    if (width < 2 || width > 16) {
        throw std::invalid_argument("dynamic grouped conv prepare: W must be in [2,16]");
    }
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument(
            "dynamic grouped conv prepare: eps must be positive and finite");
    }
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("dynamic grouped conv prepare: B must be in [1,8]");
    }
    require_tensor(residual, DType::BF16, kHidden, width, batch_size, 1, kPrepareOp, "residual");
    require_tensor(norm_weight, DType::BF16, kHidden, 1, 1, 1, kPrepareOp, "norm_weight");
    require_tensor(base_kernel, DType::BF16, kHidden, kTaps, kSides, 1, kPrepareOp, "base_kernel");
    require_tensor(prepared, DType::BF16, kHidden, width, batch_size, 1, kPrepareOp, "prepared");
    require_tensor(finish_delta, DType::BF16, kGroups, kTaps, width, batch_size, kPrepareOp,
                   "finish_delta");
    require_kernel_projection_weight(kernel_projection_weight);
    require_nonoverlap(residual, norm_weight, base_kernel, kernel_projection_weight, prepared,
                       finish_delta, workspace);

    detail::bf16_dynamic_grouped_conv_prepare_dispatch(residual, norm_weight, eps, base_kernel,
                                                       kernel_projection_weight, prepared,
                                                       finish_delta, workspace, stream);
}

std::size_t linear_dynamic_grouped_conv_add_workspace_capacity_bytes(std::int32_t input_rows,
                                                                     std::int32_t min_width,
                                                                     std::int32_t max_width,
                                                                     std::int32_t min_batch_size,
                                                                     std::int32_t max_batch_size) {
    return detail::w8_linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
        input_rows, min_width, max_width, min_batch_size, max_batch_size);
}

void linear_dynamic_grouped_conv_add(const Tensor& x, const Weight& projection_weight,
                                     const Tensor& base_kernel, const Tensor& finish_delta,
                                     Tensor& residual, WorkspaceArena& workspace,
                                     cudaStream_t stream) {
    const std::int32_t input_rows = x.ne[0];
    const std::int32_t batch_size = x.ne[2];
    const std::int32_t width      = x.ne[1];
    if (width < 2 || width > 16)
        throw std::invalid_argument("linear dynamic grouped conv add: W must be in [2,16]");
    if (input_rows != 4096 && input_rows != 17408) {
        throw std::invalid_argument("linear dynamic grouped conv add: C must be 4096 or 17408");
    }
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("linear dynamic grouped conv add: B must be in [1,8]");
    }
    require_tensor(x, DType::BF16, input_rows, width, batch_size, 1, kAddOp, "x");
    require_tensor(base_kernel, DType::BF16, kHidden, kTaps, kSides, 1, kAddOp, "base_kernel");
    require_tensor(finish_delta, DType::BF16, kGroups, kTaps, width, batch_size, kAddOp,
                   "finish_delta");
    require_tensor(residual, DType::BF16, kHidden, width, batch_size, 1, kAddOp, "residual");
    require_finish_projection_weight(projection_weight, input_rows);
    require_finish_nonoverlap(x, projection_weight, base_kernel, finish_delta, residual, workspace);

    detail::w8_linear_dynamic_grouped_conv_add_dispatch(x, projection_weight, base_kernel,
                                                        finish_delta, residual, workspace, stream);
}

} // namespace ninfer::ops
