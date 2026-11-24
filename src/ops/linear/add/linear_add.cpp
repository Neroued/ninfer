#include "ninfer/ops/linear_add.h"
#include "ops/common/host/operands.h"
#include "ops/linear/add/dispatch.h"
#include "ops/linear/add/fp8/dispatch.h"
#include "ops/linear/add/nvfp4/dispatch.h"

#include <stdexcept>

namespace ninfer::ops {
namespace {
namespace operand = common::operand;

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_add: invalid compute policy");
}

detail::LinearAddLaunch select_a16(QType type, int rows, int k, int tokens) {
    switch (type) {
    case QType::BF16:
        return detail::select_bf16_linear_add(rows, k, tokens);
    case QType::Q4_G64_FP16:
        return detail::select_q4_linear_add(rows, k, tokens);
    case QType::Q5_G64_FP16:
        return detail::select_q5_linear_add(rows, k, tokens);
    case QType::Q8_G32_FP16:
        return detail::select_q8_linear_add(rows, k, tokens);
    default:
        throw std::invalid_argument("linear_add: unsupported weight format");
    }
}

void validate_operands(const Tensor& x, const Weight& w, const Tensor& residual,
                       CudaExecutionView execution) {
    if (x.device != execution.device() || w.device != execution.device() ||
        residual.device != execution.device()) {
        throw std::invalid_argument("linear_add: operands must belong to the execution device");
    }
    const auto valid = [](const Tensor& tensor, int rows, int tokens) {
        return tensor.dtype == DType::BF16 && tensor.ne[0] == rows && tensor.ne[1] == tokens &&
               tensor.ne[2] == 1 && tensor.ne[3] == 1 && tensor.is_contiguous() &&
               operand::aligned(tensor.data, 16);
    };
    if (w.n <= 0 || w.k <= 0 || x.ne[1] <= 0 || !valid(x, w.k, x.ne[1]) ||
        !valid(residual, w.n, x.ne[1]))
        throw std::invalid_argument("linear_add: expected aligned contiguous BF16 [K,T] and [N,T]");
    operand::disjoint(operand::range(x.data, x.bytes(), "linear_add"),
                      operand::range(residual.data, residual.bytes(), "linear_add"), "linear_add");

    if (w.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        (void)common::validate_fp8_weight(w, "linear_add");
        return;
    }
    if (w.qtype == QType::NVFP4) {
        (void)common::validate_nvfp4_weight(w, "linear_add");
        return;
    }
    if (!operand::aligned(w.qdata, 16))
        throw std::invalid_argument("linear_add: weight codes must be 16-byte aligned");
    if (w.qtype == QType::BF16) {
        if (w.layout != QuantLayout::Contiguous)
            throw std::invalid_argument("linear_add: BF16 weight must be contiguous");
        return;
    }
    const int group = w.qtype == QType::Q8_G32_FP16 ? 32 : 64;
    const bool high = w.qtype == QType::Q5_G64_FP16;
    if (w.layout != QuantLayout::RowSplit || w.scale_dtype != DType::FP16 ||
        w.group_size != group || w.group != group || w.padded_shape[0] != w.n ||
        w.padded_shape[1] != w.k || !operand::aligned(w.scales, 16) ||
        (high ? !operand::aligned(w.qhigh, 16) : w.qhigh != nullptr))
        throw std::invalid_argument("linear_add: invalid row-split weight representation");
}
} // namespace

std::size_t linear_add_workspace_capacity_bytes(QType type, std::int32_t rows, std::int32_t k,
                                                LinearPolicy policy, std::int32_t first,
                                                std::int32_t last) {
    validate_policy(policy);
    if (first <= 0 || last < first)
        throw std::invalid_argument("linear_add: invalid token interval");
    if (type == QType::FP8_E4M3FN_ROW_BF16)
        return detail::fp8_linear_add_workspace_capacity_bytes(rows, k, policy, first, last);
    if (type == QType::NVFP4)
        return detail::nvfp4_linear_add_workspace_capacity_bytes(rows, k, policy, first, last);
    (void)select_a16(type, rows, k, first);
    (void)select_a16(type, rows, k, last);
    return 0;
}

std::size_t linear_add_workspace_capacity_bytes(QType type, std::int32_t rows, std::int32_t k,
                                                std::int32_t first, std::int32_t last) {
    return linear_add_workspace_capacity_bytes(type, rows, k, LinearPolicy::A16Only, first, last);
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                MemoryArena& workspace, CudaExecutionView execution) {
    validate_policy(policy);
    validate_operands(x, w, residual, execution);
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("linear_add: workspace belongs to another device");
    }
    if (w.qtype == QType::FP8_E4M3FN_ROW_BF16)
        return detail::fp8_linear_add_dispatch(x, w, residual, policy, workspace,
                                               execution.stream());
    if (w.qtype == QType::NVFP4)
        return detail::nvfp4_linear_add_dispatch(x, w, residual, policy, workspace,
                                                 execution.stream());
    select_a16(w.qtype, w.n, w.k, x.ne[1])(x, w, residual, execution.stream());
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, MemoryArena& workspace,
                CudaExecutionView execution) {
    linear_add(x, w, residual, LinearPolicy::A16Only, workspace, execution);
}
} // namespace ninfer::ops
