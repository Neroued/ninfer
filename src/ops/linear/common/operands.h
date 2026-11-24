#pragma once

#include "ops/common/host/operands.h"
#include "core/device.h"

namespace ninfer::ops::detail {

inline void validate_bf16_projection(const Tensor& x, const Weight& w,
                                     std::span<const Tensor> outputs, const Tensor* bias,
                                     CudaExecutionView execution, const char* op) {
    namespace operand = common::operand;
    operand::device(w.device, execution.device(), op, "weight");
    if (w.qtype != QType::BF16 || w.ndim != 2 || w.n <= 0 || w.k <= 0 || w.shape[0] != w.n ||
        w.shape[1] != w.k || w.padded_shape[0] != w.n || w.padded_shape[1] != w.k ||
        w.layout != QuantLayout::Contiguous || w.group != 0 || w.group_size != 0 || w.scales ||
        w.qhigh || w.high_plane_bytes || w.payload_bytes < std::uint64_t(w.n) * w.k * 2 ||
        !operand::aligned(w.qdata, 16))
        throw std::invalid_argument(std::string(op) + ": invalid contiguous BF16 weight");
    if (x.ne[1] <= 0) throw std::invalid_argument(std::string(op) + ": columns must be positive");
    operand::tensor(x, DType::BF16, {w.k, x.ne[1]}, 16, op, "input", execution.device());
    if (bias) operand::tensor(*bias, DType::BF16, {w.n}, 2, op, "bias", execution.device());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto& out = outputs[i];
        if (out.ne[0] <= 0) throw std::invalid_argument(std::string(op) + ": invalid output rows");
        operand::tensor(out, DType::BF16, {out.ne[0], x.ne[1]}, 16, op, "output",
                        execution.device());
        const auto range = operand::range(out, op);
        operand::disjoint(range, operand::range(x, op), op);
        operand::disjoint(range, operand::range(w.qdata, std::size_t(w.n) * w.k * 2, op), op);
        if (w.payload) operand::disjoint(range, operand::range(w.payload, w.payload_bytes, op), op);
        if (bias) operand::disjoint(range, operand::range(*bias, op), op);
        for (std::size_t j = 0; j < i; ++j)
            operand::disjoint(range, operand::range(outputs[j], op), op);
    }
}

} // namespace ninfer::ops::detail
