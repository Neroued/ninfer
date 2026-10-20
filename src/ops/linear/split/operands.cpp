#include "ops/linear/split/dispatch.h"
#include "ops/common/host/operands.h"

#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::linear_split {
namespace {
namespace operand         = common::operand;
constexpr const char* kOp = "linear_split";

void require(bool condition, const char* reason) {
    if (!condition) throw std::invalid_argument(std::string(kOp) + ": " + reason);
}

void matrix(const Tensor& t, int rows, int columns) {
    require(t.dtype == DType::BF16 && t.ne[0] == rows && t.ne[1] == columns && t.ne[2] == 1 &&
                t.ne[3] == 1 && t.is_contiguous() && operand::aligned(t.data, 16),
            "invalid BF16 activation/output matrix");
}

void validate_weight(const Weight& w) {
    require(w.ndim == 2 && w.shape[0] == w.n && w.shape[1] == w.k && w.padded_shape[0] == w.n &&
                w.padded_shape[1] == w.k,
            "invalid weight geometry");
    if (w.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        (void)common::validate_fp8_weight(w, kOp);
        return;
    }
    if (w.qtype == QType::NVFP4) {
        (void)common::validate_nvfp4_weight(w, kOp);
        return;
    }
    require(operand::aligned(w.qdata, 16), "weight data must be 16-byte aligned");
    if (w.qtype == QType::BF16) {
        require(w.layout == QuantLayout::Contiguous && w.group == 0 && w.group_size == 0 &&
                    !w.scales && !w.qhigh && !w.high_plane_bytes &&
                    w.payload_bytes >= std::uint64_t(w.n) * w.k * 2,
                "invalid BF16 weight");
        return;
    }
    const bool q8   = w.qtype == QType::Q8_G32_FP16;
    const bool q5   = w.qtype == QType::Q5_G64_FP16;
    const int group = q8 ? 32 : 64;
    require(w.layout == QuantLayout::RowSplit && w.group == group && w.group_size == group &&
                w.scale_dtype == DType::FP16 && operand::aligned(w.scales, q8 ? 16 : 4) &&
                (q5 ? operand::aligned(w.qhigh, 16) && w.high_plane_bytes > 0
                    : !w.qhigh && !w.high_plane_bytes),
            "invalid RowSplit weight");
}

std::array<operand::Range, 4> weight_ranges(const Weight& w) {
    const auto elements = std::uint64_t(w.n) * w.k;
    std::uint64_t codes = elements, scales = 0, high = 0;
    switch (w.qtype) {
    case QType::BF16:
        codes = elements * 2;
        break;
    case QType::Q4_G64_FP16:
        codes  = elements / 2;
        scales = elements / 64 * 2;
        break;
    case QType::Q5_G64_FP16:
        codes  = elements / 2;
        scales = elements / 64 * 2;
        high   = elements / 8;
        break;
    case QType::Q8_G32_FP16:
        scales = elements / 32 * 2;
        break;
    case QType::FP8_E4M3FN_ROW_BF16:
        scales = std::uint64_t(w.n) * 2;
        break;
    case QType::NVFP4:
        codes  = elements / 2;
        scales = elements / 16;
        break;
    default:
        throw std::logic_error("linear_split: invalid admitted weight format");
    }
    return {operand::range(w.qdata, codes, kOp),
            scales ? operand::range(w.scales, scales, kOp) : operand::Range{},
            high ? operand::range(w.qhigh, high, kOp) : operand::Range{},
            w.payload ? operand::range(w.payload, w.payload_bytes, kOp) : operand::Range{}};
}

void disjoint_weights(operand::Range range, Weights weights) {
    for (const auto* weight : {&weights.first, weights.second})
        if (weight)
            for (const auto plane : weight_ranges(*weight)) operand::disjoint(range, plane, kOp);
}
} // namespace

Weights describe_weights(const ProjectionWeights& weights) {
    if (const auto* single = std::get_if<SingleProjectionWeight>(&weights))
        return {single->weight, nullptr, single->policy};
    const auto& pair = std::get<PairedProjectionWeights>(weights);
    return {pair.first, &pair.second, LinearPolicy::A16Only};
}

void validate_interval(std::int32_t first, std::int32_t last) {
    require(first > 0 && last >= first, "invalid positive column interval");
}

void validate_operands(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
                       const SplitShape& shape, CudaExecutionView execution) {
    require(x.device == execution.device() && weights.first.device == execution.device() &&
                (!weights.second || weights.second->device == execution.device()),
            "input and weights must belong to the execution device");
    validate_interval(x.ne[1], x.ne[1]);
    matrix(x, shape.input_rows, x.ne[1]);
    validate_weight(weights.first);
    if (weights.second) validate_weight(*weights.second);
    require(outputs.size() == shape.segments.size(), "invalid output segment count");
    const auto input = operand::range(x.data, x.bytes(), kOp);
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        require(outputs[i].device == execution.device(), "output belongs to another device");
        matrix(outputs[i], shape.segments[i], x.ne[1]);
        const auto out = operand::range(outputs[i].data, outputs[i].bytes(), kOp);
        operand::disjoint(out, input, kOp);
        disjoint_weights(out, weights);
        for (std::size_t j = 0; j < i; ++j)
            operand::disjoint(out, operand::range(outputs[j].data, outputs[j].bytes(), kOp), kOp);
    }
}

void validate_scratch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
                      MemorySpan scratch) {
    const auto range = operand::range(scratch.data, scratch.bytes, kOp);
    operand::disjoint(range, operand::range(x.data, x.bytes(), kOp), kOp);
    disjoint_weights(range, weights);
    for (const auto& out : outputs)
        operand::disjoint(range, operand::range(out.data, out.bytes(), kOp), kOp);
}
} // namespace ninfer::ops::detail::linear_split
