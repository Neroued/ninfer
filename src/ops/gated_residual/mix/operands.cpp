#include "ops/gated_residual/mix/operands.h"
#include "ops/common/host/operands.h"

#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::hc::mix {
namespace {
constexpr const char* kOp = "hc_mix";

void require(bool condition, const char* reason) {
    if (!condition) throw std::invalid_argument(std::string(kOp) + ": " + reason);
}

std::size_t weight_bytes(const Weight& w) {
    return common::operand::bytes(std::uint64_t(w.n) * w.k, sizeof(std::uint16_t), kOp);
}

void describe_matrix(const Weight& w, int rows, int columns) {
    require(w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous && w.ndim == 2 &&
                w.n == rows && w.k == columns && w.shape[0] == rows && w.shape[1] == columns &&
                w.shape[2] == 1 && w.shape[3] == 1 && w.padded_shape[0] == rows &&
                w.padded_shape[1] == columns && w.padded_shape[2] == 1 && w.padded_shape[3] == 1 &&
                !w.qhigh && !w.scales && w.high_plane_bytes == 0 && w.group_size == 0 &&
                w.group == 0,
            "invalid compact BF16 weight metadata");
    require(w.payload_bytes >= weight_bytes(w), "BF16 weight backing is too small");
}

common::operand::Range tensor(const Tensor& t, DType dtype, int a, int b, int c, Device device) {
    common::operand::device(t.device, device, kOp, "operand");
    require(t.dtype == dtype && t.ne[0] == a && t.ne[1] == b && t.ne[2] == c && t.ne[3] == 1,
            "invalid tensor shape or dtype");
    require(t.is_contiguous() && common::operand::aligned(t.data, 16),
            "tensors require contiguous, 16-byte-aligned backing");
    return common::operand::range(t.data, t.bytes(), kOp);
}

std::array<common::operand::Range, 3> input_ranges(Weights weights, const Tensor& normalized) {
    return {common::operand::range(normalized.data, normalized.bytes(), kOp),
            common::operand::range(weights.down.qdata, weight_bytes(weights.down), kOp),
            common::operand::range(weights.up.qdata, weight_bytes(weights.up), kOp)};
}
} // namespace

void describe_weights(Weights weights) {
    describe_matrix(weights.down, down_rows(weights.kind), kResidualWidth);
    describe_matrix(weights.up, kResidualWidth, kRank);
}

void validate_interval(std::int32_t first, std::int32_t last) {
    require(first > 0 && last >= first, "invalid column interval");
}

Operands validate(Weights weights, const Tensor& normalized, Tensor& output, Tensor* injection,
                  CudaExecutionView execution) {
    describe_weights(weights);
    common::operand::device(weights.down.device, execution.device(), kOp, "down weight");
    common::operand::device(weights.up.device, execution.device(), kOp, "up weight");
    require(common::operand::aligned(weights.down.qdata, 16) &&
                common::operand::aligned(weights.up.qdata, 16),
            "weights require aligned device backing");
    const int m = normalized.ne[2];
    require(m > 0, "columns must be positive");
    (void)tensor(normalized, DType::BF16, kWidth, kStreams, m, execution.device());
    const auto out    = tensor(output, DType::BF16, kWidth, m, 1, execution.device());
    const auto inputs = input_ranges(weights, normalized);
    for (const auto input : inputs) common::operand::disjoint(out, input, kOp);
    if (injection) {
        const auto j = tensor(*injection, DType::FP32, kStreams, m, 1, execution.device());
        for (const auto input : inputs) common::operand::disjoint(j, input, kOp);
        common::operand::disjoint(out, j, kOp);
    }
    return {normalized.data,
            weights.down.qdata,
            weights.up.qdata,
            output.data,
            injection ? static_cast<float*>(injection->data) : nullptr,
            m};
}

void validate_scratch(Weights weights, const Tensor& normalized, const Tensor& output,
                      const Tensor* injection, MemorySpan scratch) {
    common::operand::device(scratch.device, normalized.device, kOp, "scratch");
    const auto work = common::operand::range(scratch.data, scratch.bytes, kOp);
    for (const auto input : input_ranges(weights, normalized))
        common::operand::disjoint(work, input, kOp);
    common::operand::disjoint(work, common::operand::range(output.data, output.bytes(), kOp), kOp);
    if (injection)
        common::operand::disjoint(
            work, common::operand::range(injection->data, injection->bytes(), kOp), kOp);
}
} // namespace ninfer::ops::detail::hc::mix
