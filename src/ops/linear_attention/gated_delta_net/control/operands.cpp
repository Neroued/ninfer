#include "ops/linear_attention/gated_delta_net/control/operands.h"
#include "ops/common/host/operands.h"
#include "core/platform.h"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::gdn_control {
namespace {

constexpr const char* kOp = "gdn_control";

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(std::string(kOp) + ": " + message);
}

void describe_matrix(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    const auto bytes = common::operand::bytes(std::uint64_t(rows) * columns, 2, kOp);
    require(weight.qtype == QType::BF16 && weight.layout == QuantLayout::Contiguous &&
                weight.ndim == 2 && weight.n == rows && weight.k == columns &&
                weight.shape[0] == std::uint64_t(rows) &&
                weight.shape[1] == std::uint64_t(columns) &&
                weight.padded_shape[0] == std::uint64_t(rows) &&
                weight.padded_shape[1] == std::uint64_t(columns) && weight.payload_bytes >= bytes &&
                !weight.qhigh && !weight.scales && weight.high_plane_bytes == 0 &&
                weight.group == 0 && weight.group_size == 0,
            "invalid BF16 weight metadata");
}

void matrix(const Tensor& tensor, DType type, int rows, int columns, bool allow_empty = false,
            std::size_t bf16_alignment = 16) {
    require((columns > 0 || (allow_empty && columns == 0)) && tensor.dtype == type &&
                tensor.ne[0] == rows && tensor.ne[1] == columns && tensor.ne[2] == 1 &&
                tensor.ne[3] == 1,
            "invalid matrix shape or dtype");
    if (columns == 0) return;
    require(tensor.is_contiguous() &&
                common::operand::aligned(tensor.data, type == DType::BF16 ? bf16_alignment : 4),
            "matrix must have aligned contiguous backing");
}

void vector(const Tensor& tensor, DType type, int rows, bool backing = true) {
    require(tensor.dtype == type && tensor.ne[0] == rows && tensor.ne[1] == 1 &&
                tensor.ne[2] == 1 && tensor.ne[3] == 1,
            "invalid vector shape or dtype");
    if (backing)
        require(tensor.is_contiguous() &&
                    common::operand::aligned(tensor.data, type == DType::BF16 ? 16 : 4),
                "vector must have aligned contiguous backing");
}

} // namespace

Geometry describe_profile(const ProjectionProfile& weights) {
    const auto& a = weights.first;
    require(valid_linear_policy(a.policy), "invalid activation permission");
    if (!weights.second) {
        bool supported = (a.n == 96 && (a.k == 5120 || a.k == 2560)) || (a.n == 64 && a.k == 2048);
#if NINFER_TARGET_SPARK
        supported = supported || (a.n == 48 && a.k == 2560);
#endif
        require(a.qtype == QType::BF16 && supported, "unsupported parent geometry");
        return {a.n / 2, a.k};
    }
    const auto& b = *weights.second;
    require(a.qtype == QType::BF16 && b.qtype == QType::BF16 && a.n == 48 && a.k == 5120 &&
                b.n == 48 && b.k == 5120,
            "unsupported paired geometry");
    return {48, 5120};
}

Weights describe_weights(const ProjectionWeights& weights) {
    const auto geometry = describe_profile(projection_profile(weights));
    if (const auto* single = std::get_if<SingleProjectionWeight>(&weights)) {
        const auto& w = single->weight;
        describe_matrix(w, w.n, w.k);
        return {geometry, &w, nullptr};
    }
    const auto& pair = std::get<PairedProjectionWeights>(weights);
    describe_matrix(pair.first, 48, 5120);
    describe_matrix(pair.second, 48, 5120);
    return {geometry, &pair.first, &pair.second};
}

void validate_weights(const Weights& weights) {
    require(common::operand::aligned(weights.first->qdata, 16), "invalid first weight backing");
    if (weights.second)
        require(common::operand::aligned(weights.second->qdata, 16),
                "invalid second weight backing");
}

void validate_interval(std::int32_t first, std::int32_t last) {
    require(first > 0 && last >= first, "invalid extent interval");
}

void validate_projection(const Tensor& x, const Weights& weights, const Tensor& a_log,
                         const Tensor& dt_bias, GdnControlOutputs out) {
    const auto [heads, rows] = weights.geometry;
    matrix(x, DType::BF16, rows, x.ne[1]);
    vector(a_log, DType::FP32, heads);
    vector(dt_bias, DType::FP32, heads);
    matrix(out.g, DType::FP32, heads, x.ne[1]);
    matrix(out.beta, DType::FP32, heads, x.ne[1]);
}

void validate_norm(const Tensor& x, const Tensor& weight, float eps, Geometry geometry,
                   const Tensor& normalized) {
    require(geometry.input_rows != 2560, "norm projection does not admit this geometry");
    require(eps > 0 && std::isfinite(eps), "epsilon must be positive and finite");
    vector(weight, DType::BF16, geometry.input_rows);
    matrix(normalized, DType::BF16, geometry.input_rows, x.ne[1]);
}

void validate_control(const Tensor& a, const Tensor& b, const Tensor& a_log, const Tensor& dt_bias,
                      GdnControlOutputs out) {
    matrix(a, DType::BF16, 48, a.ne[1], true, 2);
    matrix(b, DType::BF16, 48, a.ne[1], true, 2);
    matrix(out.g, DType::FP32, 48, a.ne[1], true, 2);
    matrix(out.beta, DType::FP32, 48, a.ne[1], true, 2);
    vector(a_log, DType::FP32, 48, a.ne[1] != 0);
    vector(dt_bias, DType::FP32, 48, a.ne[1] != 0);
}

void validate_storage(const Weights* weights, std::span<const Tensor* const> tensors,
                      CudaExecutionView execution, MemorySpan scratch) {
    std::array<common::operand::Range, 11> ranges{};
    std::size_t count = 0;
    const auto add    = [&](const void* pointer, std::size_t bytes) {
        ranges[count++] = common::operand::range(pointer, bytes, kOp);
    };
    if (weights) {
        const auto& first = *weights->first;
        common::operand::device(first.device, execution.device(), kOp, "weight");
        add(first.qdata, common::operand::bytes(std::uint64_t(first.n) * first.k, 2, kOp));
        if (weights->second) {
            const auto& second = *weights->second;
            common::operand::device(second.device, execution.device(), kOp, "weight");
            add(second.qdata, common::operand::bytes(std::uint64_t(second.n) * second.k, 2, kOp));
        }
    }
    for (const Tensor* tensor : tensors) {
        common::operand::device(tensor->device, execution.device(), kOp, "operand");
        add(tensor->data, common::operand::bytes(tensor->numel(), dtype_size(tensor->dtype), kOp));
    }
    if (scratch.bytes) {
        common::operand::device(scratch.device, execution.device(), kOp, "scratch");
        add(scratch.data, scratch.bytes);
    }
    common::operand::disjoint(std::span{ranges.data(), count}, kOp);
}

} // namespace ninfer::ops::detail::gdn_control
