#include "ops/ple/operands.h"
#include "ops/common/host/operands.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::ple {
namespace {

void require(bool condition, const char* label) {
    if (!condition) throw std::invalid_argument(std::string("PLE: invalid ") + label);
}

void contiguous(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                std::size_t alignment, const char* label, Device device) {
    common::operand::device(t.device, device, "PLE", label);
    const auto expected = Tensor::metadata(dtype, shape);
    require(t.dtype == dtype && t.is_contiguous() && common::operand::aligned(t.data, alignment),
            label);
    for (int d = 0; d < 4; ++d) require(t.ne[d] == expected.ne[d], label);
}

using common::operand::Range;

// A column is contiguous. The two outer axes can be padded or interleaved. One has
// at most nine entries (history columns or batch rows), so validation never scans R/S.
struct Columns {
    Range bounds;
    std::uint64_t width, small_count, small_stride, large_count, large_stride;
};

Columns columns(const Tensor& t, int count, int outer, const char* label, Device device,
                int features = kChannels) {
    common::operand::device(t.device, device, "PLE", label);
    require(count > 0 && outer > 0 && t.dtype == DType::BF16 && t.ne[0] == features &&
                t.ne[1] == count && t.ne[2] == outer && t.ne[3] == 1 && t.nb[0] == 2 &&
                common::operand::aligned(t.data, 2),
            label);
    const std::uint64_t width = features * 2;
    std::uint64_t bytes       = width;
    for (int d = 1; d <= 2; ++d) {
        if (t.ne[d] == 1) continue;
        require(t.nb[d] >= static_cast<std::int64_t>(width) && t.nb[d] % 2 == 0, label);
        const auto step = static_cast<std::uint64_t>(t.nb[d]);
        const auto n    = static_cast<std::uint64_t>(t.ne[d] - 1);
        require(step <= (std::numeric_limits<std::int64_t>::max() - bytes) / n, label);
        bytes += step * n;
    }
    const int small = count <= outer ? 1 : 2, large = 3 - small;
    Columns result{common::operand::range(t.data, bytes, "PLE"),
                   width,
                   static_cast<std::uint64_t>(t.ne[small]),
                   t.ne[small] == 1 ? width : static_cast<std::uint64_t>(t.nb[small]),
                   static_cast<std::uint64_t>(t.ne[large]),
                   t.ne[large] == 1 ? width : static_cast<std::uint64_t>(t.nb[large])};
    // Columns from different small-axis positions must not overlap any column of
    // the other progression. Only the two nearest large-axis offsets can overlap.
    for (std::uint64_t delta = 1; delta < result.small_count; ++delta) {
        const auto distance = delta * result.small_stride;
        const auto q        = distance / result.large_stride;
        if (q < result.large_count) require(distance - q * result.large_stride >= width, label);
        if (q + 1 < result.large_count)
            require((q + 1) * result.large_stride - distance >= width, label);
    }
    return result;
}

bool overlaps_columns(Range a, const Columns& b) {
    if (!common::operand::overlaps(a, b.bounds)) return false;
    const auto width = b.width;
    for (std::uint64_t i = 0; i < b.small_count; ++i) {
        const auto first = b.bounds.begin + i * b.small_stride;
        if (a.end <= first) continue;
        const auto j = a.begin < first + width ? 0 : (a.begin - first - width) / b.large_stride + 1;
        if (j < b.large_count && first + j * b.large_stride < a.end) return true;
    }
    return false;
}

void metadata(const Tensor& t, int batch, const char* label, Device device) {
    contiguous(t, DType::I32, {batch}, 4, label, device);
}

void validate_batch_extent(int batch) { require(batch >= 1 && batch <= 8, "batch extent"); }

} // namespace

void validate_gate(const PleProjectionView& projected, const Tensor& residual,
                   const PleGateParameters& parameters, const PleBatchView& batch,
                   const PleConvRecordWindowView& records, const Tensor& gated,
                   CudaExecutionView execution) {
    const int tokens = residual.ne[2], rows = residual.ne[3];
    validate_batch_extent(rows);
    require(tokens > 0, "token extent");
    for (const Tensor* t : {&residual, &gated})
        contiguous(*t, DType::BF16, {kFeatures, kStreams, tokens, rows}, 16, "activation",
                   execution.device());
    for (const Tensor* t : {&parameters.norm_key, &parameters.norm_query, &parameters.norm_conv})
        contiguous(*t, DType::BF16, {kFeatures, kStreams}, 2, "norm weight", execution.device());
    require(std::isfinite(parameters.rms_epsilon) && parameters.rms_epsilon > 0, "norm epsilon");
    metadata(batch.valid_columns, rows, "valid columns", execution.device());
    metadata(records.offsets, rows, "record offsets", execution.device());

    auto key = projected.key;
    require(key.ne[0] == kFeatures && key.ne[1] == kStreams && key.ne[2] == tokens &&
                key.ne[3] == rows && key.nb[1] == kFeatures * 2,
            "key shape/stream stride");
    key.ne[0]       = kChannels;
    key.ne[1]       = tokens;
    key.ne[2]       = rows;
    key.ne[3]       = 1;
    key.nb[1]       = projected.key.nb[2];
    key.nb[2]       = projected.key.nb[3];
    const auto keys = columns(key, tokens, rows, "key", execution.device());
    const auto values =
        columns(projected.value, tokens, rows, "value", execution.device(), kFeatures);
    const auto record =
        columns(records.values, records.values.ne[1], rows, "records", execution.device());
    const auto out = common::operand::range(gated.data, gated.bytes(), "PLE");
    for (const Tensor* input : {&residual, &parameters.norm_key, &parameters.norm_query,
                                &parameters.norm_conv, &batch.valid_columns, &records.offsets})
        require(!common::operand::overlaps(out, common::operand::range(*input, "PLE")),
                "gate output/input overlap");
    require(!overlaps_columns(out, keys) && !overlaps_columns(out, values) &&
                !overlaps_columns(out, record),
            "gate output/projection or record overlap");
    // Record writes depend on Device offsets/counts. Their disjointness from inputs is a
    // caller promise; unrelated columns and gaps may share an allocation with another view.
}

void validate_conv(const Tensor& residual, const Tensor& gated, const Tensor& kernel,
                   const PleBatchView& batch, const ConstPleConvStateView& before,
                   const PleConvRecordWindowView& current, const Tensor& updated,
                   const Tensor* norm_weight, float epsilon, const Tensor* normalized,
                   CudaExecutionView execution) {
    const int tokens = residual.ne[2], rows = residual.ne[3];
    validate_batch_extent(rows);
    require(tokens > 0, "token extent");
    for (const Tensor* t : {&residual, &gated, &updated})
        contiguous(*t, DType::BF16, {kFeatures, kStreams, tokens, rows}, 16, "activation",
                   execution.device());
    if (updated.data == residual.data)
        for (int d = 0; d < 4; ++d)
            require(updated.nb[d] == residual.nb[d], "in-place residual strides");
    contiguous(kernel, DType::BF16, {kChannels, 4}, 2, "convolution kernel", execution.device());
    metadata(batch.valid_columns, rows, "valid columns", execution.device());
    require(before.history.field.layers == 1, "history layer");
    common::operand::state_field(before.history, DType::BF16, {kChannels, kHistory}, 2, "PLE",
                                 "history", execution.device());
    common::operand::shape(before.history.records.byte_offsets, {rows, 1, 1, 1}, "PLE",
                           "source byte offsets");
    metadata(current.offsets, rows, "record offsets", execution.device());
    const auto records =
        columns(current.values, current.values.ne[1], rows, "records", execution.device());
    if (normalized) {
        contiguous(*normalized, DType::BF16, {kFeatures, kStreams, tokens, rows}, 16, "normalized",
                   execution.device());
        contiguous(*norm_weight, DType::BF16, {kFeatures, kStreams}, 2, "norm weight",
                   execution.device());
        require(std::isfinite(epsilon) && epsilon > 0, "norm epsilon");
    }
    for (const Tensor* output : {&updated, normalized}) {
        if (!output) continue;
        const auto out = common::operand::range(output->data, output->bytes(), "PLE");
        for (const Tensor* input :
             {&residual, &gated, &kernel, &batch.valid_columns,
              &before.history.records.byte_offsets, &current.offsets, norm_weight}) {
            if (!input || (output == &updated && input == &residual && output->data == input->data))
                continue;
            require(!common::operand::overlaps(
                        out, common::operand::range(input->data, input->bytes(), "PLE")),
                    "output/input overlap");
        }
        require(!overlaps_columns(out, records), "output/record overlap");
        // State fields are selected by Device offsets. Addressed input/output disjointness
        // is a caller promise; unrelated records may share the same arena.
    }
    if (normalized)
        require(!common::operand::overlaps(
                    common::operand::range(updated.data, updated.bytes(), "PLE"),
                    common::operand::range(normalized->data, normalized->bytes(), "PLE")),
                "output overlap");
}

void validate_extract(const ConstPleConvStateView& before, const PleConvRecordReadView& records,
                      const Tensor& accepted, const PleConvStateView& after,
                      CudaExecutionView execution) {
    const int batch = accepted.ne[0];
    validate_batch_extent(batch);
    metadata(accepted, batch, "accepted columns", execution.device());
    metadata(records.valid_columns, batch, "record valid columns", execution.device());
    const auto validate_history = [&](const ConstStateFieldView& state) {
        require(state.field.layers == 1, "history layer");
        common::operand::state_field(state, DType::BF16, {kChannels, kHistory}, 2, "PLE", "history",
                                     execution.device());
        common::operand::shape(state.records.byte_offsets, {batch, 1, 1, 1}, "PLE",
                               "state byte offsets");
    };
    validate_history(before.history);
    validate_history(after.history);
    (void)columns(records.values, records.values.ne[1], batch, "records", execution.device());
    // Only selected records are read/written. Their cross-row and in-place relationships
    // are Device-index promises; overlapping pool descriptors alone are not an error.
}

} // namespace ninfer::ops::detail::ple
