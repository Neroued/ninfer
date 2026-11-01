#include "core/platform.h"
#include "ops/linear_attention/gated_delta_net/input_conv/operands.h"
#include "ops/common/host/operands.h"

#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::gdn_input {
namespace {

void require_tensor(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                    std::uintptr_t alignment, const char* label) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != 1 ||
        !t.is_contiguous() || !common::operand::aligned(t.data, alignment))
        throw std::invalid_argument(std::string("GDN input: invalid ") + label);
}

void require_selector(const Tensor& t, std::int32_t batch, const char* label) {
    require_tensor(t, DType::I32, batch, 1, 1, alignof(std::int32_t), label);
}

void validate_rowsplit(const Weight& w) {
    const bool q8    = w.qtype == QType::Q8_G32_FP16;
    const bool q5    = w.qtype == QType::Q5_G64_FP16;
    const auto group = q8 ? 32U : 64U;
    if (w.layout != QuantLayout::RowSplit || w.scale_dtype != DType::FP16 ||
        w.group_size != group || w.group != static_cast<std::int32_t>(group) || w.ndim != 2 ||
        w.shape[0] != w.n || w.shape[1] != w.k || w.padded_shape[0] != w.n ||
        w.padded_shape[1] != w.k || !common::operand::aligned(w.qdata, 16) ||
        !common::operand::aligned(w.scales, q8 ? 16 : 4) ||
        (q5 ? (!common::operand::aligned(w.qhigh, 16) || w.high_plane_bytes == 0)
            : (w.qhigh != nullptr || w.high_plane_bytes != 0)))
        throw std::invalid_argument("GDN input: invalid RowSplit weight representation");
}

} // namespace

WeightProfile describe_profile(const ProjectionProfile& weights) {
    if (weights.second) {
        const auto& a = weights.first;
        const auto& b = *weights.second;
        if (a.qtype != QType::Q4_G64_FP16 || b.qtype != QType::Q5_G64_FP16 || a.n != 4096 ||
            a.k != 5120 || b.n != 12288 || b.k != 5120)
            throw std::invalid_argument("GDN input: unsupported paired weight geometry");
        return {Format::Q4Q5, {5120, 6144}, LinearPolicy::A16Only};
    }
    const auto& w = weights.first;
    if (!valid_linear_policy(w.policy))
        throw std::invalid_argument("GDN input: invalid activation permission");
    if (w.qtype == QType::Q8_G32_FP16) {
        if (w.n == 12288 && w.k == 2048) return {Format::Q8, {2048, 4096}, w.policy};
        if (w.n == 16384 && w.k == 2560) return {Format::Q8, {2560, 6144}, w.policy};
#if NINFER_TARGET_SPARK
        if (w.n == 8192 && w.k == 2560) return {Format::Q8, {2560, 3072, 1024, 1024}, w.policy};
#endif
    }
    if (w.n == 16384 && w.k == 5120) {
        if (w.qtype == QType::FP8_E4M3FN_ROW_BF16) return {Format::Fp8, {5120, 6144}, w.policy};
        if (w.qtype == QType::NVFP4) return {Format::Nvfp4, {5120, 6144}, w.policy};
    }
    throw std::invalid_argument("GDN input: unsupported single-parent weight geometry");
}

WeightOperands describe_weights(const ProjectionWeights& weights) {
    const auto profile = describe_profile(projection_profile(weights));
    if (const auto* single = std::get_if<SingleProjectionWeight>(&weights)) {
        return {profile, &single->weight, nullptr};
    }
    const auto& pair = std::get<PairedProjectionWeights>(weights);
    return {profile, &pair.first, &pair.second};
}

void validate_weights(const WeightOperands& weights) {
    switch (weights.format) {
    case Format::Q4Q5:
        validate_rowsplit(*weights.second);
        [[fallthrough]];
    case Format::Q8:
        validate_rowsplit(*weights.first);
        break;
    case Format::Fp8:
        (void)common::validate_fp8_weight(*weights.first, "GDN input");
        break;
    case Format::Nvfp4:
        (void)common::validate_nvfp4_weight(*weights.first, "GDN input");
        break;
    }
}

void validate_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first)
        throw std::invalid_argument("GDN input: invalid positive column interval");
}

void validate_batch_interval(Publication publication, std::int32_t batch, std::int32_t first,
                             std::int32_t last) {
    validate_interval(first, last);
    if (batch < 1 || batch > 8 || (batch > 1 && last > 16) ||
        (publication == Publication::Record && (first < 2 || last > 16)))
        throw std::invalid_argument("GDN convolution: unsupported batch/token interval");
}

Batch validate_convolution(const Tensor& x, const WeightOperands& weights,
                           const Tensor& convolution, const ConstStateFieldView& history,
                           const Tensor& valid_columns, const GdnConvOutputs& outputs,
                           Publication publication, const Tensor& destination) {
    const Batch batch{x.ne[1], x.ne[2]};
    validate_batch_interval(publication, batch.rows, batch.tokens, batch.tokens);
    const auto& g = weights.geometry;
    require_tensor(x, DType::BF16, g.input_rows, batch.tokens, batch.rows, 16, "x");
    require_tensor(convolution, DType::BF16, g.channels(), 4, 1, 16, "convolution weight");
    common::operand::state_field(history, DType::BF16, {g.channels(), 3}, 16, "GDN convolution",
                                 "history", x.device);
    if (history.field.layers != 1 || history.records.byte_offsets.ne[0] != batch.rows)
        throw std::invalid_argument("GDN convolution: source row count differs from batch");
    if (valid_columns.data) require_selector(valid_columns, batch.rows, "valid columns");
    require_tensor(outputs.query, DType::BF16, g.query_rows, batch.tokens, batch.rows, 16, "query");
    require_tensor(outputs.key, DType::BF16, g.key_rows, batch.tokens, batch.rows, 16, "key");
    require_tensor(outputs.value, DType::BF16, g.value_rows, batch.tokens, batch.rows, 16, "value");
    require_tensor(outputs.z, DType::BF16, g.value_rows, batch.tokens, batch.rows, 16, "z");
    if (publication == Publication::Snapshot)
        require_selector(destination, batch.rows, "snapshot bases");
    else if (publication == Publication::Record)
        require_tensor(destination, DType::BF16, g.channels(), batch.tokens, batch.rows, 16,
                       "convolution record");
    return batch;
}

void validate_storage(const WeightOperands& weights, std::span<const Tensor* const> tensors,
                      CudaExecutionView execution, MemorySpan scratch, const Tensor* alias_first,
                      const Tensor* alias_second) {
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const Tensor& t = *tensors[i];
        if (!t.data) continue;
        common::operand::device(t.device, execution.device(), "GDN input", "operand");
        for (std::size_t j = i + 1; j < tensors.size(); ++j) {
            if ((tensors[i] == alias_first && tensors[j] == alias_second) ||
                (tensors[j] == alias_first && tensors[i] == alias_second))
                continue;
            if (tensors[j]->data &&
                common::operand::overlaps(common::operand::blocks(t, "GDN input"),
                                          common::operand::blocks(*tensors[j], "GDN input")))
                throw std::invalid_argument("GDN input: tensor operands overlap");
        }
        if (common::operand::overlaps(
                common::operand::blocks(t, "GDN input"),
                common::operand::range(scratch.data, scratch.bytes, "GDN input")))
            throw std::invalid_argument("GDN input: tensor overlaps call-local scratch");
        for (const Weight* w : {weights.first, weights.second}) {
            if (w && common::operand::overlaps(
                         common::operand::blocks(t, "GDN input"),
                         common::operand::range(w->payload, w->payload_bytes, "GDN input")))
                throw std::invalid_argument("GDN input: tensor overlaps weight storage");
        }
    }
    if (scratch.bytes) {
        common::operand::device(scratch.device, execution.device(), "GDN input", "scratch");
    }
    for (const Weight* w : {weights.first, weights.second}) {
        if (w) common::operand::device(w->device, execution.device(), "GDN input", "weight");
        if (w && common::operand::overlaps(
                     common::operand::range(scratch.data, scratch.bytes, "GDN input"),
                     common::operand::range(w->payload, w->payload_bytes, "GDN input")))
            throw std::invalid_argument("GDN input: scratch overlaps weight storage");
    }
}

} // namespace ninfer::ops::detail::gdn_input
