#include "ninfer/ops/kda_input_proj.h"
#include "ops/common/host/operands.h"
#include "ops/linear/kernels/q8/q8_operands.h"
#include "ops/linear_attention/kimi_delta_attention/input_conv/launch.h"

#include <limits>

namespace ninfer::ops {
namespace {
constexpr const char* op = "kda_input_proj_conv";
constexpr int channels   = detail::kda_input::kProjectionFeatures;
constexpr int hidden     = detail::kda_input::kInputFeatures;

void validate_profile(const LinearProfile& p) {
    if (p.qtype != QType::Q8_G32_FP16 || p.layout != QuantLayout::RowSplit || p.n != channels ||
        p.k != hidden || p.policy != LinearPolicy::A16Only)
        throw std::invalid_argument("KDA input: unsupported projection profile");
}

std::size_t capacity(const LinearProfile& p, int batch, int first, int last) {
    validate_profile(p);
    if (batch < 1 || batch > 8 || first < 1 || last < first)
        throw std::invalid_argument("KDA input: invalid batch or token interval");
    if (last > std::numeric_limits<int>::max() / batch)
        throw std::overflow_error("KDA input: flattened token extent exceeds int32");
    return last <= 8 && last * batch <= 32
               ? 0
               : Tensor::metadata(DType::FP32, {channels, last, batch}).bytes();
}

void execute(const Tensor& x, const SingleProjectionWeight& qkv, const Tensor& weights,
             ConstStateFieldView source, StateFieldView destination, const Tensor& valid,
             KdaConvOutputs out, Tensor records, MemoryArena& workspace, CudaExecutionView cuda) {
    const int width = x.ne[1], batch = x.ne[2];
    const auto bytes = capacity(linear_profile(qkv), batch, width, width);
    const auto& w    = qkv.weight;
    common::operand::device(w.device, cuda.device(), op, "weight");
    common::operand::device(workspace.device(), cuda.device(), op, "workspace");
    if (w.ndim != 2 || w.group_size != 32 || w.group != 32 || w.scale_dtype != DType::FP16 ||
        w.shape[0] != channels || w.shape[1] != hidden || w.padded_shape[0] != channels ||
        w.padded_shape[1] != hidden || w.qhigh || w.high_plane_bytes)
        throw std::invalid_argument("KDA input: invalid RowSplit weights");
    common::operand::tensor(x, DType::BF16, {hidden, width, batch}, 16, op, "input", cuda.device());
    detail::validate_q8_operands(detail::q8_linear_operands(x.view({hidden, width * batch}), w));
    common::operand::tensor(weights, DType::FP32, {channels, 4}, 16, op, "convolution weights",
                            cuda.device());
    common::operand::state_field(source, DType::BF16, {channels, 3}, 16, op, "source",
                                 cuda.device());
    if (source.field.layers != 1)
        throw std::invalid_argument("KDA input: select one convolution layer");
    common::operand::shape(source.records.byte_offsets, {batch, 1, 1, 1}, op, "source offsets");
    if (valid.data)
        common::operand::tensor(valid, DType::I32, {batch}, 4, op, "valid columns", cuda.device());
    for (const auto* t : {&out.query, &out.key, &out.value})
        common::operand::tensor(*t, DType::BF16, {hidden, width, batch}, 16, op, "output",
                                cuda.device());
    if (records.data) {
        common::operand::tensor(records, DType::BF16, {channels, width, batch}, 16, op,
                                "conv records", cuda.device());
    } else {
        common::operand::state_field(destination, DType::BF16, {channels, 3}, 16, op, "destination",
                                     cuda.device());
        if (destination.field.layers != 1)
            throw std::invalid_argument("KDA input: select one destination layer");
        common::operand::shape(destination.records.byte_offsets, {batch, 1, 1, 1}, op,
                               "destination offsets");
    }
    const Tensor* outputs[] = {&out.query, &out.key, &out.value, &records};
    const Tensor* inputs[]  = {&x, &weights, &source.records.byte_offsets,
                               &destination.records.byte_offsets, &valid};
    for (std::size_t i = 0; i < std::size(outputs); ++i) {
        if (!outputs[i]->data) continue;
        const auto range = common::operand::range(*outputs[i], op);
        for (std::size_t j = 0; j < i; ++j)
            if (outputs[j]->data)
                common::operand::disjoint(range, common::operand::range(*outputs[j], op), op);
        for (const auto* input : inputs)
            if (input->data)
                common::operand::disjoint(range, common::operand::range(*input, op), op);
        common::operand::disjoint(
            range, common::operand::range(w.qdata, std::size_t(channels) * hidden, op), op);
        common::operand::disjoint(
            range, common::operand::range(w.scales, std::size_t(channels) * hidden / 16, op), op);
    }
    auto scope       = workspace.scope();
    float* projected = bytes ? static_cast<float*>(workspace.alloc_bytes(bytes).data) : nullptr;
    if (projected) {
        const auto scratch = common::operand::range(projected, bytes, op);
        for (const auto* tensor : inputs)
            if (tensor->data)
                common::operand::disjoint(scratch, common::operand::range(*tensor, op), op);
        for (const auto* tensor : outputs)
            if (tensor->data)
                common::operand::disjoint(scratch, common::operand::range(*tensor, op), op);
        common::operand::disjoint(
            scratch, common::operand::range(w.qdata, std::size_t(channels) * hidden, op), op);
        common::operand::disjoint(
            scratch, common::operand::range(w.scales, std::size_t(channels) * hidden / 16, op), op);
    }
    const detail::kda_input::Arguments a{
        static_cast<const float*>(weights.data),
        common::operand::state_field_base<const __nv_bfloat16>(source),
        records.data ? nullptr : common::operand::state_field_base<__nv_bfloat16>(destination),
        static_cast<const std::int64_t*>(source.records.byte_offsets.data),
        static_cast<const std::int64_t*>(destination.records.byte_offsets.data),
        static_cast<const std::int32_t*>(valid.data),
        static_cast<__nv_bfloat16*>(out.query.data),
        static_cast<__nv_bfloat16*>(out.key.data),
        static_cast<__nv_bfloat16*>(out.value.data),
        static_cast<__nv_bfloat16*>(records.data),
        width,
        batch};
    detail::kda_input::launch(x, w, a, projected, cuda.stream());
}
} // namespace

std::size_t kda_input_proj_conv_update_workspace_capacity_bytes(const LinearProfile& p, int b,
                                                                int first, int last) {
    return capacity(p, b, first, last);
}

std::size_t kda_input_proj_conv_record_workspace_capacity_bytes(const LinearProfile& p, int b,
                                                                int first, int last) {
    return capacity(p, b, first, last);
}

void kda_input_proj_conv_update(const Tensor& x, const SingleProjectionWeight& qkv, const Tensor& w,
                                KdaConvUpdateState s, const Tensor& valid, KdaConvOutputs out,
                                MemoryArena& workspace, CudaExecutionView cuda) {
    execute(x, qkv, w, s.source, s.destination, valid, out, {}, workspace, cuda);
}

void kda_input_proj_conv_record(const Tensor& x, const SingleProjectionWeight& qkv, const Tensor& w,
                                KdaConvSourceState s, const Tensor& valid, KdaConvOutputs out,
                                Tensor& records, MemoryArena& workspace, CudaExecutionView cuda) {
    if (!records.data) throw std::invalid_argument("KDA input: missing convolution records");
    execute(x, qkv, w, s.history, {}, valid, out, records, workspace, cuda);
}
} // namespace ninfer::ops
