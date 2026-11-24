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
    const bool q8 = p.qtype == QType::Q8_G32_FP16 && p.layout == QuantLayout::RowSplit &&
                    p.policy == LinearPolicy::A16Only;
    const bool mx = p.qtype == QType::MXFP8 && p.layout == QuantLayout::RowGroupScale &&
                    (p.policy == LinearPolicy::A16Only || p.policy == LinearPolicy::AllowA8);
    if ((!q8 && !mx) || p.n != channels || p.k != hidden)
        throw std::invalid_argument("KDA input: unsupported projection profile");
}

std::size_t capacity(const LinearProfile& p, int batch, int first, int last, bool update) {
    validate_profile(p);
    if (batch < 1 || first < 1 || last < first)
        throw std::invalid_argument("KDA input: invalid batch or token interval");
    if (p.qtype != QType::MXFP8 && last > std::numeric_limits<int>::max() / batch)
        throw std::overflow_error("KDA input: flattened token extent exceeds int32");
    (void)Tensor::metadata(DType::BF16, {hidden, last, batch}).bytes();
    return p.qtype == QType::MXFP8
               ? detail::kda_input::mxfp8_workspace_capacity(batch, first, last, update, p.policy)
               : detail::kda_input::workspace_capacity(batch, first, last, update);
}

void execute(const Tensor& x, const SingleProjectionWeight& qkv, const Tensor& weights,
             ConstStateFieldView source, StateFieldView destination, const Tensor& valid,
             KdaConvOutputs out, Tensor records, MemoryArena& workspace, CudaExecutionView cuda) {
    const int width = x.ne[1], batch = x.ne[2];
    const auto bytes = capacity(linear_profile(qkv), batch, width, width, !records.data);
    const auto& w    = qkv.weight;
    const bool mx    = w.qtype == QType::MXFP8;
    common::operand::device(w.device, cuda.device(), op, "weight");
    common::operand::device(workspace.device(), cuda.device(), op, "workspace");
    if (!mx &&
        (w.ndim != 2 || w.group_size != 32 || w.group != 32 || w.scale_dtype != DType::FP16 ||
         w.shape[0] != channels || w.shape[1] != hidden || w.padded_shape[0] != channels ||
         w.padded_shape[1] != hidden || w.qhigh || w.high_plane_bytes))
        throw std::invalid_argument("KDA input: invalid projection weight geometry");
    common::operand::tensor(x, DType::BF16, {hidden, width, batch}, 16, op, "input", cuda.device());
    if (mx) {
        (void)common::validate_mxfp8_weight(w, op);
    } else {
        detail::validate_q8_operands(
            detail::q8_linear_operands(x.view({hidden, width * batch}), w));
    }
    const auto code_bytes  = std::size_t(channels) * hidden;
    const auto scale_bytes = code_bytes / (mx ? 32 : 16);
    common::operand::tensor(weights, mx ? DType::BF16 : DType::FP32, {channels, 4}, 16, op,
                            "convolution weights", cuda.device());
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
        common::operand::disjoint(range, common::operand::range(w.qdata, code_bytes, op), op);
        common::operand::disjoint(range, common::operand::range(w.scales, scale_bytes, op), op);
    }
    auto scope         = workspace.scope();
    void* scratch_data = bytes ? workspace.alloc_bytes(bytes).data : nullptr;
    if (scratch_data) {
        const auto scratch = common::operand::range(scratch_data, bytes, op);
        for (const auto* tensor : inputs)
            if (tensor->data)
                common::operand::disjoint(scratch, common::operand::range(*tensor, op), op);
        for (const auto* tensor : outputs)
            if (tensor->data)
                common::operand::disjoint(scratch, common::operand::range(*tensor, op), op);
        common::operand::disjoint(scratch, common::operand::range(w.qdata, code_bytes, op), op);
        common::operand::disjoint(scratch, common::operand::range(w.scales, scale_bytes, op), op);
    }
    const detail::kda_input::Arguments a{
        weights.data,
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
    if (mx)
        detail::kda_input::launch_mxfp8(x, w, qkv.policy, a, scratch_data, cuda.stream());
    else
        detail::kda_input::launch(x, w, a, static_cast<float*>(scratch_data), cuda.stream());
}
} // namespace

std::size_t kda_input_proj_conv_update_workspace_capacity_bytes(const LinearProfile& p, int b,
                                                                int first, int last) {
    return capacity(p, b, first, last, true);
}

std::size_t kda_input_proj_conv_update_ragged_workspace_capacity_bytes(
    const LinearProfile& p, int min_tokens, int max_tokens, int min_segments, int max_segments) {
    const auto widths =
        common::operand::ragged_envelope(min_tokens, max_tokens, min_segments, max_segments, op);
    return capacity(p, 1, widths.first, widths.last, true);
}

std::size_t kda_input_proj_conv_record_workspace_capacity_bytes(const LinearProfile& p, int b,
                                                                int first, int last) {
    return capacity(p, b, first, last, false);
}

void kda_input_proj_conv_update(const Tensor& x, const SingleProjectionWeight& qkv, const Tensor& w,
                                KdaConvUpdateState s, const Tensor& valid, KdaConvOutputs out,
                                MemoryArena& workspace, CudaExecutionView cuda) {
    execute(x, qkv, w, s.source, s.destination, valid, out, {}, workspace, cuda);
}

void kda_input_proj_conv_update(const Tensor& x, const SingleProjectionWeight& qkv, const Tensor& w,
                                KdaConvUpdateState s, const RaggedSequenceView& ragged,
                                KdaConvOutputs out, MemoryArena& workspace,
                                CudaExecutionView cuda) {
    common::operand::ragged_sequence(ragged, op, cuda.device());
    const int num_seqs = ragged.num_seqs(), total_tokens = ragged.total_tokens();
    common::operand::tensor(x, DType::BF16, {hidden, total_tokens}, 16, op, "input", cuda.device());
    for (const auto* t : {&out.query, &out.key, &out.value})
        common::operand::tensor(*t, DType::BF16, {hidden, total_tokens}, 16, op, "output",
                                cuda.device());
    common::operand::state_field(s.source, DType::BF16, {channels, 3}, 16, op, "source",
                                 cuda.device());
    common::operand::state_field(s.destination, DType::BF16, {channels, 3}, 16, op, "destination",
                                 cuda.device());
    if (s.source.field.layers != 1 || s.destination.field.layers != 1)
        throw std::invalid_argument("KDA input: select one convolution layer");
    common::operand::shape(s.source.records.byte_offsets, {num_seqs, 1, 1, 1}, op,
                           "source offsets");
    common::operand::shape(s.destination.records.byte_offsets, {num_seqs, 1, 1, 1}, op,
                           "destination offsets");
    const std::array outputs{&out.query, &out.key, &out.value};
    const Tensor* metadata[] = {&ragged.cu_seqlens, &ragged.seq_lengths, &ragged.token_to_seq,
                                &s.source.records.byte_offsets,
                                &s.destination.records.byte_offsets};
    std::size_t peak         = 0;
    for (int seq = 0; seq < num_seqs; ++seq) {
        const int len = ragged.segment_length(seq);
        peak          = std::max(peak, capacity(linear_profile(qkv), 1, len, len, true));
    }
    common::operand::device(workspace.device(), cuda.device(), op, "workspace");
    if (peak) {
        auto scope         = workspace.scope();
        const auto scratch = common::operand::range(workspace.alloc_bytes(peak).data, peak, op);
        for (const auto* t : metadata)
            common::operand::disjoint(scratch, common::operand::range(*t, op), op);
        for (const auto* t : std::array<const Tensor*, 4>{&x, &out.query, &out.key, &out.value})
            common::operand::disjoint(scratch, common::operand::range(*t, op), op);
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto range = common::operand::range(*outputs[i], op);
        for (std::size_t j = 0; j < i; ++j)
            common::operand::disjoint(range, common::operand::range(*outputs[j], op), op);
        common::operand::disjoint(range, common::operand::range(x, op), op);
        for (const auto* m : metadata)
            common::operand::disjoint(range, common::operand::range(*m, op), op);
    }
    for (int seq = 0; seq < num_seqs; ++seq) {
        const int begin = ragged.segment_begin(seq), len = ragged.segment_length(seq);
        const auto src = common::operand::state_row(s.source, seq);
        const auto dst = common::operand::state_row(s.destination, seq);
        execute(x.slice(1, begin, len).view({hidden, len, 1}), qkv, w, src, dst, {},
                {out.query.slice(1, begin, len).view({hidden, len, 1}),
                 out.key.slice(1, begin, len).view({hidden, len, 1}),
                 out.value.slice(1, begin, len).view({hidden, len, 1})},
                {}, workspace, cuda);
    }
}

void kda_input_proj_conv_record(const Tensor& x, const SingleProjectionWeight& qkv, const Tensor& w,
                                KdaConvSourceState s, const Tensor& valid, KdaConvOutputs out,
                                Tensor& records, MemoryArena& workspace, CudaExecutionView cuda) {
    if (!records.data) throw std::invalid_argument("KDA input: missing convolution records");
    execute(x, qkv, w, s.history, {}, valid, out, records, workspace, cuda);
}
} // namespace ninfer::ops
