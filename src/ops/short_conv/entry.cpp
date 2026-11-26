#include "ninfer/ops/short_conv.h"

#include "launch.h"
#include "ops/common/host/operands.h"

#include <limits>

namespace ninfer::ops {
namespace {
namespace operand = common::operand;
using short_conv_detail::Conv;
using short_conv_detail::Word;
using attn_res_detail::require_disjoint;

constexpr const char* kConvOp  = "short_conv_residual";
constexpr const char* kFusedOp = "short_conv_block_attn_res";

const Word* words(const Tensor& t) { return static_cast<const Word*>(t.data); }

// Exactly one state destination, and either batched or packed token metadata.
struct Layout {
    ConstStateFieldView source;
    const StateFieldView* destination;
    Tensor* snapshots;
    const Tensor* slots;
    const Tensor* valid;
    const RaggedSequenceView* ragged;
};

Conv prepare(const Tensor& x, const Tensor& weight, const Layout& layout, const Tensor& residual,
             const Tensor& addend, Tensor& out, Device device, const char* op) {
    if (layout.ragged) operand::ragged_sequence(*layout.ragged, op, device);
    const auto d = x.ne[0], width = layout.ragged ? layout.ragged->total_tokens() : x.ne[1];
    const auto batch   = layout.ragged ? 1 : x.ne[2];
    const auto rows    = layout.ragged ? layout.ragged->num_seqs() : batch;
    const auto columns = std::int64_t(width) * batch;
    if (d <= 0 || width <= 0 || batch <= 0 || columns > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string(op) + ": invalid input dimensions");
    if (residual.data && !addend.data)
        throw std::invalid_argument(std::string(op) + ": residual requires addend");

    const std::size_t alignment = d % 8 == 0 ? 16 : 2;
    const auto shape            = {d, width, batch};
    operand::tensor(x, DType::BF16, shape, alignment, op, "x", device);
    operand::tensor(weight, DType::BF16, {d, 4}, alignment, op, "weight", device);
    operand::tensor(out, DType::BF16, shape, alignment, op, "conv_out", device);
    if (residual.data)
        operand::tensor(residual, DType::BF16, shape, alignment, op, "residual", device);
    if (addend.data) operand::tensor(addend, DType::BF16, shape, alignment, op, "addend", device);

    const auto validate_state = [&](const auto& field, const char* name) {
        operand::state_field(field, DType::BF16, {d, 3}, 2, op, name, device);
        operand::shape(field.records.byte_offsets, {rows, 1, 1, 1}, op, "state offsets");
        if (field.field.layers != 1)
            throw std::invalid_argument(std::string(op) + ": select one state layer");
    };
    validate_state(layout.source, "state_in");
    const auto& source_offsets = layout.source.records.byte_offsets;
    const auto* valid          = layout.valid && layout.valid->data ? layout.valid : nullptr;
    if (valid) operand::tensor(*valid, DType::I32, {rows}, 4, op, "valid columns", device);

    require_disjoint(out, {&x, &weight, &addend, &source_offsets}, op);
    if (out.data != residual.data) require_disjoint(out, {&residual}, op);
    if (valid) require_disjoint(out, {valid}, op);

    const auto src = layout.source.records.storage;
    operand::Range destination_storage;
    Word* destination;
    const std::int64_t* dst_offsets = nullptr;
    const std::int32_t* slots       = nullptr;
    if (layout.snapshots) {
        auto& snapshots = *layout.snapshots;
        if (snapshots.ne[2] < (valid ? 1 : width))
            throw std::invalid_argument(std::string(op) + ": insufficient snapshot slots");
        operand::tensor(snapshots, DType::BF16, {d, 3, snapshots.ne[2]}, 2, op, "snapshots",
                        device);
        operand::tensor(*layout.slots, DType::I32, {rows}, 4, op, "snapshot slots", device);
        require_disjoint(
            snapshots, {&x, &weight, &out, &residual, &addend, &source_offsets, layout.slots}, op);
        if (valid) require_disjoint(snapshots, {valid}, op);
        require_disjoint(out, {layout.slots}, op);
        destination         = static_cast<Word*>(snapshots.data);
        destination_storage = operand::range(snapshots, op);
        slots               = static_cast<const std::int32_t*>(layout.slots->data);
    } else {
        const auto& state = *layout.destination;
        validate_state(state, "state_out");
        require_disjoint(out, {&state.records.byte_offsets}, op);
        destination = operand::state_field_base<Word>(state);
        destination_storage =
            operand::range(state.records.storage.data, state.records.storage.bytes, op);
        dst_offsets = static_cast<const std::int64_t*>(state.records.byte_offsets.data);
    }

    const std::int32_t* prefix       = nullptr;
    const std::int32_t* token_to_seq = nullptr;
    if (layout.ragged) {
        const auto& seq = *layout.ragged;
        require_disjoint(out, {&seq.cu_seqlens, &seq.seq_lengths, &seq.token_to_seq}, op);
        prefix       = static_cast<const std::int32_t*>(seq.cu_seqlens.data);
        token_to_seq = static_cast<const std::int32_t*>(seq.token_to_seq.data);
    }

    return {words(x),
            words(weight),
            words(residual),
            words(addend),
            operand::state_field_base<const Word>(layout.source),
            static_cast<const std::int64_t*>(source_offsets.data),
            dst_offsets,
            valid ? static_cast<const std::int32_t*>(valid->data) : nullptr,
            slots,
            prefix,
            token_to_seq,
            static_cast<Word*>(out.data),
            destination,
            d,
            static_cast<int>(columns),
            width,
            operand::overlaps(operand::range(src.data, src.bytes, op), destination_storage)};
}

void fused(const Tensor& x, const Tensor& weight, const Layout& layout, const Tensor& residual,
           const Tensor& conv_addend, const Tensor& stack, const Tensor& attnres_addend,
           const Tensor& queries, const Tensor& norm_weight, const BlockAttnResParameters& params,
           Tensor& conv_out, Tensor& mixed_out, CudaExecutionView execution) {
    const auto device = execution.device();
    const auto conv = prepare(x, weight, layout, residual, conv_addend, conv_out, device, kFusedOp);
    const auto mix  = attn_res_detail::prepare_mix(stack, attnres_addend, queries, norm_weight,
                                                   params, mixed_out, conv.d, conv.width,
                                                   conv.columns / conv.width, device, kFusedOp);
    require_disjoint(conv_out, {&stack, &attnres_addend, &queries, &norm_weight}, kFusedOp);
    require_disjoint(
        mixed_out,
        {&x, &weight, &conv_out, &residual, &conv_addend, &layout.source.records.byte_offsets},
        kFusedOp);
    if (layout.valid) require_disjoint(mixed_out, {layout.valid}, kFusedOp);
    if (layout.snapshots) {
        require_disjoint(mixed_out, {layout.snapshots, layout.slots}, kFusedOp);
        require_disjoint(*layout.snapshots, {&stack, &attnres_addend, &queries, &norm_weight},
                         kFusedOp);
    } else {
        require_disjoint(mixed_out, {&layout.destination->records.byte_offsets}, kFusedOp);
    }
    if (layout.ragged)
        require_disjoint(
            mixed_out,
            {&layout.ragged->cu_seqlens, &layout.ragged->seq_lengths, &layout.ragged->token_to_seq},
            kFusedOp);
    short_conv_detail::launch_fused(conv, mix, execution);
}

} // namespace

void short_conv_residual(const Tensor& x, const Tensor& weight, ConstStateFieldView state_in,
                         StateFieldView state_out, const Tensor& valid_columns,
                         const Tensor& residual, const Tensor& addend, Tensor& conv_out,
                         CudaExecutionView execution) {
    const Layout layout{state_in, &state_out, nullptr, nullptr, &valid_columns, nullptr};
    short_conv_detail::launch(
        prepare(x, weight, layout, residual, addend, conv_out, execution.device(), kConvOp),
        execution);
}

void short_conv_residual_snapshot(const Tensor& x, const Tensor& weight,
                                  ConstStateFieldView state_in, Tensor& snapshots,
                                  const Tensor& valid_columns, const Tensor& snapshot_base_slots,
                                  const Tensor& residual, const Tensor& addend, Tensor& conv_out,
                                  CudaExecutionView execution) {
    const Layout layout{state_in,       nullptr, &snapshots, &snapshot_base_slots,
                        &valid_columns, nullptr};
    short_conv_detail::launch(
        prepare(x, weight, layout, residual, addend, conv_out, execution.device(), kConvOp),
        execution);
}

void short_conv_residual(const Tensor& x, const Tensor& weight, const RaggedSequenceView& ragged,
                         ConstStateFieldView state_in, StateFieldView state_out,
                         const Tensor& residual, const Tensor& addend, Tensor& conv_out,
                         CudaExecutionView execution) {
    const Layout layout{state_in, &state_out, nullptr, nullptr, nullptr, &ragged};
    short_conv_detail::launch(
        prepare(x, weight, layout, residual, addend, conv_out, execution.device(), kConvOp),
        execution);
}

void short_conv_block_attn_res(const Tensor& x, const Tensor& conv_weight,
                               ConstStateFieldView state_in, StateFieldView state_out,
                               const Tensor& valid_columns, const Tensor& conv_residual,
                               const Tensor& conv_addend, const Tensor& stack,
                               const Tensor& attnres_addend, const Tensor& queries,
                               const Tensor& norm_weight, const BlockAttnResParameters& params,
                               Tensor& conv_out, Tensor& mixed_out, CudaExecutionView execution) {
    fused(x, conv_weight, {state_in, &state_out, nullptr, nullptr, &valid_columns, nullptr},
          conv_residual, conv_addend, stack, attnres_addend, queries, norm_weight, params, conv_out,
          mixed_out, execution);
}

void short_conv_block_attn_res_snapshot(
    const Tensor& x, const Tensor& conv_weight, ConstStateFieldView state_in, Tensor& snapshots,
    const Tensor& valid_columns, const Tensor& snapshot_base_slots, const Tensor& conv_residual,
    const Tensor& conv_addend, const Tensor& stack, const Tensor& attnres_addend,
    const Tensor& queries, const Tensor& norm_weight, const BlockAttnResParameters& params,
    Tensor& conv_out, Tensor& mixed_out, CudaExecutionView execution) {
    fused(x, conv_weight,
          {state_in, nullptr, &snapshots, &snapshot_base_slots, &valid_columns, nullptr},
          conv_residual, conv_addend, stack, attnres_addend, queries, norm_weight, params, conv_out,
          mixed_out, execution);
}

void short_conv_block_attn_res(const Tensor& x, const Tensor& conv_weight,
                               const RaggedSequenceView& ragged, ConstStateFieldView state_in,
                               StateFieldView state_out, const Tensor& conv_residual,
                               const Tensor& conv_addend, const Tensor& stack,
                               const Tensor& attnres_addend, const Tensor& queries,
                               const Tensor& norm_weight, const BlockAttnResParameters& params,
                               Tensor& conv_out, Tensor& mixed_out, CudaExecutionView execution) {
    fused(x, conv_weight, {state_in, &state_out, nullptr, nullptr, nullptr, &ragged}, conv_residual,
          conv_addend, stack, attnres_addend, queries, norm_weight, params, conv_out, mixed_out,
          execution);
}

} // namespace ninfer::ops
