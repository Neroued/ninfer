#include "ninfer/ops/causal_conv1d_silu.h"
#include <algorithm>
#include <array>
#include "ops/common/host/operands.h"
#include "ops/linear_attention/causal_conv/launch.h"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::ops {
namespace {

void require_x_shape(const Tensor& x) {
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: x must have shape [C,T]");
    }
    if (x.ne[0] <= 0) { throw std::invalid_argument("causal_conv1d: C must be positive"); }
}

void require_snapshot_x_shape(const Tensor& x) {
    constexpr std::int32_t kMaximumWidth = 16;
    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    if (x.ne[0] <= 0 || width <= 0 || batch <= 0 || x.ne[3] != 1 ||
        (batch > 1 && width > kMaximumWidth) ||
        std::int64_t(width) * batch > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("causal_conv1d: unsupported snapshot [C,W,B] shape");
    }
}

void require_weight_shape(const Tensor& weight, std::int32_t C) {
    if (weight.ne[0] != C || weight.ne[1] != 4 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: weight must have shape [C,4]");
    }
}

void require_state_shape(const Tensor& conv_state, std::int32_t C) {
    if (conv_state.ne[0] != C || conv_state.ne[1] != 3 || conv_state.ne[2] != 1 ||
        conv_state.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: conv_state must have shape [C,3]");
    }
}

void require_snapshot_state_shape(const Tensor& conv_states, std::int32_t C, std::int32_t width,
                                  std::int32_t batch) {
    if (conv_states.ne[0] != C || conv_states.ne[1] != 3 ||
        conv_states.ne[2] < std::int64_t(width) * batch || conv_states.ne[3] != 1) {
        throw std::invalid_argument(
            "causal_conv1d: conv_states must have shape [C,3,S] with S>=B*W");
    }
}

void require_out_shape(const Tensor& x, const Tensor& out) {
    for (int d = 0; d < 4; ++d) {
        if (out.ne[d] != x.ne[d]) {
            throw std::invalid_argument("causal_conv1d: out shape must match x");
        }
    }
}

void require_non_empty_accessible(const Tensor& x, const Tensor& weight, const Tensor& conv_state,
                                  const Tensor& out, Device device) {
    common::operand::contiguous_nonnull(x, "causal_conv1d", "x", device);
    common::operand::contiguous_nonnull(weight, "causal_conv1d", "weight", device);
    common::operand::contiguous_nonnull(conv_state, "causal_conv1d", "conv_state", device);
    common::operand::contiguous_nonnull(out, "causal_conv1d", "out", device);
}

detail::CausalConvSplitGeometry require_split_geometry(const Tensor& x, const Tensor& out0,
                                                       const Tensor& out1, const Tensor& out2) {
    if (out0.dtype != DType::BF16 || out1.dtype != DType::BF16 || out2.dtype != DType::BF16) {
        throw std::invalid_argument("causal_conv1d: split destinations must be BF16");
    }
    if (out0.ne[2] != 1 || out0.ne[3] != 1 || out1.ne[2] != 1 || out1.ne[3] != 1 ||
        out2.ne[2] != 1 || out2.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: split destinations must be rank-2");
    }
    if (out0.ne[1] != x.ne[1] || out1.ne[1] != x.ne[1] || out2.ne[1] != x.ne[1]) {
        throw std::invalid_argument("causal_conv1d: split destinations must match the x column "
                                    "count");
    }
    if (x.ne[0] == 8192 && out0.ne[0] == 2048 && out1.ne[0] == 2048 && out2.ne[0] == 4096) {
        return detail::CausalConvSplitGeometry::Rows2048x2048x4096;
    }
    if (x.ne[0] == 10240 && out0.ne[0] == 2048 && out1.ne[0] == 2048 && out2.ne[0] == 6144) {
        return detail::CausalConvSplitGeometry::Rows2048x2048x6144;
    }
    if (x.ne[0] == 5120 && out0.ne[0] == 1024 && out1.ne[0] == 1024 && out2.ne[0] == 3072) {
        return detail::CausalConvSplitGeometry::Rows1024x1024x3072;
    }
    throw std::invalid_argument("causal_conv1d: split received an unregistered row profile");
}

// The prefill split route reads and writes __nv_bfloat162. The requirement is stated uniformly for
// the entry rather than per route, so a caller does not have to know which route its column count
// selects.
void require_pair_aligned(const Tensor& tensor, const char* label) {
    constexpr std::uintptr_t kPairBytes = 4; // one BF16 pair
    const auto address                  = reinterpret_cast<std::uintptr_t>(tensor.data);
    if ((address & (kPairBytes - 1)) != 0) {
        throw std::invalid_argument(std::string("causal_conv1d: ") + label +
                                    " must be four-byte aligned");
    }
}

// The contract allows the two state arguments to be disjoint or exactly the same storage, and
// nothing in between: every route either reads a state element before publishing it from another
// thread, or declares both pointers restrict.
void require_state_alias_rule(const Tensor& conv_state_in, const Tensor& conv_state_out) {
    if (conv_state_in.data == conv_state_out.data) { return; }
    if (common::operand::overlaps(common::operand::range(conv_state_in, "causal_conv1d"),
                                  common::operand::range(conv_state_out, "causal_conv1d"))) {
        throw std::invalid_argument("causal_conv1d: conv_state_in and conv_state_out must be "
                                    "disjoint or exactly the same storage");
    }
}

void require_output_nonoverlap(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                               const Tensor& conv_state_out, const Tensor& out) {
    const Tensor* const written[2] = {&out, &conv_state_out};
    const Tensor* const others[3]  = {&x, &weight, &conv_state_in};
    for (const Tensor* target : written) {
        for (const Tensor* other : others) {
            // The state pair is governed by require_state_alias_rule, which admits an exact alias.
            if (target == &conv_state_out && other == &conv_state_in) { continue; }
            if (common::operand::overlaps(common::operand::range(*target, "causal_conv1d"),
                                          common::operand::range(*other, "causal_conv1d"))) {
                throw std::invalid_argument(
                    "causal_conv1d: out and conv_state_out must not overlap x, weight, or "
                    "conv_state_in");
            }
        }
    }
    if (common::operand::overlaps(common::operand::range(out, "causal_conv1d"),
                                  common::operand::range(conv_state_out, "causal_conv1d"))) {
        throw std::invalid_argument("causal_conv1d: out must not overlap conv_state_out");
    }
}

void require_split_nonoverlap(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                              const Tensor& conv_state_out, const Tensor& out0, const Tensor& out1,
                              const Tensor& out2) {
    const Tensor* const writes[4] = {&out0, &out1, &out2, &conv_state_out};
    const Tensor* const others[4] = {&x, &weight, &conv_state_in, &conv_state_out};
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            if (common::operand::overlaps(common::operand::range(*writes[i], "causal_conv1d"),
                                          common::operand::range(*writes[j], "causal_conv1d"))) {
                throw std::invalid_argument(
                    "causal_conv1d: split destinations must not overlap each other");
            }
        }
        for (const Tensor* other : others) {
            // The state pair is governed by require_state_alias_rule, which admits an exact alias.
            if (writes[i] == &conv_state_out && other == &conv_state_in) { continue; }
            if (writes[i] == other) { continue; }
            if (common::operand::overlaps(common::operand::range(*writes[i], "causal_conv1d"),
                                          common::operand::range(*other, "causal_conv1d"))) {
                throw std::invalid_argument(
                    "causal_conv1d: split destinations must not overlap x, weight, or the state");
            }
        }
    }
}

} // namespace

void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                        Tensor& conv_state_out, Tensor& out, CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 ||
        conv_state_in.dtype != DType::BF16 || conv_state_out.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("causal_conv1d: x/weight/conv_state/out must be BF16");
    }

    const std::int64_t n = common::operand::numel(x, "causal_conv1d");
    (void)common::operand::numel(weight, "causal_conv1d");
    (void)common::operand::numel(conv_state_in, "causal_conv1d");
    (void)common::operand::numel(conv_state_out, "causal_conv1d");
    (void)common::operand::numel(out, "causal_conv1d");

    require_x_shape(x);
    require_weight_shape(weight, x.ne[0]);
    require_state_shape(conv_state_in, x.ne[0]);
    require_state_shape(conv_state_out, x.ne[0]);
    require_out_shape(x, out);
    if (n == 0) { return; }

    require_non_empty_accessible(x, weight, conv_state_in, out, execution.device());
    common::operand::contiguous_nonnull(conv_state_out, "causal_conv1d", "conv_state_out",
                                        execution.device());
    require_state_alias_rule(conv_state_in, conv_state_out);
    require_output_nonoverlap(x, weight, conv_state_in, conv_state_out, out);
    const auto plan =
        detail::select_causal_conv_plan(x, weight, conv_state_in, conv_state_out, out);
    detail::launch_causal_conv(x, weight, conv_state_in, conv_state_out, out, plan, stream);
}

void causal_conv1d_silu_split(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                              Tensor& conv_state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                              CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 ||
        conv_state_in.dtype != DType::BF16 || conv_state_out.dtype != DType::BF16) {
        throw std::invalid_argument("causal_conv1d: x/weight/conv_state must be BF16");
    }
    require_x_shape(x);
    require_weight_shape(weight, x.ne[0]);
    require_state_shape(conv_state_in, x.ne[0]);
    require_state_shape(conv_state_out, x.ne[0]);
    const detail::CausalConvSplitGeometry geometry = require_split_geometry(x, out0, out1, out2);

    const std::int64_t n = common::operand::numel(x, "causal_conv1d");
    (void)common::operand::numel(weight, "causal_conv1d");
    (void)common::operand::numel(conv_state_in, "causal_conv1d");
    (void)common::operand::numel(conv_state_out, "causal_conv1d");
    (void)common::operand::numel(out0, "causal_conv1d");
    (void)common::operand::numel(out1, "causal_conv1d");
    (void)common::operand::numel(out2, "causal_conv1d");
    if (n == 0) { return; }

    const std::pair<const Tensor*, const char*> operands[7] = {{&x, "x"},
                                                               {&weight, "weight"},
                                                               {&conv_state_in, "conv_state_in"},
                                                               {&conv_state_out, "conv_state_out"},
                                                               {&out0, "out0"},
                                                               {&out1, "out1"},
                                                               {&out2, "out2"}};
    for (const auto& operand : operands) {
        common::operand::contiguous_nonnull(*operand.first, "causal_conv1d", operand.second,
                                            execution.device());
        require_pair_aligned(*operand.first, operand.second);
    }
    require_state_alias_rule(conv_state_in, conv_state_out);
    require_split_nonoverlap(x, weight, conv_state_in, conv_state_out, out0, out1, out2);

    detail::launch_causal_conv_split(x, weight, conv_state_in, conv_state_out, out0, out1, out2,
                                     geometry, stream);
}

void causal_conv1d_silu(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                        CudaExecutionView execution) {
    causal_conv1d_silu(x, weight, conv_state, conv_state, out, execution);
}

void causal_conv1d_silu_snapshot(const Tensor& x, const Tensor& weight, ConstStateFieldView source,
                                 Tensor& snapshots, const Tensor& valid_columns,
                                 const Tensor& snapshot_base_slots, Tensor& out,
                                 CudaExecutionView execution) {
    const bool masked = valid_columns.data != nullptr;
    require_snapshot_x_shape(x);
    const auto channels = x.ne[0];
    const auto batch    = x.ne[2];
    common::operand::tensor(x, DType::BF16, {channels, x.ne[1], batch}, 2, "causal_conv1d", "x",
                            execution.device());
    common::operand::tensor(weight, DType::BF16, {channels, 4}, 2, "causal_conv1d", "weight",
                            execution.device());
    common::operand::tensor(out, DType::BF16, {channels, x.ne[1], batch}, 2, "causal_conv1d", "out",
                            execution.device());
    common::operand::state_field(source, DType::BF16, {channels, 3}, 2, "causal_conv1d", "source",
                                 execution.device());
    if (source.field.layers != 1 || source.records.byte_offsets.ne[0] != batch) {
        throw std::invalid_argument("causal_conv1d: source row count differs from batch");
    }
    require_snapshot_state_shape(snapshots, channels, x.ne[1], batch);
    common::operand::tensor(snapshots, DType::BF16, {channels, 3, snapshots.ne[2]}, 2,
                            "causal_conv1d", "snapshots", execution.device());
    if (masked) {
        common::operand::tensor(valid_columns, DType::I32, {batch}, alignof(std::int32_t),
                                "causal_conv1d", "valid columns", execution.device());
    }
    common::operand::tensor(snapshot_base_slots, DType::I32, {batch}, alignof(std::int32_t),
                            "causal_conv1d", "snapshot bases", execution.device());
    const std::array<const Tensor*, 7> operands{&x,
                                                &weight,
                                                &snapshots,
                                                &valid_columns,
                                                &source.records.byte_offsets,
                                                &snapshot_base_slots,
                                                &out};
    for (std::size_t i = 0; i < operands.size(); ++i) {
        if (!operands[i]->data) continue;
        for (std::size_t j = i + 1; j < operands.size(); ++j) {
            if (operands[j]->data &&
                common::operand::overlaps(common::operand::range(*operands[i], "causal_conv1d"),
                                          common::operand::range(*operands[j], "causal_conv1d"))) {
                throw std::invalid_argument("causal_conv1d: snapshot operands overlap");
            }
        }
    }
    detail::launch_causal_conv_snapshot(
        x, weight, source, snapshots, valid_columns, snapshot_base_slots, out,
        detail::select_causal_conv_snapshot_plan(x.ne[1], batch, masked), execution.stream());
}

namespace {
namespace operand                    = common::operand;
constexpr const char* ragged_conv_op = "causal_conv1d";

template <class Launch>
void ragged_conv(const Tensor& x, const Tensor& weight, const RaggedSequenceView& seq,
                 ConstStateFieldView source, StateFieldView destination,
                 std::span<Tensor* const> outputs, std::size_t alignment, Launch launch,
                 CudaExecutionView execution) {
    const auto device       = execution.device();
    constexpr const auto op = ragged_conv_op;
    operand::ragged_sequence(seq, op, device);
    const int c = x.ne[0], n = seq.total_tokens(), rows = seq.num_seqs();
    operand::tensor(x, DType::BF16, {c, n}, alignment, op, "input", device);
    operand::tensor(weight, DType::BF16, {c, 4}, alignment, op, "weight", device);
    for (const auto& field : std::array<ConstStateFieldView, 2>{source, destination}) {
        operand::state_field(field, DType::BF16, {c, 3}, alignment, op, "state", device);
        operand::shape(field.records.byte_offsets, {rows, 1, 1, 1}, op, "state offsets");
        if (field.field.layers != 1)
            throw std::invalid_argument("causal_conv1d: select one convolution layer");
    }
    const Tensor* inputs[] = {&x,
                              &weight,
                              &seq.cu_seqlens,
                              &seq.seq_lengths,
                              &seq.token_to_seq,
                              &source.records.byte_offsets,
                              &destination.records.byte_offsets};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto& out = *outputs[i];
        operand::tensor(out, DType::BF16, {out.ne[0], n}, alignment, op, "output", device);
        for (const auto* input : inputs)
            operand::disjoint(operand::range(out, op), operand::range(*input, op), op);
        for (std::size_t j = 0; j < i; ++j)
            operand::disjoint(operand::range(out, op), operand::range(*outputs[j], op), op);
    }

    auto in = source.field.metadata(), out = destination.field.metadata();
    // These are field bases; only the kernels resolve the selected records.
    in.data   = const_cast<std::byte*>(operand::state_field_base<const std::byte>(source));
    out.data  = operand::state_field_base<std::byte>(destination);
    in.device = out.device = device;
    const auto* src        = static_cast<const std::int64_t*>(source.records.byte_offsets.data);
    const auto* dst = static_cast<const std::int64_t*>(destination.records.byte_offsets.data);
    const detail::CausalConvRagged ragged{{src, dst},
                                          static_cast<const std::int32_t*>(seq.cu_seqlens.data),
                                          static_cast<const std::int32_t*>(seq.seq_lengths.data),
                                          static_cast<const std::int32_t*>(seq.token_to_seq.data),
                                          n,
                                          rows};
    // Fixed-partition capture permits the short route to be selected from the host mirror.
    const bool short_segments =
        rows > 1 && rows <= 64 &&
        std::adjacent_find(seq.host_cu_seqlens.begin(), seq.host_cu_seqlens.end(),
                           [](int begin, int end) { return end - begin > 16; }) ==
            seq.host_cu_seqlens.end();
    launch(in, out, ragged, short_segments);
}
} // namespace

void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const RaggedSequenceView& seq,
                        ConstStateFieldView source, StateFieldView destination, Tensor& out,
                        CudaExecutionView execution) {
    require_x_shape(x);
    require_out_shape(x, out);
    const std::array outputs{&out};
    ragged_conv(
        x, weight, seq, source, destination, outputs, 2,
        [&](const Tensor& src, Tensor& dst, detail::CausalConvRagged ragged, bool short_segments) {
            if (ragged.segments > 1)
                return detail::launch_causal_conv_ragged(x, weight, src, dst, out, ragged,
                                                         short_segments, execution.stream());
            auto route = detail::select_causal_conv_plan(x, weight, src, dst, out);
            // Device-selected scalar fields only guarantee natural BF16 alignment.
            if (route == detail::CausalConvRoute::PrefillPairs)
                route = detail::CausalConvRoute::PrefillScalar;
            detail::launch_causal_conv(x, weight, src, dst, out, route, execution.stream(),
                                       ragged.offsets);
        },
        execution);
}

void causal_conv1d_silu_split(const Tensor& x, const Tensor& weight, const RaggedSequenceView& seq,
                              ConstStateFieldView source, StateFieldView destination, Tensor& out0,
                              Tensor& out1, Tensor& out2, CudaExecutionView execution) {
    const auto geometry = require_split_geometry(x, out0, out1, out2);
    const std::array outputs{&out0, &out1, &out2};
    ragged_conv(
        x, weight, seq, source, destination, outputs, 4,
        [&](const Tensor& src, Tensor& dst, detail::CausalConvRagged ragged, bool short_segments) {
            if (ragged.segments > 1)
                return detail::launch_causal_conv_split_ragged(x, weight, src, dst, out0, out1,
                                                               out2, geometry, ragged,
                                                               short_segments, execution.stream());
            detail::launch_causal_conv_split(x, weight, src, dst, out0, out1, out2, geometry,
                                             execution.stream(), ragged.offsets);
        },
        execution);
}

} // namespace ninfer::ops
