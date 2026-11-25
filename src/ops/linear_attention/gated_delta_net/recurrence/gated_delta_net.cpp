#include "ops/common/host/operands.h"
#include "ninfer/ops/gated_delta_net.h"
#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrence/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"
#include "ops/linear_attention/gated_delta_net/recurrence/launch.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

namespace gdn     = detail::gated_delta_net;
namespace chunked = gdn::chunked;
using gdn::RecurrenceGeometry;

RecurrenceGeometry require_geometry(const Tensor& q, const Tensor& v) {
    const RecurrenceGeometry geometry{q.ne[1], v.ne[1], q.ne[2]};
    if (q.ne[0] != detail::gated_delta_net::kStateDim) {
        throw std::invalid_argument("gated_delta_net: state/head dimension must be 128");
    }
    if (!detail::gated_delta_net::are_head_counts_valid(geometry.qk_heads, geometry.value_heads)) {
        throw std::invalid_argument(
            "gated_delta_net: value heads must be at least q/k heads and divisible by them");
    }
    if (geometry.tokens <= 0) {
        throw std::invalid_argument("gated_delta_net: T must be positive");
    }
    return geometry;
}

RecurrenceGeometry validate_recurrent(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& g, const Tensor& beta, float scale,
                                      const Tensor& ssm_state, const Tensor& out, Device device) {
    common::operand::dtype(q, DType::BF16, "gated_delta_net", "q must be BF16");
    common::operand::dtype(k, DType::BF16, "gated_delta_net", "k must be BF16");
    common::operand::dtype(v, DType::BF16, "gated_delta_net", "v must be BF16");
    common::operand::dtype(out, DType::BF16, "gated_delta_net", "out must be BF16");
    common::operand::dtype(g, DType::FP32, "gated_delta_net", "g must be FP32");
    common::operand::dtype(beta, DType::FP32, "gated_delta_net", "beta must be FP32");
    if (!detail::gated_delta_net::is_supported_state_dtype(ssm_state.dtype)) {
        throw std::invalid_argument("gated_delta_net: ssm_state must be FP32 or BF16");
    }

    const RecurrenceGeometry geometry = require_geometry(q, v);
    common::operand::shape(
        q, {detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, 1},
        "gated_delta_net", "q");
    common::operand::shape(
        k, {detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, 1},
        "gated_delta_net", "k");
    common::operand::shape(
        v, {detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, 1},
        "gated_delta_net", "v");
    common::operand::shape(
        out, {detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, 1},
        "gated_delta_net", "out");
    common::operand::shape(g, {geometry.value_heads, geometry.tokens, 1, 1}, "gated_delta_net",
                           "g");
    common::operand::shape(beta, {geometry.value_heads, geometry.tokens, 1, 1}, "gated_delta_net",
                           "beta");
    common::operand::shape(ssm_state,
                           {detail::gated_delta_net::kStateDim, detail::gated_delta_net::kStateDim,
                            geometry.value_heads, 1},
                           "gated_delta_net", "ssm_state");

    common::operand::contiguous_nonnull(q, "gated_delta_net", "q", device);
    common::operand::contiguous_nonnull(k, "gated_delta_net", "k", device);
    common::operand::contiguous_nonnull(v, "gated_delta_net", "v", device);
    common::operand::contiguous_nonnull(g, "gated_delta_net", "g", device);
    common::operand::contiguous_nonnull(beta, "gated_delta_net", "beta", device);
    common::operand::contiguous_nonnull(ssm_state, "gated_delta_net", "ssm_state", device);
    common::operand::contiguous_nonnull(out, "gated_delta_net", "out", device);

    detail::gated_delta_net::require_scale(scale, "gated_delta_net");
    return geometry;
}

RecurrenceGeometry
validate_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                                const Tensor& beta, float scale, ConstStateFieldView source_state,
                                StateFieldView destination_state, const Tensor& valid_columns,
                                const Tensor& out, Device device) {
    common::operand::dtype(q, DType::BF16, "gated_delta_net", "q must be BF16");
    common::operand::dtype(k, DType::BF16, "gated_delta_net", "k must be BF16");
    common::operand::dtype(v, DType::BF16, "gated_delta_net", "v must be BF16");
    common::operand::dtype(out, DType::BF16, "gated_delta_net", "out must be BF16");
    common::operand::dtype(g, DType::FP32, "gated_delta_net", "g must be FP32");
    common::operand::dtype(beta, DType::FP32, "gated_delta_net", "beta must be FP32");

    const RecurrenceGeometry geometry = require_geometry(q, v);
    const std::int32_t batch          = q.ne[3];
    if (batch <= 0 || geometry.tokens != 1) {
        throw std::invalid_argument("gated_delta_net: batch update requires positive B and W=1");
    }
    common::operand::shape(
        q, {detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, batch},
        "gated_delta_net", "q");
    common::operand::shape(
        k, {detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, batch},
        "gated_delta_net", "k");
    common::operand::shape(
        v, {detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, batch},
        "gated_delta_net", "v");
    common::operand::shape(
        out, {detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, batch},
        "gated_delta_net", "out");
    common::operand::shape(g, {geometry.value_heads, geometry.tokens, batch, 1}, "gated_delta_net",
                           "g");
    common::operand::shape(beta, {geometry.value_heads, geometry.tokens, batch, 1},
                           "gated_delta_net", "beta");
    const DType state_dtype = source_state.field.tensor.dtype;
    if (!detail::gated_delta_net::is_supported_state_dtype(state_dtype) ||
        destination_state.field.tensor.dtype != state_dtype) {
        throw std::invalid_argument(
            "gated_delta_net: source and destination recurrent states must be matching FP32 or "
            "BF16");
    }
    for (const auto& state : std::array<ConstStateFieldView, 2>{source_state, destination_state}) {
        if (state.field.layers != 1)
            throw std::invalid_argument("gated_delta_net: select one recurrent state layer");
        common::operand::state_field(state, state_dtype, {128, 128, geometry.value_heads}, 16,
                                     "gated_delta_net", "recurrent state", device);
        common::operand::shape(state.records.byte_offsets, {batch, 1, 1, 1}, "gated_delta_net",
                               "state byte offsets");
    }

    common::operand::contiguous_nonnull(q, "gated_delta_net", "q", device);
    common::operand::contiguous_nonnull(k, "gated_delta_net", "k", device);
    common::operand::contiguous_nonnull(v, "gated_delta_net", "v", device);
    common::operand::contiguous_nonnull(g, "gated_delta_net", "g", device);
    common::operand::contiguous_nonnull(beta, "gated_delta_net", "beta", device);
    common::operand::contiguous_nonnull(out, "gated_delta_net", "out", device);
    if (valid_columns.data)
        common::operand::tensor(valid_columns, DType::I32, {batch}, 4, "gated_delta_net",
                                "valid columns", device);

    detail::gated_delta_net::require_scale(scale, "gated_delta_net");

    return geometry;
}

void validate_recurrence(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                         const Tensor& beta, float scale, const Tensor& ssm_state_in,
                         const Tensor& ssm_state_out, const Tensor& out, Device device) {
    // ssm_state_out carries the running-state contract validated by validate_recurrent;
    // ssm_state_in is an equally-shaped read view (may alias ssm_state_out for in-place).
    const RecurrenceGeometry geometry =
        validate_recurrent(q, k, v, g, beta, scale, ssm_state_out, out, device);
    common::operand::dtype(ssm_state_in, ssm_state_out.dtype, "gated_delta_net",
                           "ssm_state_in dtype must match ssm_state_out");
    common::operand::shape(ssm_state_in,
                           {detail::gated_delta_net::kStateDim, detail::gated_delta_net::kStateDim,
                            geometry.value_heads, 1},
                           "gated_delta_net", "ssm_state_in");
    common::operand::contiguous_nonnull(ssm_state_in, "gated_delta_net", "ssm_state_in", device);
}

gdn::RecurrenceOperands make_operands(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& g, const Tensor& beta, float scale,
                                      const Tensor& ssm_state_in, Tensor& ssm_state_out,
                                      Tensor& out, const std::int64_t* source_offset,
                                      const std::int64_t* destination_offset) {
    return {static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const float*>(g.data),
            static_cast<const float*>(beta.data),
            ssm_state_in.data,
            ssm_state_out.data,
            ssm_state_out.dtype,
            static_cast<__nv_bfloat16*>(out.data),
            q.ne[1],
            v.ne[1],
            q.ne[2],
            scale,
            source_offset,
            destination_offset};
}

void execute(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g, const Tensor& beta,
             float scale, bool normalize_qk, MemoryArena& ws, const Tensor& ssm_state_in,
             Tensor& ssm_state_out, Tensor& out, CudaExecutionView execution,
             const std::int64_t* source_offset      = nullptr,
             const std::int64_t* destination_offset = nullptr);

} // namespace

std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t qk_heads,
                                                     std::int32_t value_heads,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (!detail::gated_delta_net::are_head_counts_valid(qk_heads, value_heads) || min_tokens <= 0 ||
        max_tokens < min_tokens) {
        throw std::invalid_argument("gated_delta_net workspace: invalid profile or interval");
    }
    return detail::gated_delta_net::recurrence_workspace_layout(qk_heads, value_heads, max_tokens)
        .total_bytes;
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, MemoryArena& ws,
                     Tensor& ssm_state, Tensor& out, CudaExecutionView execution) {
    gated_delta_net(q, k, v, g, beta, scale, normalize_qk, ws, ssm_state, ssm_state, out,
                    execution);
}

void gated_delta_net_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  bool normalize_qk, ConstStateFieldView source_state,
                                  StateFieldView destination_state, const Tensor& valid_columns,
                                  Tensor& out, CudaExecutionView execution) {
    const auto stream = execution.stream();
    validate_recurrent_batch_update(q, k, v, g, beta, scale, source_state, destination_state,
                                    valid_columns, out, execution.device());

    detail::gated_delta_net::launch_recurrent_batch_update(q, k, v, g, beta, scale, normalize_qk,
                                                           source_state, destination_state,
                                                           valid_columns, out, stream);
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, MemoryArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     CudaExecutionView execution) {
    if (ws.device() != execution.device()) {
        throw std::invalid_argument("gated_delta_net: workspace belongs to another device");
    }
    validate_recurrence(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out,
                        execution.device());
    execute(q, k, v, g, beta, scale, normalize_qk, ws, ssm_state_in, ssm_state_out, out, execution);
}

namespace {
void execute(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g, const Tensor& beta,
             float scale, bool normalize_qk, MemoryArena& ws, const Tensor& ssm_state_in,
             Tensor& ssm_state_out, Tensor& out, CudaExecutionView execution,
             const std::int64_t* source_offset, const std::int64_t* destination_offset) {
    const auto plan = detail::gated_delta_net::select_recurrence_plan(
        q.ne[1], v.ne[1], q.ne[2], execution.multiprocessor_count());
    if (plan.route == detail::gated_delta_net::RecurrenceRoute::Recurrent) {
        detail::gated_delta_net::launch_recurrent_inout(
            q, k, v, g, beta, scale, normalize_qk, ssm_state_in, ssm_state_out, out,
            execution.stream(), source_offset, destination_offset);
        return;
    }
    auto scope         = ws.scope();
    const auto& layout = plan.workspace;
    const auto backing = ws.alloc_bytes(layout.total_bytes);
    auto* qk           = static_cast<chunked::QkChunk*>(layout.qk.bind(backing).data);
    auto* control      = static_cast<chunked::ControlChunk*>(layout.control.bind(backing).data);
    const auto args    = make_operands(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out,
                                       source_offset, destination_offset);
    chunked::launch_prepare(args, qk, control, normalize_qk, execution.stream());
    chunked::launch_recurrence(args, qk, control, plan.value_tile, execution.stream());
}

template <class Visit>
void for_each_chunk_window(const RaggedSequenceView& seq, gdn::RaggedSegments segments,
                           int first_chunked, int cap, Visit&& visit) {
    chunked::PackedChunks window{segments};
    int longest = 0;
    for (int s = 0; s < segments.count; ++s) {
        const int length = seq.segment_length(s);
        if (length < first_chunked) continue;
        const int chunks = chunked::chunk_count(length);
        if (window.chunked_segments && chunks > cap - window.total_chunks) {
            visit(window, longest);
            window.chunked_segments = 0;
            window.total_chunks     = 0;
            longest                 = 0;
        }
        window.inline_index[window.chunked_segments++] = {s, window.total_chunks};
        window.total_chunks += chunks;
        longest = std::max(longest, length);
    }
    if (window.chunked_segments) visit(window, longest);
}

void launch_chunk_window(const gdn::RecurrenceOperands& args, const RaggedSequenceView& seq,
                         const chunked::PackedChunks& window, int longest, chunked::QkChunk* qk,
                         chunked::ControlChunk* control, bool normalize_qk,
                         CudaExecutionView execution) {
    const auto stream = execution.stream();
    const int tile    = gdn::select_ragged_value_tile(args.qk_heads, args.value_heads, longest,
                                                      window.chunked_segments, window.total_chunks,
                                                      execution.multiprocessor_count());
    if (window.chunked_segments == 1) {
        // A sole segment can exceed the packed cap and uses the direct chunked kernels.
        const int s            = window.inline_index[0].segment;
        const int begin        = seq.segment_begin(s);
        const auto qk_start    = static_cast<std::int64_t>(begin) * args.qk_heads * gdn::kStateDim;
        const auto gate_start  = static_cast<std::int64_t>(begin) * args.value_heads;
        const auto value_start = gate_start * gdn::kStateDim;
        auto local             = args;
        local.q += qk_start;
        local.k += qk_start;
        local.v += value_start;
        local.out += value_start;
        local.g += gate_start;
        local.beta += gate_start;
        local.tokens = longest;
        local.source_offset += s;
        local.destination_offset += s;
        chunked::launch_prepare(local, qk, control, normalize_qk, stream);
        chunked::launch_recurrence(local, qk, control, tile, stream);
    } else {
        chunked::launch_prepare_ragged(args, window, qk, control, normalize_qk, stream);
        chunked::launch_recurrence_ragged(args, window, qk, control, tile, stream);
    }
}
} // namespace

std::size_t gated_delta_net_ragged_workspace_capacity_bytes(int qk_heads, int value_heads,
                                                            int min_tokens, int max_tokens,
                                                            int min_segments, int max_segments) {
    (void)common::operand::ragged_envelope(min_tokens, max_tokens, min_segments, max_segments,
                                           "gated_delta_net");
    if (!gdn::are_head_counts_valid(qk_heads, value_heads)) {
        throw std::invalid_argument("gated_delta_net workspace: invalid head counts");
    }
    return gdn::ragged_workspace_capacity(qk_heads, value_heads, max_tokens, min_segments,
                                          max_segments);
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk,
                     const RaggedSequenceView& seq, MemoryArena& ws, ConstStateFieldView source,
                     StateFieldView destination, Tensor& out, CudaExecutionView execution) {
    namespace operand        = common::operand;
    constexpr const char* op = "gated_delta_net";
    const auto device        = execution.device();
    operand::ragged_sequence(seq, op, device);
    operand::device(ws.device(), device, op, "workspace");
    const int n            = seq.total_tokens(), h = v.ne[1];
    const DType state_type = source.field.tensor.dtype;
    if (!gdn::is_supported_state_dtype(state_type) ||
        destination.field.tensor.dtype != state_type) {
        throw std::invalid_argument(
            "gated_delta_net: source and destination recurrent states must be matching FP32 or "
            "BF16");
    }
    for (const auto& state : std::array<ConstStateFieldView, 2>{source, destination}) {
        operand::state_field(state, state_type, {128, 128, h}, 16, op, "state", device);
        operand::shape(state.records.byte_offsets, {seq.num_seqs(), 1, 1, 1}, op, "state offsets");
        if (state.field.layers != 1)
            throw std::invalid_argument("gated_delta_net: select one recurrent state layer");
    }
    auto src = source.field.metadata(), dst = destination.field.metadata();
    src.data   = const_cast<void*>(operand::state_field_base<const void>(source));
    dst.data   = operand::state_field_base<void>(destination);
    src.device = dst.device = device;
    const auto geometry     = validate_recurrent(q, k, v, g, beta, scale, dst, out, device);
    if (n != geometry.tokens) throw std::invalid_argument("gated_delta_net: ragged token count");
    const Tensor* inputs[] = {&q,
                              &k,
                              &v,
                              &g,
                              &beta,
                              &seq.cu_seqlens,
                              &seq.seq_lengths,
                              &seq.token_to_seq,
                              &source.records.byte_offsets,
                              &destination.records.byte_offsets};
    const auto output      = operand::range(out, op);
    for (const auto* in : inputs) operand::disjoint(output, operand::range(*in, op), op);
    const auto check_workspace = [&](MemorySpan backing) {
        const auto scratch = operand::range(backing.data, backing.bytes, op);
        for (const auto* in : inputs) operand::disjoint(scratch, operand::range(*in, op), op);
        operand::disjoint(scratch, output, op);
    };
    const auto* src_offsets = static_cast<const std::int64_t*>(source.records.byte_offsets.data);
    const auto* dst_offsets =
        static_cast<const std::int64_t*>(destination.records.byte_offsets.data);
    const int segments = seq.num_seqs();
    if (segments == 1) {
        const auto bytes = gdn::recurrence_workspace_layout(geometry.qk_heads, h, n).total_bytes;
        if (bytes) {
            auto scope = ws.scope();
            check_workspace(ws.alloc_bytes(bytes));
        }
        execute(q, k, v, g, beta, scale, normalize_qk, ws, src, dst, out, execution, src_offsets,
                dst_offsets);
        return;
    }

    const int first = gdn::first_chunked_tokens(geometry.qk_heads, h);
    chunked::PackedChunks packed{};
    packed.segments = {static_cast<const int*>(seq.cu_seqlens.data),
                       static_cast<const int*>(seq.seq_lengths.data), segments};
    int longest     = 0;
    for (int s = 0; s < segments; ++s) {
        const int length = seq.segment_length(s);
        if (length < first) continue;
        longest = std::max(longest, length);
        if (packed.chunked_segments < chunked::kInlineSeqs)
            packed.inline_index[packed.chunked_segments] = {s, packed.total_chunks};
        ++packed.chunked_segments;
        packed.total_chunks += chunked::chunk_count(length);
    }
    auto args = make_operands(q, k, v, g, beta, scale, src, dst, out, src_offsets, dst_offsets);
    const auto stream = execution.stream();
    if (packed.chunked_segments == 0) {
        gdn::launch_recurrent_ragged(args, packed.segments, normalize_qk, 0, nullptr, stream);
        return;
    }
    const int cap       = gdn::max_packed_window_chunks(geometry.qk_heads, h);
    const bool windowed = packed.chunked_segments >= 2 && packed.total_chunks > cap;
    int peak_chunks = packed.total_chunks, peak_segments = packed.chunked_segments;
    if (windowed) {
        peak_chunks   = 0;
        peak_segments = 0;
        for_each_chunk_window(seq, packed.segments, first, cap,
                              [&](const chunked::PackedChunks& window, int) {
                                  peak_chunks   = std::max(peak_chunks, window.total_chunks);
                                  peak_segments = std::max(peak_segments, window.chunked_segments);
                              });
    }
    const auto layout =
        chunked::packed_workspace_layout(geometry.qk_heads, h, peak_chunks, peak_segments);
    auto scope         = ws.scope();
    const auto backing = ws.alloc_bytes(layout.total_bytes);
    check_workspace(backing);
    if (layout.index.bytes)
        packed.device_index =
            static_cast<chunked::ChunkIndexEntry*>(layout.index.bind(backing).data);
    auto* qk             = static_cast<chunked::QkChunk*>(layout.qk.bind(backing).data);
    auto* control        = static_cast<chunked::ControlChunk*>(layout.control.bind(backing).data);
    const bool has_short = packed.chunked_segments != segments;
    if (packed.device_index) {
        if (has_short) {
            gdn::launch_recurrent_ragged(args, packed.segments, normalize_qk, first,
                                         packed.device_index, stream);
        } else {
            chunked::launch_build_chunk_index(packed.segments, first, packed.device_index, stream);
        }
    }
    if (windowed) {
        for_each_chunk_window(seq, packed.segments, first, cap,
                              [&](const chunked::PackedChunks& window, int longest_in_window) {
                                  launch_chunk_window(args, seq, window, longest_in_window, qk,
                                                      control, normalize_qk, execution);
                              });
    } else {
        launch_chunk_window(args, seq, packed, longest, qk, control, normalize_qk, execution);
    }
    if (has_short && !packed.device_index)
        gdn::launch_recurrent_ragged(args, packed.segments, normalize_qk, first, nullptr, stream);
}

} // namespace ninfer::ops
