#include "ninfer/ops/kimi_delta_attention.h"
#include "ops/common/host/operands.h"
#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include "ops/linear_attention/kimi_delta_attention/launch.h"

namespace ninfer::ops {
namespace {
namespace kda            = detail::kimi_delta_attention;
constexpr const char* op = "kimi_delta_attention";

kda::Arguments arguments(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                         const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                         float lower_bound, float scale, const void* state_in, void* state_out,
                         DType state_dtype, Tensor& out) {
    return {static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const __nv_bfloat16*>(g.data),
            static_cast<const __nv_bfloat16*>(beta.data),
            static_cast<const float*>(a_log.data),
            static_cast<const float*>(dt_bias.data),
            state_in,
            state_out,
            state_dtype,
            static_cast<__nv_bfloat16*>(out.data),
            q.ne[1],
            v.ne[1],
            q.ne[2],
            lower_bound,
            scale};
}

kda::Arguments token_slice(kda::Arguments args, std::int64_t begin, int length) {
    args.tokens = length;
    args.q += begin * args.qk_heads * kda::kStateDim;
    args.k += begin * args.qk_heads * kda::kStateDim;
    args.v += begin * args.value_heads * kda::kStateDim;
    args.g += begin * args.value_heads * kda::kStateDim;
    args.beta += begin * args.value_heads;
    args.out += begin * args.value_heads * kda::kStateDim;
    return args;
}

void launch_single_sequence(const kda::Arguments& args, kda::Chunk* packets,
                            const kda::RecurrencePlan& plan, CudaExecutionView execution) {
    const auto stream = execution.stream();
    for (std::int64_t begin = 0; begin < args.tokens; begin += plan.window_tokens) {
        auto window = token_slice(args, begin,
                                  std::min<std::int64_t>(plan.window_tokens, args.tokens - begin));
        if (begin) {
            window.state_in      = args.state_out;
            window.source_offset = args.destination_offset;
        }
        kda::launch_prepare(window, packets, stream);
        kda::launch_chunk_recurrence(window, packets, plan.value_tile, execution);
    }
}
} // namespace

std::size_t kimi_delta_attention_workspace_capacity_bytes(int qk, int value, int first, int last) {
    if (!kda::valid_heads(qk, value) || first < 1 || last < first)
        throw std::invalid_argument("kimi_delta_attention workspace: invalid heads or interval");
    return kda::recurrence_workspace_capacity(value, first, last);
}

std::size_t kimi_delta_attention_ragged_workspace_capacity_bytes(int qk, int value, int min_tokens,
                                                                 int max_tokens, int min_segments,
                                                                 int max_segments) {
    (void)common::operand::ragged_envelope(min_tokens, max_tokens, min_segments, max_segments, op);
    if (!kda::valid_heads(qk, value))
        throw std::invalid_argument("kimi_delta_attention workspace: invalid heads");
    return kda::ragged_workspace_capacity(value, min_tokens, max_tokens, min_segments,
                                          max_segments);
}

void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                          float lower_bound, float scale, MemoryArena& workspace, Tensor& state,
                          Tensor& out, CudaExecutionView execution) {
    kimi_delta_attention(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, workspace, state,
                         state, out, execution);
}

void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                          float lower_bound, float scale, MemoryArena& workspace,
                          const Tensor& state_in, Tensor& state_out, Tensor& out,
                          CudaExecutionView execution) {
    common::operand::device(workspace.device(), execution.device(), op, "workspace");
    const auto p = kda::validate_inputs(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, out,
                                        execution.device());
    if (p.batch != 1) throw std::invalid_argument("kimi_delta_attention: full window requires B=1");
    if (!kda::is_supported_state_dtype(state_in.dtype) || state_in.dtype != state_out.dtype)
        throw std::invalid_argument(
            "kimi_delta_attention: recurrent state must be matching FP32 or BF16");
    const DType state_dtype = state_in.dtype;
    for (const auto* state : {&state_in, static_cast<const Tensor*>(&state_out)}) {
        common::operand::tensor(*state, state_dtype, {128, 128, p.value_heads}, 16, op, "State",
                                execution.device());
        for (const auto* t :
             {&q, &k, &v, &g, &beta, &a_log, &dt_bias, static_cast<const Tensor*>(&out)})
            common::operand::disjoint(common::operand::range(*state, op),
                                      common::operand::range(*t, op), op);
    }
    if (state_in.data != state_out.data)
        common::operand::disjoint(common::operand::range(state_in, op),
                                  common::operand::range(state_out, op), op);
    const auto args = arguments(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, state_in.data,
                                state_out.data, state_dtype, out);
    const auto plan =
        kda::select_recurrence_plan(p.value_heads, p.tokens, execution.multiprocessor_count());
    if (plan.route == kda::RecurrenceRoute::Recurrent) {
        kda::launch_recurrent(args, execution.stream());
        return;
    }
    auto scope    = workspace.scope();
    auto* packets = static_cast<kda::Chunk*>(workspace.alloc_bytes(plan.workspace_bytes).data);
    for (const auto* tensor :
         {&q, &k, &v, &g, &beta, &a_log, &dt_bias, &state_in,
          static_cast<const Tensor*>(&state_out), static_cast<const Tensor*>(&out)})
        common::operand::disjoint(common::operand::range(packets, plan.workspace_bytes, op),
                                  common::operand::range(*tensor, op), op);
    launch_single_sequence(args, packets, plan, execution);
}

void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                          float lower_bound, float scale, const RaggedSequenceView& ragged,
                          MemoryArena& workspace, ConstStateFieldView source,
                          StateFieldView destination, Tensor& out, CudaExecutionView execution) {
    common::operand::device(workspace.device(), execution.device(), op, "workspace");
    common::operand::ragged_sequence(ragged, op, execution.device());
    const auto p = kda::validate_inputs(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, out,
                                        execution.device());
    if (p.batch != 1 || p.tokens != ragged.total_tokens())
        throw std::invalid_argument("kimi_delta_attention: ragged shape mismatch");
    const kda::Geometry state_geometry{p.qk_heads, p.value_heads, 1, ragged.num_seqs()};
    const DType state_dtype = kda::validate_state(source, state_geometry, execution.device());
    if (kda::validate_state(destination, state_geometry, execution.device()) != state_dtype)
        throw std::invalid_argument(
            "kimi_delta_attention: source and destination state dtypes must match");
    const Tensor* metadata[] = {&ragged.cu_seqlens, &ragged.seq_lengths, &ragged.token_to_seq,
                                &source.records.byte_offsets, &destination.records.byte_offsets};
    for (const auto* meta : metadata)
        common::operand::disjoint(common::operand::range(out, op),
                                  common::operand::range(*meta, op), op);

    auto args          = arguments(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale,
                                   common::operand::state_field_base<const void>(source),
                                   common::operand::state_field_base<void>(destination), state_dtype,
                                   out);
    args.source_offset = static_cast<const std::int64_t*>(source.records.byte_offsets.data);
    args.destination_offset =
        static_cast<const std::int64_t*>(destination.records.byte_offsets.data);
    const int num_seqs = ragged.num_seqs();
    const int sms            = execution.multiprocessor_count();
    const auto stream        = execution.stream();
    const auto check_scratch = [&](MemorySpan span) {
        const auto scratch = common::operand::range(span.data, span.bytes, op);
        for (const auto* tensor :
             {&q, &k, &v, &g, &beta, &a_log, &dt_bias, static_cast<const Tensor*>(&out)})
            common::operand::disjoint(scratch, common::operand::range(*tensor, op), op);
        for (const auto* tensor : metadata)
            common::operand::disjoint(scratch, common::operand::range(*tensor, op), op);
    };
    const auto launch_direct = [&](const kda::Arguments& sequence, bool has_short,
                                   kda::RaggedSegments segments) {
        const auto plan = kda::select_recurrence_plan(p.value_heads, sequence.tokens, sms);
        if (plan.route == kda::RecurrenceRoute::Recurrent) {
            kda::launch_recurrent(sequence, stream);
            return;
        }
        auto scope      = workspace.scope();
        const auto span = workspace.alloc_bytes(plan.workspace_bytes);
        check_scratch(span);
        if (has_short)
            kda::launch_ragged_recurrent(args, segments, kda::kFirstChunkedTokens, stream);
        launch_single_sequence(sequence, static_cast<kda::Chunk*>(span.data), plan, execution);
    };
    if (num_seqs == 1) {
        launch_direct(args, false, {});
        return;
    }

    kda::PackedChunks packed{};
    packed.segments      = {static_cast<const int*>(ragged.cu_seqlens.data),
                            static_cast<const int*>(ragged.seq_lengths.data), num_seqs};
    int chunked_segments = 0, total_chunks = 0;
    for (int s = 0; s < num_seqs; ++s) {
        const int length = ragged.segment_length(s);
        if (length < kda::kFirstChunkedTokens) continue;
        if (chunked_segments < kda::kInlineSeqs)
            packed.inline_index[chunked_segments] = {s, total_chunks};
        ++chunked_segments;
        total_chunks += kda::chunk_count(length);
    }
    if (chunked_segments == 0) {
        kda::launch_ragged_recurrent(args, packed.segments, kda::kFirstChunkedTokens, stream);
        return;
    }
    const bool has_short = chunked_segments < num_seqs;
    if (chunked_segments == 1) {
        const int s   = packed.inline_index[0].segment;
        auto sequence = token_slice(args, ragged.segment_begin(s), ragged.segment_length(s));
        sequence.source_offset += s;
        sequence.destination_offset += s;
        launch_direct(sequence, has_short, packed.segments);
        return;
    }
    if (sms <= 0)
        throw std::invalid_argument("kimi_delta_attention: positive multiprocessor count required");

    const auto layout = kda::ragged_chunk_workspace(p.value_heads, total_chunks, chunked_segments);
    auto scope        = workspace.scope();
    const auto span   = workspace.alloc_bytes(layout.bytes);
    check_scratch(span);
    auto* packets = static_cast<kda::Chunk*>(layout.packets.bind(span).data);
    if (has_short)
        kda::launch_ragged_recurrent(args, packed.segments, kda::kFirstChunkedTokens, stream);
    if (layout.index.bytes) {
        auto* index         = static_cast<kda::ChunkIndexEntry*>(layout.index.bind(span).data);
        packed.device_index = index;
        kda::launch_chunk_index(packed.segments, index, stream);
    }

    const int window_chunks = layout.window_chunks;
    const auto next_chunks  = [&](int& sequence) {
        int length;
        do {
            length = ragged.segment_length(++sequence);
        } while (length < kda::kFirstChunkedTokens);
        return kda::chunk_count(length);
    };
    int sequence = -1, last_segment = 0, segment_end = next_chunks(sequence);
    for (int begin = 0; begin < total_chunks;) {
        if (segment_end == begin) {
            ++last_segment;
            segment_end += next_chunks(sequence);
        }
        const int first_segment = last_segment;
        const int end           = begin + std::min(window_chunks, total_chunks - begin);
        int longest_in_window   = std::min(segment_end, end) - begin;
        int longest_segment     = first_segment;
        while (segment_end < end) {
            ++last_segment;
            const int chunks  = next_chunks(sequence);
            const int overlap = std::min(chunks, end - segment_end);
            if (overlap > longest_in_window) {
                longest_in_window = overlap;
                longest_segment   = last_segment;
            }
            segment_end += chunks;
        }
        const int active_segments = last_segment - first_segment + 1;
        const int tile = kda::select_value_tile(p.value_heads, sms, active_segments, end - begin,
                                                longest_in_window);
        packed.window  = {begin, end, first_segment, active_segments, longest_segment};
        kda::launch_prepare(args, packets, packed, stream);
        kda::launch_chunk_recurrence(args, packets, tile, packed, execution);
        begin = end;
    }
}

void kimi_delta_attention_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, const Tensor& a_log,
                                       const Tensor& dt_bias, float lower_bound, float scale,
                                       ConstStateFieldView source, StateFieldView destination,
                                       const Tensor& valid, Tensor& out,
                                       CudaExecutionView execution) {
    const auto p = kda::validate_inputs(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, out,
                                        execution.device());
    if (p.tokens != 1)
        throw std::invalid_argument("kimi_delta_attention: batch update requires T=1");
    const DType state_dtype = kda::validate_state(source, p, execution.device());
    if (kda::validate_state(destination, p, execution.device()) != state_dtype)
        throw std::invalid_argument(
            "kimi_delta_attention: source and destination state dtypes must match");
    if (valid.data) {
        common::operand::tensor(valid, DType::I32, {p.batch}, 4, op, "valid columns",
                                execution.device());
        common::operand::disjoint(common::operand::range(out, op),
                                  common::operand::range(valid, op), op);
    }
    for (const auto* metadata : {&source.records.byte_offsets, &destination.records.byte_offsets})
        common::operand::disjoint(common::operand::range(out, op),
                                  common::operand::range(*metadata, op), op);
    kda::launch_batch_update(
        arguments(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale,
                  common::operand::state_field_base<const void>(source),
                  common::operand::state_field_base<void>(destination), state_dtype, out),
        static_cast<const std::int64_t*>(source.records.byte_offsets.data),
        static_cast<const std::int64_t*>(destination.records.byte_offsets.data),
        static_cast<const std::int32_t*>(valid.data), p.batch, execution.stream());
}

void kimi_delta_attention_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& g, const Tensor& beta, const Tensor& a_log,
                                        const Tensor& dt_bias, float lower_bound, float scale,
                                        ConstStateFieldView source, const Tensor& valid,
                                        KdaRecurrenceRecordView records, Tensor& out,
                                        CudaExecutionView execution) {
    const auto p = kda::validate_inputs(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, out,
                                        execution.device());
    const DType state_dtype = kda::validate_state(source, p, execution.device());
    if (valid.data)
        common::operand::tensor(valid, DType::I32, {p.batch}, 4, op, "valid columns",
                                execution.device());
    const auto [key, delta, alpha] = ninfer::detail::KdaReplayRecordAccess::planes(records);
    common::operand::tensor(key, DType::BF16, {128, p.qk_heads, p.tokens, p.batch}, 16, op,
                            "key record", execution.device());
    common::operand::tensor(delta, DType::FP32, {128, p.value_heads, p.tokens, p.batch}, 16, op,
                            "delta record", execution.device());
    common::operand::tensor(alpha, DType::FP32, {128, p.value_heads, p.tokens, p.batch}, 16, op,
                            "alpha record", execution.device());
    const Tensor* outputs[] = {&key, &delta, &alpha, &out};
    for (std::size_t i = 0; i < std::size(outputs); ++i) {
        for (std::size_t j = 0; j < i; ++j)
            common::operand::disjoint(common::operand::range(*outputs[i], op),
                                      common::operand::range(*outputs[j], op), op);
        for (const auto* input :
             {&q, &k, &v, &g, &beta, &a_log, &dt_bias,
              static_cast<const Tensor*>(&source.records.byte_offsets), &valid}) {
            if (!input->data) continue;
            common::operand::disjoint(common::operand::range(*outputs[i], op),
                                      common::operand::range(*input, op), op);
        }
    }
    kda::launch_record(
        arguments(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale,
                  common::operand::state_field_base<const void>(source), nullptr, state_dtype, out),
        static_cast<const std::int64_t*>(source.records.byte_offsets.data),
        static_cast<const std::int32_t*>(valid.data),
        {static_cast<__nv_bfloat16*>(key.data), static_cast<float*>(delta.data),
         static_cast<float*>(alpha.data)},
        p.batch, execution.stream());
}
} // namespace ninfer::ops
