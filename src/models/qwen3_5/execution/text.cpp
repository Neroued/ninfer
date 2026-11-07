#include "models/qwen3_5/execution/text.h"

#include "core/nvtx.h"
#include "core/transfer.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/dense_mlp.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/workspace.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sigmoid_mul.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_shape(const Tensor& value, DType dtype, std::initializer_list<std::int32_t> shape,
                   const char* name) {
    bool matches     = value.dtype == dtype && value.data && value.is_contiguous();
    std::size_t axis = 0;
    for (const auto extent : shape) matches &= value.ne[axis++] == extent;
    for (; axis < 4; ++axis) matches &= value.ne[axis] == 1;
    if (!matches) throw std::invalid_argument(std::string(name) + " has invalid shape or layout");
}

bool is_sequence(const TextStateUpdate& update) {
    return std::holds_alternative<SequenceUpdate>(update) ||
           std::holds_alternative<PackedSequenceUpdate>(update) ||
           std::holds_alternative<MixedUpdate>(update);
}

// Segments cover [0, tokens) in order, address distinct rows of the [rows] metadata outside
// [reserved_first, reserved_first + reserved_count) and leave every published State separate
// from every other segment's State.
void validate_segments(std::span<const SequenceSegment> segments, std::int32_t tokens,
                       std::int32_t rows, bool states, std::int32_t reserved_first = 0,
                       std::int32_t reserved_count = 0) {
    if (segments.empty()) throw std::invalid_argument("Main packed update has no segment");
    std::int64_t next = 0;
    for (std::size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = segments[i];
        const auto row      = static_cast<std::int64_t>(segment.row);
        if (segment.tokens == 0 || segment.token_offset != next || row >= rows ||
            (row >= reserved_first && row < std::int64_t(reserved_first) + reserved_count))
            throw std::invalid_argument("Main packed segment has an invalid extent or row");
        if (states && (segment.source_offset < 0 || segment.destination_offset < 0))
            throw std::invalid_argument("Main packed segment State offsets are invalid");
        next += segment.tokens;
        for (std::size_t j = 0; j < i; ++j) {
            const auto& other = segments[j];
            if (other.row == segment.row ||
                (states && (other.destination_offset == segment.destination_offset ||
                            other.destination_offset == segment.source_offset ||
                            segment.destination_offset == other.source_offset)))
                throw std::invalid_argument("Main packed segments share a row or State");
        }
    }
    if (next != tokens)
        throw std::invalid_argument("Main packed segments do not cover the token axis");
}

} // namespace

TextRun::TextRun(TextExecutionView execution, TextInputs inputs, TextStateUpdate update,
                 ops::CausalAttentionExecutionEnvelope envelope, Tensor hidden,
                 const DFlashFeatureSink* features)
    : view_(execution), inputs_(std::move(inputs)), update_(update), envelope_(envelope),
      hidden_(hidden), traversal_scope_(view_.workspace.scope()) {
    const auto& config = view_.parameters.profile->config.text;
    const auto width   = inputs_.ids.ne[0];
    const auto batch   = inputs_.ids.ne[1];
    if (width <= 0 || batch <= 0 ||
        std::int64_t(width) * batch > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("Main input extent must fit positive int32");
    const auto columns  = width * batch;
    const auto channels = dimension(config.hidden_size);
    require_shape(inputs_.ids, DType::I32, {width, batch}, "Main IDs");
    require_shape(inputs_.cache_positions, DType::I32, {width, batch}, "Main cache positions");
    if (inputs_.rope_positions.ne[1] != 1 && inputs_.rope_positions.ne[1] != 3)
        throw std::invalid_argument("Main RoPE positions require one or three axes");
    require_shape(inputs_.rope_positions, DType::I32, {columns, inputs_.rope_positions.ne[1]},
                  "Main RoPE positions");
    const auto* packed = std::get_if<PackedSequenceUpdate>(&update_);
    const auto* mixed  = std::get_if<MixedUpdate>(&update_);
    const auto rows    = packed || mixed ? inputs_.kv_table_rows.ne[0] : batch;
    if ((packed || mixed) && (batch != 1 || rows <= 0 || !inputs_.valid_columns.data))
        throw std::invalid_argument(
            "Packed and mixed updates require one token axis and row metadata");
    require_shape(inputs_.kv_table_rows, DType::I32, {rows}, "Main KV table rows");
    if (inputs_.valid_columns.data)
        require_shape(inputs_.valid_columns, DType::I32, {rows}, "Main valid columns");
    require_shape(hidden_, DType::BF16, {channels, width, batch}, "Main hidden output");
    if (inputs_.input_embeddings)
        require_shape(*inputs_.input_embeddings, DType::BF16, {channels, width, batch},
                      "Main input embeddings");
    if (packed) {
        if (features) throw std::invalid_argument("PackedSequenceUpdate has no feature sink");
        validate_segments(packed->segments, width, rows, config.gdn.has_value());
    } else if (mixed) {
        const auto& steps = mixed->steps;
        if (features) throw std::invalid_argument("MixedUpdate has no feature sink");
        if (inputs_.input_embeddings)
            throw std::invalid_argument("MixedUpdate carries text tokens only");
        if (steps.width == 0 || steps.rows == 0 || steps.width > 16 ||
            std::int64_t(steps.row_offset) + steps.rows > rows ||
            std::int64_t(steps.token_offset) + std::int64_t(steps.width) * steps.rows != width)
            throw std::invalid_argument("Main mixed decode rows do not close the token axis");
        validate_segments(mixed->segments, static_cast<std::int32_t>(steps.token_offset), rows,
                          config.gdn.has_value(), static_cast<std::int32_t>(steps.row_offset),
                          static_cast<std::int32_t>(steps.rows));
        if (config.gdn) {
            const auto step_rows = static_cast<std::int32_t>(steps.rows);
            require_shape(inputs_.state_source_offsets, DType::I64, {step_rows},
                          "Main source State offsets");
            if (!mixed->records) {
                if (steps.width != 1)
                    throw std::invalid_argument("StepUpdate requires one token per row");
                require_shape(inputs_.state_destination_offsets, DType::I64, {step_rows},
                              "Main destination State offsets");
            }
        }
    } else if (const auto* sequence = std::get_if<SequenceUpdate>(&update_)) {
        if (batch != 1) throw std::invalid_argument("SequenceUpdate requires one sequence");
        if (config.gdn && (sequence->source_offset < 0 || sequence->destination_offset < 0))
            throw std::invalid_argument("Main sequence State offsets are invalid");
    } else if (config.gdn) {
        require_shape(inputs_.state_source_offsets, DType::I64, {batch},
                      "Main source State offsets");
        if (std::holds_alternative<StepUpdate>(update_)) {
            if (width != 1) throw std::invalid_argument("StepUpdate requires one token per row");
            require_shape(inputs_.state_destination_offsets, DType::I64, {batch},
                          "Main destination State offsets");
        }
    }
    if (features) {
        const auto element_bytes = static_cast<std::int64_t>(dtype_size(DType::BF16));
        if (features->layers.empty() || !features->features.data ||
            features->features.dtype != DType::BF16 ||
            features->features.ne[0] !=
                std::int64_t(channels) * static_cast<std::int64_t>(features->layers.size()) ||
            features->features.nb[0] != element_bytes ||
            features->features.nb[1] < std::int64_t(features->features.ne[0]) * element_bytes)
            throw std::invalid_argument("Main feature sink has invalid geometry");
        if (features->residency_slots.data) {
            require_shape(features->residency_slots, DType::I32, {batch},
                          "Main feature residency slots");
            require_shape(inputs_.valid_columns, DType::I32, {batch}, "Main feature valid columns");
            if (features->features.ne[1] != width || features->features.ne[3] != 1)
                throw std::invalid_argument("Main feature sink has invalid sequence width");
        } else {
            if (features->features.ne[1] < columns || features->features.ne[2] != 1 ||
                features->features.ne[3] != 1)
                throw std::invalid_argument("Main compact feature sink has invalid capacity");
            if (features->positions.data &&
                (features->positions.dtype != DType::I32 || features->positions.numel() < columns ||
                 !features->positions.is_contiguous()))
                throw std::invalid_argument("Main feature position sink has invalid capacity");
        }
        for (const auto layer : features->layers)
            if (layer >= view_.parameters.text.layers.size())
                throw std::invalid_argument("Main feature layer is outside the model");
        features_ = *features;
    }
    residual_ = view_.workspace.alloc(DType::BF16, {channels, columns});
}

TextRun::~TextRun() {
    if (stage_ == Stage::Layer || stage_ == Stage::PostMixer) {
        try {
            cancel();
        } catch (...) { std::terminate(); }
    }
}

void TextRun::set_attention_envelope(ops::CausalAttentionExecutionEnvelope envelope) noexcept {
    envelope_ = envelope;
}

void TextRun::prefix() {
    if (inputs_.input_embeddings) {
        const auto& embedding = *inputs_.input_embeddings;
        copy_bytes_async({residual_.data, residual_.bytes(), residual_.device},
                         {embedding.data, embedding.bytes(), embedding.device}, view_.execution);
    } else {
        ops::embedding(inputs_.ids.view({residual_.ne[1]}), view_.parameters.text.token_embedding,
                       residual_, view_.execution);
    }
}

void TextRun::attention(const BlockParameters& block, std::uint32_t compact_layer) {
    auto& work        = view_.workspace;
    const auto cuda   = view_.execution;
    const auto& cfg   = view_.parameters.profile->config.text;
    const auto& attn  = *cfg.attention;
    const auto& p     = std::get<AttentionParameters>(block.mixer);
    const auto tokens = residual_.ne[1];
    const auto width  = inputs_.ids.ne[0];
    const auto batch  = inputs_.ids.ne[1];
    const auto dim    = dimension(attn.head_dim);
    const auto heads  = dimension(attn.num_attention_heads);
    const auto kv     = dimension(attn.num_key_value_heads);

    const auto projection = workspace::text_attention_projection(work, cfg, tokens);
    Tensor normalized     = projection.hidden;
    ops::rmsnorm(residual_, block.input_norm, cfg.rms_norm_eps, true, normalized, cuda);
    Tensor query = projection.query;
    Tensor gate  = projection.gate;
    Tensor key   = projection.key;
    Tensor value = projection.value;
    attention_projection(normalized, p, query, gate, key, value, work, cuda);
    query = query.view({dim, heads, tokens});
    gate  = gate.view({dim, heads, tokens});
    key   = key.view({dim, kv, tokens});
    value = value.view({dim, kv, tokens});

    const auto results = workspace::text_attention_results(work, cfg, tokens);
    Tensor q           = results.normalized_query.view({dim, heads, tokens});
    Tensor k           = results.normalized_key.view({dim, kv, tokens});
    Tensor a           = results.attention.view({dim, heads, tokens});
    ops::rmsnorm(query, p.query_norm, cfg.rms_norm_eps, true, q, cuda);
    ops::rmsnorm(key, p.key_norm, cfg.rms_norm_eps, true, k, cuda);
    text_rope(inputs_.rope_positions, *cfg.rope_parameters, q, k, cuda);
    Tensor qb        = q.view({dim, heads, width, batch});
    Tensor kb        = k.view({dim, kv, width, batch});
    Tensor vb        = value.view({dim, kv, width, batch});
    Tensor ab        = a.view({dim, heads, width, batch});
    const auto scale = static_cast<float>(1.0 / std::sqrt(double(attn.head_dim)));
    const auto cache = view_.cache.batch_layer_view(compact_layer);
    const auto* packed = std::get_if<PackedSequenceUpdate>(&update_);
    const auto* mixed  = std::get_if<MixedUpdate>(&update_);
    if (packed || mixed) {
        // Each segment is an independent B=1 sequence, so it uses the existing Op with its own
        // positions, table row and valid length. The envelope bounds every segment's visible keys.
        for (const auto& segment : packed ? packed->segments : mixed->segments) {
            const auto begin  = static_cast<std::int32_t>(segment.token_offset);
            const auto length = static_cast<std::int32_t>(segment.tokens);
            const auto row    = static_cast<std::int32_t>(segment.row);
            Tensor qs         = q.slice(2, begin, length).view({dim, heads, length, 1});
            Tensor ks         = k.slice(2, begin, length).view({dim, kv, length, 1});
            Tensor vs         = value.slice(2, begin, length).view({dim, kv, length, 1});
            Tensor as         = a.slice(2, begin, length).view({dim, heads, length, 1});
            ops::causal_softmax_attention(
                qs, ks, vs, inputs_.cache_positions.slice(0, begin, length),
                inputs_.valid_columns.slice(0, row, 1), inputs_.kv_table_rows.slice(0, row, 1),
                {dim, heads, kv}, scale, cache, envelope_, work, as, cuda);
        }
        if (mixed) {
            // The decode rows are the [W,B] tail of the token axis and attend with their own
            // envelope.
            const auto& steps = mixed->steps;
            const auto begin  = static_cast<std::int32_t>(steps.token_offset);
            const auto count  = static_cast<std::int32_t>(steps.width * steps.rows);
            const auto w      = static_cast<std::int32_t>(steps.width);
            const auto b      = static_cast<std::int32_t>(steps.rows);
            const auto first  = static_cast<std::int32_t>(steps.row_offset);
            Tensor qs         = q.slice(2, begin, count).view({dim, heads, w, b});
            Tensor ks         = k.slice(2, begin, count).view({dim, kv, w, b});
            Tensor vs         = value.slice(2, begin, count).view({dim, kv, w, b});
            Tensor as         = a.slice(2, begin, count).view({dim, heads, w, b});
            ops::causal_softmax_attention(
                qs, ks, vs, inputs_.cache_positions.slice(0, begin, count).view({w, b}),
                inputs_.valid_columns.slice(0, first, b), inputs_.kv_table_rows.slice(0, first, b),
                {dim, heads, kv}, scale, cache, steps.envelope, work, as, cuda);
        }
    } else {
        ops::causal_softmax_attention(qb, kb, vb, inputs_.cache_positions, inputs_.valid_columns,
                                      inputs_.kv_table_rows, {dim, heads, kv}, scale, cache,
                                      envelope_, work, ab, cuda);
    }
    ops::sigmoid_mul(gate, a, cuda);
    ops::linear_add(a.view({dimension(attn.query_width()), tokens}), p.output.weight, residual_,
                    p.output.policy, work, cuda);
}

void TextRun::gdn(const BlockParameters& block, std::uint32_t compact_layer) {
    auto& work           = view_.workspace;
    const auto cuda      = view_.execution;
    const auto& cfg      = view_.parameters.profile->config.text;
    const auto& gdn      = *cfg.gdn;
    const auto& p        = std::get<GdnParameters>(block.mixer);
    const auto tokens    = residual_.ne[1];
    const auto width     = inputs_.ids.ne[0];
    const auto batch     = inputs_.ids.ne[1];
    const auto qh        = dimension(gdn.linear_num_key_heads);
    const auto vh        = dimension(gdn.linear_num_value_heads);
    const auto dk        = dimension(gdn.linear_key_head_dim);
    const auto dv        = dimension(gdn.linear_value_head_dim);
    const auto scale     = static_cast<float>(1.0 / std::sqrt(double(gdn.linear_key_head_dim)));
    const auto* sequence = std::get_if<SequenceUpdate>(&update_);
    const auto* packed   = std::get_if<PackedSequenceUpdate>(&update_);
    const auto* mixed    = std::get_if<MixedUpdate>(&update_);
    const auto* replay   = std::get_if<ReplayRecords>(&update_);
    // Segments run the chunked recurrence on a prefix of the token axis; a mixed update appends
    // decode rows as a [W,B] tail that uses the decode kernels.
    const auto segments =
        packed ? packed->segments : mixed ? mixed->segments : std::span<const SequenceSegment>{};
    const auto* record_source = replay ? &replay->records : mixed ? mixed->records : nullptr;
    const auto prefix_tokens = mixed ? static_cast<std::int32_t>(mixed->steps.token_offset) : tokens;

    const auto control = workspace::gdn_control(work, cfg, tokens);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    gdn_norm_control(residual_, block.input_norm, cfg.rms_norm_eps, p, h, g, beta, work, cuda);

    const auto projection = workspace::gdn_projection(work, cfg, tokens);
    Tensor z              = projection.output_gate.view({dv, vh, tokens});
    Tensor qc             = projection.query;
    Tensor kc             = projection.key;
    Tensor vc             = projection.value;
    if (sequence || packed || mixed) {
        Tensor qkv    = workspace::gdn_prefill_conv(work, cfg, tokens);
        Tensor z_flat = z.view({dimension(gdn.value_width()), tokens});
        {
            Tensor h_prefix   = h.slice(1, 0, prefix_tokens);
            Tensor qkv_prefix = qkv.slice(1, 0, prefix_tokens);
            Tensor z_prefix   = z_flat.slice(1, 0, prefix_tokens);
            gdn_projection(h_prefix, p, qkv_prefix, z_prefix, work, cuda);
        }
        if (sequence) {
            Tensor source = view_.states.conv_tensor(compact_layer, sequence->source_offset);
            Tensor destination =
                view_.states.conv_tensor(compact_layer, sequence->destination_offset);
            ops::causal_conv1d_silu_split(qkv, p.convolution, source, destination, qc, kc, vc,
                                          cuda);
        } else {
            // The convolution window is per sequence: each segment starts from its own State and
            // publishes its own trailing window.
            for (const auto& segment : segments) {
                const auto begin  = static_cast<std::int32_t>(segment.token_offset);
                const auto length = static_cast<std::int32_t>(segment.tokens);
                Tensor x          = qkv.slice(1, begin, length);
                Tensor query      = qc.slice(1, begin, length);
                Tensor key        = kc.slice(1, begin, length);
                Tensor value      = vc.slice(1, begin, length);
                Tensor source     = view_.states.conv_tensor(compact_layer, segment.source_offset);
                Tensor destination =
                    view_.states.conv_tensor(compact_layer, segment.destination_offset);
                ops::causal_conv1d_silu_split(x, p.convolution, source, destination, query, key,
                                              value, cuda);
            }
        }
        if (mixed) {
            // Decode rows project and convolve through the decode kernels on their tail columns.
            const auto& steps = mixed->steps;
            const auto begin  = static_cast<std::int32_t>(steps.token_offset);
            const auto count  = static_cast<std::int32_t>(steps.width * steps.rows);
            const auto w      = static_cast<std::int32_t>(steps.width);
            const auto b      = static_cast<std::int32_t>(steps.rows);
            const auto first  = static_cast<std::int32_t>(steps.row_offset);
            Tensor input      = h.slice(1, begin, count).view({dimension(cfg.hidden_size), w, b});
            Tensor query_out  = qc.slice(1, begin, count).view({dimension(gdn.key_width()), w, b});
            Tensor key_out    = kc.slice(1, begin, count).view({dimension(gdn.key_width()), w, b});
            Tensor value_out  = vc.slice(1, begin, count).view({dimension(gdn.value_width()), w, b});
            Tensor gate_out   = z_flat.slice(1, begin, count).view({dimension(gdn.value_width()), w, b});
            Tensor valid      = inputs_.valid_columns.slice(0, first, b);
            const auto source = view_.states.layer_view(compact_layer, inputs_.state_source_offsets);
            if (record_source) {
                const auto records = record_source->layer(compact_layer, b);
                Tensor conv_record = records.conv;
                gdn_projection_record(input, p, source.conv, valid, conv_record, query_out,
                                      key_out, value_out, gate_out, work, cuda);
            } else {
                gdn_projection_update(
                    input, p, source.conv,
                    view_.states.layer_view(compact_layer, inputs_.state_destination_offsets).conv,
                    valid, query_out, key_out, value_out, gate_out, work, cuda);
            }
        }
    } else {
        Tensor projection_input = h.view({dimension(cfg.hidden_size), width, batch});
        Tensor query_output     = qc.view({dimension(gdn.key_width()), width, batch});
        Tensor key_output       = kc.view({dimension(gdn.key_width()), width, batch});
        Tensor value_output     = vc.view({dimension(gdn.value_width()), width, batch});
        Tensor gate_output      = z.view({dimension(gdn.value_width()), width, batch});
        const auto source = view_.states.layer_view(compact_layer, inputs_.state_source_offsets);
        if (replay) {
            const auto records = replay->records.layer(compact_layer, batch);
            Tensor conv_record = records.conv;
            gdn_projection_record(projection_input, p, source.conv, inputs_.valid_columns,
                                  conv_record, query_output, key_output, value_output, gate_output,
                                  work, cuda);
        } else {
            gdn_projection_update(
                projection_input, p, source.conv,
                view_.states.layer_view(compact_layer, inputs_.state_destination_offsets).conv,
                inputs_.valid_columns, query_output, key_output, value_output, gate_output, work,
                cuda);
        }
    }

    Tensor q = qc.view({dk, qh, tokens});
    Tensor k = kc.view({dk, qh, tokens});
    Tensor v = vc.view({dv, vh, tokens});
    Tensor o = workspace::gdn_recurrent_output(work, cfg, tokens).view({dv, vh, tokens});
    if (sequence) {
        Tensor source = view_.states.recurrent_tensor(compact_layer, sequence->source_offset);
        Tensor destination =
            view_.states.recurrent_tensor(compact_layer, sequence->destination_offset);
        ops::gated_delta_net(q, k, v, g, beta, scale, true, work, source, destination, o, cuda);
    } else if (packed || mixed) {
        for (const auto& segment : segments) {
            const auto begin  = static_cast<std::int32_t>(segment.token_offset);
            const auto length = static_cast<std::int32_t>(segment.tokens);
            Tensor source     = view_.states.recurrent_tensor(compact_layer, segment.source_offset);
            Tensor destination =
                view_.states.recurrent_tensor(compact_layer, segment.destination_offset);
            Tensor qs = q.slice(2, begin, length), ks = k.slice(2, begin, length);
            Tensor vs = v.slice(2, begin, length), os = o.slice(2, begin, length);
            ops::gated_delta_net(qs, ks, vs, g.slice(1, begin, length),
                                 beta.slice(1, begin, length), scale, true, work, source,
                                 destination, os, cuda);
        }
        if (mixed) {
            const auto& steps = mixed->steps;
            const auto begin  = static_cast<std::int32_t>(steps.token_offset);
            const auto count  = static_cast<std::int32_t>(steps.width * steps.rows);
            const auto w      = static_cast<std::int32_t>(steps.width);
            const auto b      = static_cast<std::int32_t>(steps.rows);
            const auto first  = static_cast<std::int32_t>(steps.row_offset);
            const auto source =
                view_.states.layer_view(compact_layer, inputs_.state_source_offsets).recurrent;
            Tensor qb = q.slice(2, begin, count).view({dk, qh, w, b});
            Tensor kb = k.slice(2, begin, count).view({dk, qh, w, b});
            Tensor vb = v.slice(2, begin, count).view({dv, vh, w, b});
            Tensor gb = g.slice(1, begin, count).view({vh, w, b});
            Tensor bb = beta.slice(1, begin, count).view({vh, w, b});
            Tensor ob = o.slice(2, begin, count).view({dv, vh, w, b});
            if (record_source) {
                const auto records = record_source->layer(compact_layer, b);
                ops::gated_delta_net_replay_record(
                    qb, kb, vb, gb, bb, scale, source, inputs_.valid_columns.slice(0, first, b),
                    {records.key, records.value, records.gate}, ob, cuda);
            } else {
                ops::gated_delta_net_batch_update(
                    qb, kb, vb, gb, bb, scale, true, source,
                    view_.states.layer_view(compact_layer, inputs_.state_destination_offsets)
                        .recurrent,
                    ob, cuda);
            }
        }
    } else {
        const auto source =
            view_.states.layer_view(compact_layer, inputs_.state_source_offsets).recurrent;
        Tensor qb = q.view({dk, qh, width, batch});
        Tensor kb = k.view({dk, qh, width, batch});
        Tensor vb = v.view({dv, vh, width, batch});
        Tensor gb = g.view({vh, width, batch});
        Tensor bb = beta.view({vh, width, batch});
        Tensor ob = o.view({dv, vh, width, batch});
        if (replay) {
            const auto records = replay->records.layer(compact_layer, batch);
            ops::gated_delta_net_replay_record(
                qb, kb, vb, gb, bb, scale, source, inputs_.valid_columns,
                {records.key, records.value, records.gate}, ob, cuda);
        } else {
            ops::gated_delta_net_batch_update(
                qb, kb, vb, gb, bb, scale, true, source,
                view_.states.layer_view(compact_layer, inputs_.state_destination_offsets).recurrent,
                ob, cuda);
        }
    }

    Tensor normalized = workspace::gdn_normalized_output(work, cfg, tokens).view({dv, vh, tokens});
    ops::silu_gated_rmsnorm(o, p.norm, z, cfg.rms_norm_eps, normalized, cuda);
    ops::linear_add(normalized.view({dimension(gdn.value_width()), tokens}), p.output.weight,
                    residual_, p.output.policy, work, cuda);
}

ops::MoeExecutionHints TextRun::next_projection_hints(std::uint32_t layer) const {
    const auto next = std::size_t(layer) + 1;
    return next < view_.parameters.text.layers.size()
               ? view_.parameters.text.layers[next].projection_prefetch
               : ops::MoeExecutionHints{};
}

void TextRun::mixer(std::uint32_t layer) {
    const auto& config  = view_.parameters.profile->config.text;
    const auto& block   = view_.parameters.text.layers.at(layer);
    const bool full     = config.layer_types[layer] == MixerKind::FullAttention;
    const bool sequence = is_sequence(update_);
    nvtx::ScopedRange range(
        full ? (sequence ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention)
             : (sequence ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn),
        full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
    auto scope         = view_.workspace.scope();
    const auto compact = config.compact_layer_indices[layer];
    if (full)
        attention(block, compact);
    else
        gdn(block, compact);
}

void TextRun::post_mixer(std::uint32_t layer, const moe::GpuCall* call) {
    const auto& config = view_.parameters.profile->config.text;
    const auto& block  = view_.parameters.text.layers.at(layer);
    nvtx::ScopedRange range(is_sequence(update_) ? nvtx::Name::PrefillPostMixer
                                                 : nvtx::Name::VerifyPostMixer,
                            nvtx::Category::PostMixer, layer);
    auto scope = view_.workspace.scope();
    Tensor h   = workspace::post_mixer_hidden(view_.workspace, config, residual_.ne[1]);
    ops::rmsnorm(residual_, block.post_attention_norm, config.rms_norm_eps, true, h,
                 view_.execution);
    if (const auto* dense = std::get_if<DenseParameters>(&block.ffn)) {
        if (call) throw std::invalid_argument("GPU MoE body supplied for a dense layer");
        dense_mlp(h, *dense, residual_, DenseMlpUse::Text, view_.workspace, view_.execution);
    } else if (call) {
        (*call)(h, residual_, view_.workspace, next_projection_hints(layer));
    } else {
        const auto& parameters = std::get<moe::Parameters>(block.ffn);
        moe::execute(h, parameters, parameters.resident, ops::MoeEpilogue::AddResidual, residual_,
                     view_.workspace, view_.execution, next_projection_hints(layer));
    }
}

void TextRun::capture_features(std::uint32_t layer) {
    if (!features_) return;
    const auto found = std::find(features_->layers.begin(), features_->layers.end(), layer);
    if (found == features_->layers.end()) return;
    const auto feature = static_cast<std::int32_t>(found - features_->layers.begin());
    Tensor target      = features_->features.slice(0, feature * residual_.ne[0], residual_.ne[0]);
    if (features_->residency_slots.data) {
        Tensor source = residual_.view({residual_.ne[0], inputs_.ids.ne[0], inputs_.ids.ne[1]});
        ops::scatter_bf16_batch(source, features_->residency_slots, inputs_.valid_columns, target,
                                view_.execution);
    } else {
        const auto bytes  = std::size_t(residual_.ne[0]) * dtype_size(DType::BF16);
        const auto extent = std::size_t(residual_.ne[1] - 1) * target.nb[1] + bytes;
        copy_rows_async({target.data, extent, target.device}, target.nb[1],
                        {residual_.data, residual_.bytes(), residual_.device}, residual_.nb[1],
                        bytes, residual_.ne[1], view_.execution);
    }
}

void TextRun::finish() {
    if (features_ && !features_->residency_slots.data && features_->positions.data) {
        const auto bytes = std::size_t(residual_.ne[1]) * sizeof(std::int32_t);
        copy_bytes_async({features_->positions.data, bytes, features_->positions.device},
                         {inputs_.cache_positions.data, bytes, inputs_.cache_positions.device},
                         view_.execution);
    }
    Tensor hidden          = hidden_.view({residual_.ne[0], residual_.ne[1]});
    const auto& parameters = view_.parameters;
    ops::rmsnorm(residual_, parameters.text.final_norm,
                 parameters.profile->config.text.rms_norm_eps, true, hidden, view_.execution);
}

bool TextRun::start() {
    if (stage_ != Stage::Idle) throw std::logic_error("Main traversal has already started");
    if (view_.moe) {
        auto activity = view_.moe->begin(residual_.ne[1]);
        if (activity) activity_.emplace(std::move(*activity));
    }
    stage_ = Stage::Layer;
    prefix();
    return poll();
}

void TextRun::complete_layer() {
    capture_features(layer_);
    post_mixer_scope_.reset();
    ++layer_;
    stage_ = Stage::Layer;
}

bool TextRun::poll() {
    if (stage_ == Stage::Done) return true;
    if (stage_ != Stage::Layer && stage_ != Stage::PostMixer)
        throw std::logic_error("Main traversal is not active");
    if (stage_ == Stage::PostMixer) {
        if (!view_.moe->poll()) return false;
        complete_layer();
    }
    const auto& parameters = view_.parameters;
    const auto& config     = parameters.profile->config.text;
    while (layer_ < parameters.text.layers.size()) {
        const auto& block  = parameters.text.layers[layer_];
        const auto* routed = std::get_if<moe::Parameters>(&block.ffn);
        if (view_.moe && routed && !view_.moe->try_before_layer(routed->bank)) return false;
        mixer(layer_);
        if (view_.moe && routed) {
            post_mixer_scope_.emplace(view_.workspace.scope());
            post_mixer_hidden_ =
                workspace::post_mixer_hidden(view_.workspace, config, residual_.ne[1]);
            ops::rmsnorm(residual_, block.post_attention_norm, config.rms_norm_eps, true,
                         post_mixer_hidden_, view_.execution);
            stage_ = Stage::PostMixer;
            if (!view_.moe->start(routed->bank, post_mixer_hidden_, residual_, view_.workspace,
                                  next_projection_hints(layer_)))
                return false;
        } else {
            post_mixer(layer_);
        }
        complete_layer();
    }
    finish();
    activity_.reset();
    stage_ = Stage::Done;
    return true;
}

void TextRun::cancel() {
    if (stage_ == Stage::Done || stage_ == Stage::Cancelled) return;
    if (view_.moe && activity_) view_.moe->cancel();
    post_mixer_scope_.reset();
    activity_.reset();
    stage_ = Stage::Cancelled;
}

} // namespace ninfer::models::qwen3_5::execution
