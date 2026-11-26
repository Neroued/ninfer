#include "models/inkling2/execution/relative_attention.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_split.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace ninfer::models::inkling2::execution {
namespace {

constexpr auto swa_bucket = ops::ContextBucket::K2;

struct AttentionShape {
    std::int32_t tokens;
    std::int32_t rows;
    bool ragged = false;

    Tensor heads(const Tensor& t, std::int32_t dim, std::int32_t count) const {
        if (ragged) return t.view({dim, count, tokens});
        return t.view({dim, count, tokens, rows});
    }

    Tensor columns(const Tensor& t) const {
        if (ragged) return t.view({t.ne[0], tokens});
        return t.view({t.ne[0], tokens, rows});
    }
};

Tensor input_stream(const Tensor& input, int stream) {
    return input.slice(2, stream, 1).view({input.ne[0], input.ne[1]});
}

void project_attention(const Tensor& input, const AttentionPassParameters& pass,
                       workspace::AttentionBuffers& buffers, MemoryArena& arena,
                       CudaExecutionView cuda) {
    const auto q = input_stream(input, 0);
    if (pass.qkv_split) {
        const std::array outputs{buffers.k, buffers.v, buffers.q, buffers.gate, buffers.r};
        ops::linear_split(q, *pass.qkv_split, outputs, arena, cuda);
        return;
    }
    const auto k = input_stream(input, 1);
    const auto v = input_stream(input, 2);
    const std::array outputs{buffers.q, buffers.gate, buffers.r};
    ops::linear_split(q, pass.qrg_split, outputs, arena, cuda);
    ops::linear_pair(k, v, pass.key.weight, pass.value.weight, buffers.k, buffers.v, cuda);
    if (pass.k_conv_proj)
        ops::linear_pair(k, v, pass.k_conv_proj->weight, pass.v_conv_proj->weight, buffers.k_dyn,
                         buffers.v_dyn, cuda);
}

ops::RelativeAttentionPrepareParams prepare_params(const TextConfig& config, bool is_swa) {
    return {
        .qk_clip_threshold  = config.qk_l2_clip_threshold,
        .rel_clip_threshold = config.rel_l2_clip_threshold,
        .clip_eps           = 1e-6F,
        .v_norm_eps         = config.norm_epsilon,
        .use_log_scaling =
            !is_swa && config.log_scaling_n_floor.has_value() && config.log_scaling_alpha > 0.0F,
        .log_scaling_floor =
            config.log_scaling_n_floor ? dimension(*config.log_scaling_n_floor) : 8192,
        .log_scaling_alpha = config.log_scaling_alpha,
    };
}

ops::RelativeAttentionPrepareInputs
prepare_inputs(const TextConfig& config, const PhysicalLayerParameters& layer,
               ops::AttentionHeadGeometry geom, AttentionShape shape, const Tensor& input,
               const Tensor& positions, const workspace::AttentionBuffers& buffers) {
    const int taps = dimension(config.sconv_kernel_size);
    return {
        .q             = shape.heads(buffers.q, geom.head_dim, geom.query_heads),
        .r             = shape.heads(buffers.r, dimension(config.d_rel), geom.query_heads),
        .k             = shape.heads(buffers.k, geom.head_dim, geom.kv_heads),
        .v             = shape.heads(buffers.v, geom.head_dim, geom.kv_heads),
        .q_scale       = layer.q_scale,
        .r_scale       = layer.r_scale,
        .k_conv_weight = layer.k_conv,
        .v_conv_weight = layer.v_conv,
        .k_dyn = buffers.k_dyn.data ? shape.heads(buffers.k_dyn, taps, geom.kv_heads) : Tensor{},
        .v_dyn = buffers.v_dyn.data ? shape.heads(buffers.v_dyn, taps, geom.kv_heads) : Tensor{},
        .positions   = positions,
        .sink_x      = shape.columns(input_stream(input, 0)),
        .sink_weight = layer.sink,
    };
}

ops::RelativeAttentionQueryInputs query_inputs(const TextConfig& config,
                                               const PhysicalLayerParameters& layer,
                                               ops::AttentionHeadGeometry geom,
                                               AttentionShape shape, const Tensor& positions,
                                               const workspace::AttentionBuffers& buffers) {
    return {
        .q               = shape.heads(buffers.q_out, geom.head_dim, geom.query_heads),
        .r               = shape.heads(buffers.r_out, dimension(config.d_rel), geom.query_heads),
        .rel_logits_proj = layer.rel_logits_proj,
        .gate            = shape.heads(buffers.gate, geom.head_dim, geom.query_heads),
        .sink            = shape.columns(buffers.sink),
        .positions       = positions,
    };
}

float attention_scale(ops::AttentionHeadGeometry geometry) {
    return 1.0F / std::sqrt(static_cast<float>(geometry.head_dim));
}

} // namespace

ops::OpGraphProfile FullAttentionReadout::profile(ops::ContextBucket bucket,
                                                  std::int32_t sms) const {
    return ops::relative_softmax_attention_full_graph_profile(
        geometry, cache.storage, bucket, inputs.q.ne[3], inputs.q.ne[2], sms,
        /*masked=*/static_cast<bool>(valid_columns.data));
}

std::size_t FullAttentionReadout::scratch_bytes(ops::ContextBucketRange buckets,
                                                std::int32_t sms) const {
    const int width = inputs.q.ne[2], rows = inputs.q.ne[3];
    return ops::relative_softmax_attention_full_workspace_capacity_bytes(
        geometry, cache.storage, buckets, rows, width, width, sms);
}

void FullAttentionReadout::execute(ops::ContextBucket bucket, CudaExecutionView cuda) const {
    MemoryArena workspace(scratch);
    auto output = out;
    ops::relative_softmax_attention(inputs, valid_columns, kv_table_rows, geometry, scale, cache,
                                    bucket, workspace, output, cuda);
}

RelativeAttentionRun::RelativeAttentionRun(const Parameters& p, const LogicalLayerConfig& logical,
                                           const BatchView& batch, AttentionExecution execution,
                                           const Tensor& input, Tensor output, MemoryArena& arena,
                                           CudaExecutionView cuda, const RecordWindow* records)
    : config_(p.profile->config.text), logical_(logical),
      layer_(p.text.layers.at(logical.physical_layer)),
      pass_(layer_.attention_passes.at(logical.pass_index)),
      geometry_(p.profile->text.layers.at(logical.physical_layer).attention_geometry),
      batch_(batch), states_(execution.states), input_(input), output_(output),
      full_cache_(logical.is_swa ? PagedKVBatchLayerView{}
                                 : execution.cache.layer(logical.attention_state_index)),
      swa_cache_(logical.is_swa
                     ? states_.swa_layer(logical.attention_state_index, batch.destination_offsets)
                     : CyclicKVCacheLayerView{}),
      arena_(arena), cuda_(cuda), scope_(arena.scope()),
      buffers_(workspace::attention(arena, config_, logical.is_swa, swa_cache_.storage,
                                    batch.tokens, batch.rows, /*ragged=*/false,
                                    /*need_chunk_kv=*/batch.tokens > 1 && !records)),
      query_(query_inputs(config_, layer_, geometry_, {batch.tokens, batch.rows}, batch.positions,
                          buffers_)),
      attended_(
          buffers_.out.view({geometry_.head_dim, geometry_.query_heads, batch.tokens, batch.rows})),
      records_(records) {
    if (records_ && logical_.is_swa)
        buffers_.chunk_kv = records_->layers.at(logical_.logical_index).swa_chunk;
}

void RelativeAttentionRun::project() { project_attention(input_, pass_, buffers_, arena_, cuda_); }

void RelativeAttentionRun::prepare() {
    const bool swa    = logical_.is_swa;
    const auto slot   = logical_.attention_state_index;
    const auto inputs = prepare_inputs(config_, layer_, geometry_, {batch_.tokens, batch_.rows},
                                       input_, batch_.positions, buffers_);
    const ops::RelativeAttentionPrepareOutputs outputs{query_.q, query_.r, query_.sink};
    const auto params = prepare_params(config_, swa);
    const auto k_in   = states_.k_conv(swa, slot, batch_.source_offsets);
    const auto k_out  = states_.k_conv(swa, slot, batch_.destination_offsets);
    const auto v_in   = states_.v_conv(swa, slot, batch_.source_offsets);
    const auto v_out  = states_.v_conv(swa, slot, batch_.destination_offsets);

    if (records_) {
        auto record = records_->layers.at(logical_.logical_index);
        if (swa)
            ops::relative_attention_prepare_snapshot(
                inputs, batch_.valid_columns, records_->base_slots, k_in, record.k_conv, v_in,
                record.v_conv, record.swa_chunk, params, outputs, cuda_);
        else
            // Full attention writes a provisional paged suffix. The Session publishes only
            // the adopted history frontier, so the rejected suffix is never read as history.
            ops::relative_attention_prepare_snapshot(
                inputs, batch_.valid_columns, batch_.main_tables, records_->base_slots, k_in,
                record.k_conv, v_in, record.v_conv, full_cache_, params, outputs, cuda_);
    } else if (!swa) {
        ops::relative_attention_prepare(inputs, batch_.valid_columns, batch_.main_tables, k_in,
                                        k_out, v_in, v_out, full_cache_, params, outputs, cuda_);
    } else if (batch_.tokens == 1) {
        ops::relative_attention_prepare(inputs, batch_.valid_columns, k_in, k_out, v_in, v_out,
                                        swa_cache_, params, outputs, cuda_);
    } else {
        ops::relative_attention_prepare(inputs, batch_.valid_columns, k_in, k_out, v_in, v_out,
                                        buffers_.chunk_kv, params, outputs, cuda_);
    }
}

FullAttentionReadout RelativeAttentionRun::readout(ops::ContextBucketRange buckets) {
    FullAttentionReadout r{
        .geometry      = geometry_,
        .scale         = attention_scale(geometry_),
        .inputs        = query_,
        .valid_columns = batch_.valid_columns,
        .kv_table_rows = batch_.main_tables,
        .cache         = full_cache_,
        .scratch       = {},
        .out           = attended_,
    };
    auto scope       = arena_.scope();
    const auto bytes = r.scratch_bytes(buckets, cuda_.multiprocessor_count());
    r.scratch = bytes ? arena_.alloc_bytes(bytes, 256) : MemorySpan{nullptr, 0, cuda_.device()};
    return r;
}

void RelativeAttentionRun::attend_swa() {
    ops::relative_softmax_attention(query_, batch_.valid_columns, geometry_, config_.sliding_window,
                                    attention_scale(geometry_), swa_cache_, buffers_.chunk_kv,
                                    swa_bucket, arena_, attended_, cuda_);
}

void RelativeAttentionRun::finish() {
    ops::linear(buffers_.out, pass_.output.weight, output_, pass_.output.policy, arena_, cuda_);
}

void RelativeAttentionRun::execute(const FullAttentionSubmit& submit) {
    project();
    prepare();
    if (logical_.is_swa) {
        attend_swa();
        if (batch_.tokens > 1 && !records_)
            ops::relative_attention_commit_cyclic_kv(buffers_.chunk_kv, batch_.positions,
                                                     batch_.valid_columns, swa_cache_, cuda_);
    } else if (submit) {
        const auto& records = full_cache_.k_pages.records;
        const auto capacity = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            config_.max_context, std::uint64_t(records.page_offsets.ne[0]) * records.page_tokens));
        submit(readout(ops::ContextBucketRange::through(capacity)));
    } else {
        const auto bucket = ops::select_context_bucket(batch_.max_visible_tokens);
        readout({bucket, bucket}).execute(bucket, cuda_);
    }
    finish();
}

void relative_attention_ragged(const Parameters& p, const LogicalLayerConfig& logical,
                               const BatchView& batch, const ops::RaggedSequenceView& ragged,
                               AttentionExecution execution, const Tensor& input, Tensor& output,
                               MemoryArena& arena, CudaExecutionView cuda) {
    const auto& config   = p.profile->config.text;
    const auto& layer    = p.text.layers.at(logical.physical_layer);
    const auto& pass     = layer.attention_passes.at(logical.pass_index);
    const auto geom      = p.profile->text.layers.at(logical.physical_layer).attention_geometry;
    const auto slot      = logical.attention_state_index;
    const bool swa       = logical.is_swa;
    const int tokens     = ragged.total_tokens();
    const auto swa_cache = swa ? execution.states.swa_layer(slot, batch.destination_offsets)
                               : CyclicKVCacheLayerView{};
    const AttentionShape shape{tokens, 1, /*ragged=*/true};
    auto scope   = arena.scope();
    auto buffers = workspace::attention(arena, config, swa, swa_cache.storage, tokens, 1,
                                        /*ragged=*/true, /*need_chunk_kv=*/true);
    project_attention(input, pass, buffers, arena, cuda);

    const auto inputs = prepare_inputs(config, layer, geom, shape, input, batch.positions, buffers);
    const auto query  = query_inputs(config, layer, geom, shape, batch.positions, buffers);
    const ops::RelativeAttentionPrepareOutputs outputs{query.q, query.r, query.sink};
    const auto params = prepare_params(config, swa);
    const auto k_in   = execution.states.k_conv(swa, slot, batch.source_offsets);
    const auto k_out  = execution.states.k_conv(swa, slot, batch.destination_offsets);
    const auto v_in   = execution.states.v_conv(swa, slot, batch.source_offsets);
    const auto v_out  = execution.states.v_conv(swa, slot, batch.destination_offsets);
    auto attended     = shape.heads(buffers.out, geom.head_dim, geom.query_heads);
    const float scale = attention_scale(geom);
    if (swa) {
        ops::relative_attention_prepare(inputs, ragged, k_in, k_out, v_in, v_out, buffers.chunk_kv,
                                        params, outputs, cuda);
        ops::relative_softmax_attention(query, ragged, geom, config.sliding_window, scale,
                                        swa_cache, buffers.chunk_kv, swa_bucket, arena, attended,
                                        cuda);
        ops::relative_attention_commit_cyclic_kv(buffers.chunk_kv, ragged, batch.positions,
                                                 swa_cache, cuda);
    } else {
        const auto cache  = execution.cache.layer(slot);
        const auto bucket = ops::select_context_bucket(batch.max_visible_tokens);
        ops::relative_attention_prepare(inputs, ragged, batch.main_tables, k_in, k_out, v_in, v_out,
                                        cache, params, outputs, cuda);
        ops::relative_softmax_attention(query, ragged, batch.main_tables, geom, scale, cache,
                                        bucket, arena, attended, cuda);
    }
    ops::linear(buffers.out, pass.output.weight, output, pass.output.policy, arena, cuda);
}

} // namespace ninfer::models::inkling2::execution
