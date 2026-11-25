#include "models/qwen3_5/execution/draft.h"
#include "models/qwen3_5/execution/linear.h"
#include <cmath>
#include "models/qwen3_5/execution/workspace.h"

#include "core/nvtx.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sequence.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

void append_context_impl(DraftExecutionView& execution, const Tensor& features,
                         const Tensor& positions, const Tensor& commit_counts, const Tensor& lanes,
                         const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    {
        const auto& target         = execution.parameters.profile->config.text;
        const auto& config         = *execution.parameters.profile->config.draft;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        nvtx::ScopedRange append_range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash,
                                       static_cast<std::uint64_t>(columns));
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] !=
                dimension(target.hidden_size * std::uint64_t(config.target_layer_ids.size())) ||
            features.ne[3] != 1 || positions.dtype != DType::I32 || positions.ne[0] != width ||
            positions.ne[1] != batch || commit_counts.dtype != DType::I32 ||
            commit_counts.ne[0] != batch || lanes.dtype != DType::I64 || lanes.ne[0] != batch ||
            table_rows.dtype != DType::I32 || table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window =
            batch == 1 && width > dimension(config.sliding_window.value_or(0));
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset =
            replace_local_window ? width - dimension(config.sliding_window.value_or(0)) : 0;
        const int local_width =
            replace_local_window ? dimension(config.sliding_window.value_or(0)) : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window
                ? static_cast<std::uint32_t>(dimension(config.sliding_window.value_or(0)))
                : envelope.min_count,
            replace_local_window
                ? static_cast<std::uint32_t>(dimension(config.sliding_window.value_or(0)))
                : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            local_counts = execution.work.alloc(DType::I32, {1});
            ops::set_i32_scalar(local_counts, dimension(config.sliding_window.value_or(0)),
                                execution.cuda);
        }

        const int projected_width = config.dflash2.has_value() ? local_width : width;
        const Tensor input        = config.dflash2.has_value() && replace_local_window
                                        ? features.slice(1, local_offset, local_width)
                                        : features;
        const auto roots =
            workspace::dflash_context(execution.work, target, config, projected_width * batch);
        Tensor projected = roots.projected;
        Tensor context   = roots.normalized;
        project(input.view(
                    {dimension(target.hidden_size * std::uint64_t(config.target_layer_ids.size())),
                     projected_width * batch}),
                execution.parameters.draft->feature_projection, projected, execution.work,
                execution.cuda);
        ops::rmsnorm(projected, execution.parameters.draft->context_norm, config.rms_norm_eps,
                     false, context, execution.cuda);

        if (config.dflash2) {
            std::array<ops::ContextKVMaterializeLayerView, ops::kContextKVMaterializeLayers> layers;
            if (config.num_hidden_layers != layers.size() ||
                config.local_layer_count() != layers.size()) {
                throw std::invalid_argument(
                    "masked draft context: config is outside the fused materialization contract");
            }
            for (int layer = 0; layer < dimension(config.num_hidden_layers); ++layer) {
                const auto& weights = execution.parameters.draft->layers[layer];
                layers[layer]       = {weights.context_key.weight, weights.context_value.weight,
                                       weights.key_norm, execution.local_layer(layer, lanes)};
            }
            const Tensor local_positions =
                replace_local_window ? positions.slice(0, local_offset, local_width) : positions;
            ops::context_kv_materialize(
                context.view({dimension(target.hidden_size), local_width, batch}), local_positions,
                local_counts, layers, *execution.parameters.profile->draft->context_materialization,
                {local_envelope.min_count, local_envelope.max_count}, execution.work,
                execution.cuda);
        } else {
            for (int layer = 0; layer < dimension(config.num_hidden_layers); ++layer) {
                auto layer_scope = execution.work.scope();
                const auto& weight =
                    execution.parameters.draft->layers.at(static_cast<std::size_t>(layer));
                const bool local_layer =
                    config.layer_types[layer] == dflash::AttentionKind::SlidingAttention;
                const int layer_width   = local_layer ? local_width : width;
                const int layer_columns = layer_width * batch;
                Tensor layer_context    = local_layer && replace_local_window
                                              ? context.slice(1, local_offset, local_width)
                                              : context;
                Tensor layer_positions  = local_layer && replace_local_window
                                              ? positions.slice(0, local_offset, local_width)
                                              : positions;
                auto layer_roots =
                    workspace::dflash_context_layer(execution.work, target, config, layer_columns);
                Tensor key_raw = layer_roots.key_raw.view(
                    {dimension(config.attention.head_dim),
                     dimension(config.attention.num_key_value_heads), layer_columns});
                Tensor value = layer_roots.value.view(
                    {dimension(config.attention.head_dim),
                     dimension(config.attention.num_key_value_heads), layer_columns});
                Tensor key_flat =
                    key_raw.view({dimension(config.attention.key_width()), layer_columns});
                Tensor value_flat =
                    value.view({dimension(config.attention.key_width()), layer_columns});
                ops::linear_pair(layer_context, layer_context, weight.context_key.weight,
                                 weight.context_value.weight, key_flat, value_flat, execution.cuda);
                Tensor key = layer_roots.key.view({dimension(config.attention.head_dim),
                                                   dimension(config.attention.num_key_value_heads),
                                                   layer_columns});
                ops::rmsnorm(key_raw, weight.key_norm, config.rms_norm_eps, false, key,
                             execution.cuda);
                ops::rope(layer_positions.view({layer_columns}),
                          dimension(config.attention.head_dim), config.rope_theta, key,
                          execution.cuda);
                Tensor key_batch =
                    key.view({dimension(config.attention.head_dim),
                              dimension(config.attention.num_key_value_heads), layer_width, batch});
                Tensor value_batch    = value.view({dimension(config.attention.head_dim),
                                                    dimension(config.attention.num_key_value_heads),
                                                    layer_width, batch});
                Tensor position_batch = layer_positions.view({layer_width, batch});
                if (local_layer) {
                    ops::kv_cache_append_prefix(
                        key_batch, value_batch, position_batch, local_counts, local_envelope,
                        execution.local_layer(config.compact_layer_index(layer), lanes),
                        execution.cuda);
                } else {
                    ops::kv_cache_append_prefix(
                        key_batch, value_batch, position_batch, commit_counts, table_rows, envelope,
                        execution.full_layer(config.compact_layer_index(layer)), execution.cuda);
                }
            }
        }
    }
}

void prepare_dynamic_branch(DraftExecutionView& execution, const Tensor& residual,
                            const Tensor& norm, float eps, const DynamicConvParameters& weights,
                            workspace::DFlash2BranchRoots& branch) {
    auto scope      = execution.work.scope();
    const int width = residual.ne[1], batch = residual.ne[2];
    MemoryArena scratch(execution.work.alloc_bytes(
        ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(residual.ne[0], width,
                                                                           width, batch, batch)));
    ops::rmsnorm_dynamic_grouped_conv_prepare(residual, norm, eps, weights.base_kernel,
                                              weights.kernel_projection.weight, branch.prepared,
                                              branch.finish_delta, scratch, execution.cuda);
}

void finish_dynamic_branch(DraftExecutionView& execution, const Tensor& input,
                           const LinearParameters& projection, const DynamicConvParameters& weights,
                           const Tensor& finish_delta, Tensor& residual) {
    auto scope      = execution.work.scope();
    const int width = input.ne[1], batch = input.ne[2];
    MemoryArena scratch(
        execution.work.alloc_bytes(ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
            input.ne[0], width, width, batch, batch)));
    ops::linear_dynamic_grouped_conv_add(input, projection.weight, weights.base_kernel,
                                         finish_delta, residual, scratch, execution.cuda);
}

template <class Readout>
void submit_attention(Readout readout, ops::ContextBucket bucket, ops::ContextBucketRange buckets,
                      const std::function<void(const Readout&)>& submit,
                      DraftExecutionView& execution) {
    if (!submit) {
        readout.execute(bucket, execution.work, execution.cuda);
        return;
    }
    auto scope       = execution.work.scope();
    const auto bytes = readout.scratch_bytes(buckets);
    readout.scratch =
        bytes ? execution.work.alloc_bytes(bytes) : MemorySpan{nullptr, 0, execution.cuda.device()};
    submit(readout);
}

struct DraftAttentionTensors {
    Tensor query, key, value, output;
};

DraftAttentionTensors dflash2_attention(MemoryArena& work, const dflash::AttentionConfig& config,
                                        int width, int batch) {
    const auto dim   = dimension(config.head_dim);
    const auto heads = dimension(config.num_attention_heads);
    const auto kv    = dimension(config.num_key_value_heads);
    return {work.alloc(DType::BF16, {dim, heads, width, batch}),
            work.alloc(DType::BF16, {dim, kv, width, batch}),
            work.alloc(DType::BF16, {dim, kv, width, batch}),
            work.alloc(DType::BF16, {dim, heads, width, batch})};
}

DraftAttentionTensors dflash_attention(const workspace::DFlashAttentionRoots& roots,
                                       const dflash::AttentionConfig& config, int width,
                                       int batch) {
    const auto dim   = dimension(config.head_dim);
    const auto heads = dimension(config.num_attention_heads);
    const auto kv    = dimension(config.num_key_value_heads);
    return {roots.query.view({dim, heads, width, batch}), roots.key.view({dim, kv, width, batch}),
            roots.value.view({dim, kv, width, batch}),
            roots.attention.view({dim, heads, width, batch})};
}

void submit_layer_attention(DraftExecutionView& execution, const DraftInputs& inputs,
                            const Tensor& positions, const DraftAttentionTensors& tensors,
                            std::uint32_t layer, DraftBuckets buckets,
                            const DraftReadoutSubmit& submit) {
    const auto& config = *execution.parameters.profile->config.draft;
    const ops::AttentionHeadGeometry geometry{dimension(config.attention.head_dim),
                                              dimension(config.attention.num_attention_heads),
                                              dimension(config.attention.num_key_value_heads)};
    const auto scale   = static_cast<float>(1.0 / std::sqrt(double(geometry.head_dim)));
    const auto compact = config.compact_layer_index(layer);
    if (config.layer_types[layer] == dflash::AttentionKind::SlidingAttention) {
        const SlidingWindowReadout readout{tensors.query,
                                           tensors.key,
                                           tensors.value,
                                           positions,
                                           inputs.valid_columns,
                                           tensors.output,
                                           execution.local_layer(compact, inputs.state_offsets),
                                           geometry,
                                           scale,
                                           {}};
        submit_attention(readout, buckets.local, submit.buckets, submit.local, execution);
    } else {
        const ContextAttentionReadout readout{tensors.query,
                                              tensors.key,
                                              tensors.value,
                                              inputs.frontiers,
                                              inputs.valid_columns,
                                              inputs.full_kv_rows,
                                              tensors.output,
                                              execution.full_layer(compact),
                                              geometry,
                                              scale,
                                              {}};
        submit_attention(readout, buckets.full, submit.buckets, submit.full, execution);
    }
}

void propose_dflash2_batch(DraftExecutionView& execution, const DraftInputs& inputs,
                           DraftOutputs& outputs, int batch, int width, int g, DraftBuckets buckets,
                           const DraftReadoutSubmit& submit) {
    if (execution.parameters.profile->config.draft->dflash2) {
        const auto& target             = execution.parameters.profile->config.text;
        const auto& config             = *execution.parameters.profile->config.draft;
        const int columns              = width * batch;
        const int mask_columns         = g * batch;
        auto& work                     = execution.work;
        const auto cuda                = execution.cuda;
        const DraftParameters& weights = *execution.parameters.draft;
        Tensor anchors                 = inputs.anchors;
        Tensor frontiers               = inputs.frontiers;
        Tensor valid_columns           = inputs.valid_columns;
        Tensor ids                     = outputs.ids;
        Tensor positions               = outputs.positions;
        ops::prepare_masked_block(anchors, frontiers, valid_columns,
                                  dimension(config.mask_token_id), ids, positions, cuda);
        Tensor residual = work.alloc(DType::BF16, {dimension(target.hidden_size), width, batch});
        Tensor flat_residual = residual.view({dimension(target.hidden_size), columns});
        ops::embedding(ids.view({columns}), execution.parameters.text.token_embedding,
                       flat_residual, cuda);
        for (std::size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
            const auto& layer = weights.layers[layer_index];
            nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                          layer_index);
            {
                auto scope  = work.scope();
                auto branch = workspace::dflash2_branch(work, target, config, width, batch);
                prepare_dynamic_branch(execution, residual, layer.input_norm, config.rms_norm_eps,
                                       *layer.attention_conv, branch);
                auto tensors = dflash2_attention(work, config.attention, width, batch);
                Tensor query_flat =
                    tensors.query.view({dimension(config.attention.query_width()), columns});
                Tensor key_flat =
                    tensors.key.view({dimension(config.attention.key_width()), columns});
                Tensor value_flat =
                    tensors.value.view({dimension(config.attention.key_width()), columns});
                std::array outputs{query_flat, key_flat, value_flat};
                ops::linear_split(branch.prepared.view({dimension(target.hidden_size), columns}),
                                  layer.query_key_value.weight, outputs, cuda);
                ops::rmsnorm_rope(positions, layer.query_norm, layer.key_norm, config.rms_norm_eps,
                                  config.rope_theta, tensors.query, tensors.key, cuda);
                submit_layer_attention(execution, inputs, positions, tensors, layer_index, buckets,
                                       submit);
                finish_dynamic_branch(
                    execution,
                    tensors.output.view({dimension(config.attention.query_width()), width, batch}),
                    layer.output, *layer.attention_conv, branch.finish_delta, residual);
            }
            {
                auto scope  = work.scope();
                auto branch = workspace::dflash2_branch(work, target, config, width, batch);
                prepare_dynamic_branch(execution, residual, layer.post_attention_norm,
                                       config.rms_norm_eps, *layer.mlp_conv, branch);
                Tensor intermediate =
                    work.alloc(DType::BF16, {dimension(config.intermediate_size), width, batch});
                Tensor intermediate_flat =
                    intermediate.view({dimension(config.intermediate_size), columns});
                project_swiglu(branch.prepared.view({dimension(target.hidden_size), columns}),
                               layer.mlp.gate_up, intermediate_flat, work, cuda);
                finish_dynamic_branch(execution, intermediate, layer.mlp.down, *layer.mlp_conv,
                                      branch.finish_delta, residual);
            }
        }
        Tensor hidden = work.alloc(DType::BF16, {dimension(target.hidden_size), mask_columns});
        ops::rmsnorm_pack_tail(residual, weights.final_norm, config.rms_norm_eps, hidden, cuda);
        Tensor candidates = outputs.candidate_ids;
        Tensor ids_flat =
            candidates.view({dimension(config.dflash2->selector_top_k), mask_columns});
        Tensor scores =
            work.alloc(DType::FP32, {dimension(config.dflash2->selector_top_k), mask_columns});
        if (execution.proposal_head == ProposalHead::Full) {
            ops::linear_topk(hidden, execution.parameters.draft->output_head.weight,
                             dimension(execution.parameters.profile->public_tokens), ids_flat,
                             scores, work, cuda);
        } else {
            const auto& head = *execution.parameters.proposal;
            if (head.token_ids) {
                ops::linear_topk(hidden, head.head.weight, *head.token_ids, ids_flat, scores, work,
                                 cuda);
            } else {
                ops::linear_topk(hidden, head.head.weight,
                                 dimension(execution.parameters.profile->public_tokens), ids_flat,
                                 scores, work, cuda);
            }
        }
        Tensor projected =
            work.alloc(DType::BF16, {dimension(config.dflash2->selector_rank), mask_columns});
        project(hidden, weights.selector->hidden_projection, projected, work, cuda);
        Tensor drafts     = outputs.drafts;
        Tensor proposal_q = outputs.proposal_q;
        ops::candidate_selector_path(
            candidates, scores.view({dimension(config.dflash2->selector_top_k), g, batch}),
            projected.view({dimension(config.dflash2->selector_rank), g, batch}), anchors,
            weights.selector->predecessor_codebook, weights.selector->successor_codebook,
            dimension(execution.parameters.profile->public_tokens), frontiers, inputs.sampling,
            drafts, proposal_q, outputs.confidences, inputs.live_rows, work, cuda);
    }
}

void propose_batch_impl(DraftExecutionView& execution, const DraftInputs& inputs,
                        DraftOutputs& outputs, std::int32_t batch_size, int width, int g,
                        DraftBuckets buckets, const DraftReadoutSubmit& submit) {
    if (execution.parameters.profile->config.draft->dflash2) {
        nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                         static_cast<std::uint64_t>(width) * batch_size);
        propose_dflash2_batch(execution, inputs, outputs, batch_size, width, g, buckets, submit);
    } else {
        const auto& target         = execution.parameters.profile->config.text;
        const auto& config         = *execution.parameters.profile->config.draft;
        const std::int32_t columns = width * batch_size;
        nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                         static_cast<std::uint64_t>(columns));
        Tensor anchors       = inputs.anchors;
        Tensor frontiers     = inputs.frontiers;
        Tensor valid_columns = inputs.valid_columns;
        Tensor ids           = outputs.ids;
        Tensor positions     = outputs.positions;
        Tensor drafts        = outputs.drafts;

        ops::prepare_masked_block(anchors, frontiers, valid_columns,
                                  dimension(config.mask_token_id), ids, positions, execution.cuda);
        Tensor residual =
            execution.work.alloc(DType::BF16, {dimension(target.hidden_size), columns});
        ops::embedding(ids.view({columns}), execution.parameters.text.token_embedding, residual,
                       execution.cuda);

        for (int layer = 0; layer < dimension(config.num_hidden_layers); ++layer) {
            nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                          static_cast<std::uint64_t>(layer));
            const auto& weight =
                execution.parameters.draft->layers.at(static_cast<std::size_t>(layer));
            {
                nvtx::ScopedRange attention_range(nvtx::Name::DFlashAttention,
                                                  nvtx::Category::Attention,
                                                  static_cast<std::uint64_t>(layer));
                auto attention_scope = execution.work.scope();
                auto roots = workspace::dflash_attention(execution.work, target, config, columns);
                ops::rmsnorm(residual, weight.input_norm, config.rms_norm_eps, false, roots.hidden,
                             execution.cuda);
                Tensor query_raw = roots.query_raw.view(
                    {dimension(config.attention.head_dim),
                     dimension(config.attention.num_attention_heads), columns});
                Tensor key_raw =
                    roots.key_raw.view({dimension(config.attention.head_dim),
                                        dimension(config.attention.num_key_value_heads), columns});
                Tensor value =
                    roots.value.view({dimension(config.attention.head_dim),
                                      dimension(config.attention.num_key_value_heads), columns});
                Tensor query_flat =
                    query_raw.view({dimension(config.attention.query_width()), columns});
                Tensor key_flat = key_raw.view({dimension(config.attention.key_width()), columns});
                Tensor value_flat = value.view({dimension(config.attention.key_width()), columns});
                std::array outputs{query_flat, key_flat, value_flat};
                ops::linear_split(roots.hidden, weight.query_key_value.weight, outputs,
                                  execution.cuda);
                Tensor query =
                    roots.query.view({dimension(config.attention.head_dim),
                                      dimension(config.attention.num_attention_heads), columns});
                Tensor key =
                    roots.key.view({dimension(config.attention.head_dim),
                                    dimension(config.attention.num_key_value_heads), columns});
                ops::rmsnorm(query_raw, weight.query_norm, config.rms_norm_eps, false, query,
                             execution.cuda);
                ops::rmsnorm(key_raw, weight.key_norm, config.rms_norm_eps, false, key,
                             execution.cuda);
                ops::rope(positions.view({columns}), dimension(config.attention.head_dim),
                          config.rope_theta, query, key, execution.cuda);
                submit_layer_attention(execution, inputs, positions,
                                       dflash_attention(roots, config.attention, width, batch_size),
                                       layer, buckets, submit);
                project_add(
                    roots.attention.view({dimension(config.attention.query_width()), columns}),
                    weight.output, residual, execution.work, execution.cuda);
            }
            {
                nvtx::ScopedRange mlp_range(nvtx::Name::DFlashMlp, nvtx::Category::PostMixer,
                                            static_cast<std::uint64_t>(layer));
                auto mlp_scope = execution.work.scope();
                auto roots     = workspace::dflash_mlp(execution.work, target, config, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, config.rms_norm_eps, false,
                             roots.hidden, execution.cuda);
                project_swiglu(roots.hidden, weight.mlp.gate_up, roots.intermediate, execution.work,
                               execution.cuda);
                project_add(roots.intermediate, weight.mlp.down, residual, execution.work,
                            execution.cuda);
            }
        }

        Tensor packed =
            execution.work.alloc(DType::BF16, {dimension(target.hidden_size), g * batch_size});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes     = static_cast<std::size_t>(dimension(target.hidden_size)) *
                                      static_cast<std::size_t>(g) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(dimension(target.hidden_size)) * width * element_bytes;
        const auto* source =
            static_cast<const std::byte*>(residual.data) +
            static_cast<std::size_t>(dimension(target.hidden_size)) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     execution.cuda.stream()));
        Tensor proposal_hidden =
            execution.work.alloc(DType::BF16, {dimension(target.hidden_size), g * batch_size});
        ops::rmsnorm(packed, execution.parameters.draft->final_norm, config.rms_norm_eps, false,
                     proposal_hidden, execution.cuda);
        Tensor flat_drafts = drafts.view({g * batch_size});
        Tensor confidences = outputs.confidences.view({g * batch_size});
        if (execution.proposal_head == ProposalHead::Full) {
            Tensor logits =
                execution.work.alloc(DType::BF16, {dimension(target.vocab_size), g * batch_size});
            project(proposal_hidden, execution.parameters.draft->output_head, logits,
                    execution.work, execution.cuda);
            ops::argmax_with_probability(logits, flat_drafts, confidences,
                                         dimension(execution.parameters.profile->public_tokens),
                                         execution.work, execution.cuda);
        } else {
            if (!execution.parameters.proposal.has_value()) {
                throw std::logic_error("optimized DFlash proposal head is unavailable");
            }
            const auto& proposal = *execution.parameters.proposal;
            Tensor logits =
                execution.work.alloc(DType::BF16, {dimension(proposal.rows), g * batch_size});
            project(proposal_hidden, proposal.head, logits, execution.work, execution.cuda);
            ops::argmax_with_probability(
                logits, flat_drafts, confidences,
                proposal.token_ids ? dimension(proposal.rows)
                                   : dimension(execution.parameters.profile->public_tokens),
                execution.work, execution.cuda);
            if (proposal.token_ids)
                ops::remap_ids_inplace(flat_drafts,
                                       static_cast<const std::int32_t*>(proposal.token_ids->data),
                                       dimension(proposal.rows), execution.cuda);
        }
    }
}


} // namespace

CyclicKVCacheLayerView DraftExecutionView::local_layer(std::uint32_t layer,
                                                       const Tensor& offsets) const {
    return local.layer_view(layer, offsets);
}

PagedKVBatchLayerView DraftExecutionView::full_layer(std::uint32_t layer) const {
    if (!full) throw std::logic_error("draft full context is unavailable");
    return full->batch_layer_view(layer);
}

ops::OpGraphProfile SlidingWindowReadout::profile(ops::ContextBucket bucket) const {
    return ops::sliding_window_attention_graph_profile(geometry, context.capacity, bucket,
                                                       query.ne[3], query.ne[2]);
}

std::size_t SlidingWindowReadout::scratch_bytes(ops::ContextBucketRange buckets) const {
    return ops::sliding_window_attention_workspace_capacity_bytes(
        geometry, context.capacity, buckets, query.ne[3], query.ne[2], query.ne[2]);
}

void SlidingWindowReadout::execute(ops::ContextBucket bucket, MemoryArena& work,
                                   CudaExecutionView cuda) const {
    auto out = output;
    ops::sliding_window_attention(query, key, value, positions, valid_columns, geometry,
                                  context.capacity, scale, context, bucket, work, out, cuda);
}

void SlidingWindowReadout::execute(ops::ContextBucket bucket, CudaExecutionView cuda) const {
    MemoryArena work(scratch);
    execute(bucket, work, cuda);
}

ops::OpGraphProfile ContextAttentionReadout::profile(ops::ContextBucket bucket) const {
    return ops::context_softmax_attention_graph_profile(geometry, bucket, query.ne[3], query.ne[2]);
}

std::size_t ContextAttentionReadout::scratch_bytes(ops::ContextBucketRange buckets) const {
    return ops::context_softmax_attention_workspace_capacity_bytes(geometry, buckets, query.ne[3],
                                                                   query.ne[2], query.ne[2]);
}

void ContextAttentionReadout::execute(ops::ContextBucket bucket, MemoryArena& work,
                                      CudaExecutionView cuda) const {
    auto out = output;
    ops::context_softmax_attention(query, key, value, frontiers, valid_columns, table_rows,
                                   geometry, scale, context, bucket, work, out, cuda);
}

void ContextAttentionReadout::execute(ops::ContextBucket bucket, CudaExecutionView cuda) const {
    MemoryArena work(scratch);
    execute(bucket, work, cuda);
}

void draft_append_context(DraftExecutionView execution, const Tensor& features,
                          const Tensor& positions, const Tensor& counts, const Tensor& offsets,
                          const Tensor& table_rows,
                          ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (!execution.parameters.draft) throw std::logic_error("draft parameters are unavailable");
    auto scope = execution.work.scope();
    append_context_impl(execution, features, positions, counts, offsets, table_rows, envelope);
}

void draft_append_context(DraftExecutionView execution, const Tensor& features,
                          const Tensor& positions, const ops::RaggedSequenceView& ragged,
                          const ops::RaggedSequenceView& local_ragged, const Tensor& local_indices,
                          const Tensor& offsets, const Tensor& table_rows) {
    if (!execution.parameters.draft) throw std::logic_error("draft parameters are unavailable");
    const auto& target      = execution.parameters.profile->config.text;
    const auto& config      = *execution.parameters.profile->config.draft;
    const int n             = ragged.total_tokens();
    const int segments      = ragged.num_seqs();
    const int n_local       = local_ragged.total_tokens();
    const int feature_dim   = dimension(target.hidden_size *
                                        std::uint64_t(config.target_layer_ids.size()));
    const int hidden        = dimension(target.hidden_size);
    const int window        = dimension(config.sliding_window.value_or(0));
    nvtx::ScopedRange append_range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash,
                                   static_cast<std::uint64_t>(n));
    if (n <= 0 || segments <= 0 || local_ragged.num_seqs() != segments || n_local <= 0 ||
        n_local > n || features.dtype != DType::BF16 || features.ne[0] != feature_dim ||
        features.ne[1] != n || features.ne[2] != 1 || features.ne[3] != 1 ||
        positions.dtype != DType::I32 || positions.ne[0] != n || positions.ne[1] != 1 ||
        offsets.dtype != DType::I64 || offsets.ne[0] != segments ||
        table_rows.dtype != DType::I32 || table_rows.ne[0] != segments) {
        throw std::invalid_argument("DFlash packed context append inputs are invalid");
    }
    for (int s = 0; s < segments; ++s) {
        if (local_ragged.segment_length(s) != std::min(ragged.segment_length(s), window)) {
            throw std::invalid_argument("DFlash packed local window mismatch");
        }
    }
    const bool truncated = n_local < n;
    if (truncated && (local_indices.dtype != DType::I32 || local_indices.ne[0] != n_local ||
                      local_indices.ne[1] != 1)) {
        throw std::invalid_argument("DFlash packed local indices are invalid");
    }
    auto scope = execution.work.scope();
    if (config.dflash2) {
        Tensor input           = features;
        Tensor local_positions = positions;
        if (truncated) {
            input           = execution.work.alloc(DType::BF16, {feature_dim, n_local});
            local_positions = execution.work.alloc(DType::I32, {n_local});
            Tensor in_view  = input.view({feature_dim, 1, n_local});
            Tensor pos_view = local_positions.view({1, 1, n_local});
            ops::gather_batch_rows(features.view({feature_dim, 1, n}), local_indices, in_view,
                                   execution.cuda);
            ops::gather_batch_rows(positions.view({1, 1, n}), local_indices, pos_view,
                                   execution.cuda);
        }
        const auto roots = workspace::dflash_context(execution.work, target, config, n_local);
        Tensor projected = roots.projected;
        Tensor context   = roots.normalized;
        project(input, execution.parameters.draft->feature_projection, projected, execution.work,
                execution.cuda);
        ops::rmsnorm(projected, execution.parameters.draft->context_norm, config.rms_norm_eps,
                     false, context, execution.cuda);
        std::array<ops::ContextKVMaterializeLayerView, ops::kContextKVMaterializeLayers> layers;
        if (config.num_hidden_layers != layers.size() ||
            config.local_layer_count() != layers.size()) {
            throw std::invalid_argument(
                "masked draft context: config is outside the fused materialization contract");
        }
        for (int layer = 0; layer < dimension(config.num_hidden_layers); ++layer) {
            const auto& weights = execution.parameters.draft->layers[layer];
            layers[layer]       = {weights.context_key.weight, weights.context_value.weight,
                                   weights.key_norm, execution.local_layer(layer, offsets)};
        }
        ops::context_kv_materialize(context, local_ragged, local_positions, layers,
                                    *execution.parameters.profile->draft->context_materialization,
                                    execution.work, execution.cuda);
        return;
    }

    const auto roots = workspace::dflash_context(execution.work, target, config, n);
    Tensor projected = roots.projected;
    Tensor context   = roots.normalized;
    project(features, execution.parameters.draft->feature_projection, projected, execution.work,
            execution.cuda);
    ops::rmsnorm(projected, execution.parameters.draft->context_norm, config.rms_norm_eps, false,
                 context, execution.cuda);
    Tensor local_context   = context;
    Tensor local_positions = positions;
    if (truncated) {
        local_context   = execution.work.alloc(DType::BF16, {hidden, n_local});
        local_positions = execution.work.alloc(DType::I32, {n_local});
        Tensor lc_view  = local_context.view({hidden, 1, n_local});
        Tensor lp_view  = local_positions.view({1, 1, n_local});
        ops::gather_batch_rows(context.view({hidden, 1, n}), local_indices, lc_view,
                               execution.cuda);
        ops::gather_batch_rows(positions.view({1, 1, n}), local_indices, lp_view, execution.cuda);
    }
    const auto head_dim  = dimension(config.attention.head_dim);
    const auto kv_heads  = dimension(config.attention.num_key_value_heads);
    const auto key_width = dimension(config.attention.key_width());
    for (int layer = 0; layer < dimension(config.num_hidden_layers); ++layer) {
        auto layer_scope = execution.work.scope();
        const auto& weight =
            execution.parameters.draft->layers.at(static_cast<std::size_t>(layer));
        const bool local_layer =
            config.layer_types[layer] == dflash::AttentionKind::SlidingAttention;
        const int layer_columns = local_layer ? n_local : n;
        Tensor layer_context    = local_layer ? local_context : context;
        Tensor layer_positions  = local_layer ? local_positions : positions;
        auto layer_roots =
            workspace::dflash_context_layer(execution.work, target, config, layer_columns);
        Tensor key_raw    = layer_roots.key_raw.view({head_dim, kv_heads, layer_columns});
        Tensor value      = layer_roots.value.view({head_dim, kv_heads, layer_columns});
        Tensor key_flat   = key_raw.view({key_width, layer_columns});
        Tensor value_flat = value.view({key_width, layer_columns});
        ops::linear_pair(layer_context, layer_context, weight.context_key.weight,
                         weight.context_value.weight, key_flat, value_flat, execution.cuda);
        Tensor key = layer_roots.key.view({head_dim, kv_heads, layer_columns});
        ops::rmsnorm(key_raw, weight.key_norm, config.rms_norm_eps, false, key, execution.cuda);
        ops::rope(layer_positions, head_dim, config.rope_theta, key, execution.cuda);
        if (local_layer) {
            ops::kv_cache_append(key, value, local_ragged, layer_positions,
                                 execution.local_layer(config.compact_layer_index(layer), offsets),
                                 execution.cuda);
        } else {
            ops::kv_cache_append(key, value, ragged, layer_positions, table_rows,
                                 execution.full_layer(config.compact_layer_index(layer)),
                                 execution.cuda);
        }
    }
}

void draft_propose(DraftExecutionView execution, const DraftInputs& inputs, DraftOutputs& outputs,
                   DraftBuckets buckets, const DraftReadoutSubmit& submit) {
    if (!execution.parameters.draft) throw std::logic_error("draft parameters are unavailable");
    const auto width = outputs.ids.ne[0], batch = outputs.ids.ne[1];
    const auto g = outputs.drafts.ne[0];
    if (width < 2 || width > 16 || batch < 1 || g < 1 || g >= width ||
        outputs.drafts.ne[1] != batch) {
        throw std::invalid_argument("draft proposal has invalid geometry");
    }
    auto scope = execution.work.scope();
    propose_batch_impl(execution, inputs, outputs, batch, width, g, buckets, submit);
}

void bind_draft_readouts(DraftExecutionView execution, const DraftInputs& inputs,
                         const DraftOutputs& outputs, const DraftReadoutSubmit& submit) {
    const auto& target = execution.parameters.profile->config.text;
    const auto& config = *execution.parameters.profile->config.draft;
    const auto width = outputs.ids.ne[0], batch = outputs.ids.ne[1];
    const auto columns = width * batch;
    auto scope         = execution.work.scope();
    (void)execution.work.alloc(DType::BF16, {dimension(target.hidden_size), columns});
    const DraftBuckets buckets{submit.buckets.first, submit.buckets.first};
    for (std::uint32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        auto layer_scope = execution.work.scope();
        DraftAttentionTensors tensors;
        if (config.dflash2) {
            (void)workspace::dflash2_branch(execution.work, target, config, width, batch);
            tensors = dflash2_attention(execution.work, config.attention, width, batch);
        } else {
            const auto roots = workspace::dflash_attention(execution.work, target, config, columns);
            tensors          = dflash_attention(roots, config.attention, width, batch);
        }
        submit_layer_attention(execution, inputs, outputs.positions, tensors, layer, buckets,
                               submit);
    }
}

} // namespace ninfer::models::qwen3_5::execution
