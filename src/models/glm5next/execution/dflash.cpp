#include "models/glm5next/execution/dflash.h"

#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm_gathered_features.h"
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>
#include <cmath>

namespace ninfer::models::glm5next::execution {
namespace {
void prepare_dynamic_branch(const Tensor& residual, const Tensor& norm, float epsilon,
                            const DynamicConvParameters& conv, workspace::DraftBranch& branch,
                            MemoryArena& arena, CudaExecutionView cuda) {
    auto scope      = arena.scope();
    const int width = residual.ne[1], batch = residual.ne[2];
    MemoryArena scratch(
        arena.alloc_bytes(ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
            residual.ne[0], width, width, batch, batch)));
    ops::rmsnorm_dynamic_grouped_conv_prepare(residual, norm, epsilon, conv.base,
                                              conv.projection.weight, branch.prepared, branch.delta,
                                              scratch, cuda);
}

void finish_dynamic_branch(const Tensor& input, const LinearParameters& projection,
                           const DynamicConvParameters& conv, workspace::DraftBranch& branch,
                           Tensor& residual, ParallelExecution& parallel, CudaExecutionView cuda) {
    ops::linear_dynamic_grouped_conv(input, projection.weight, conv.base, branch.delta,
                                     branch.partial, cuda);
    parallel.sum(branch.partial, cuda);
    ops::residual_add(branch.partial, residual, cuda);
}

void scratch(WorkspaceLayoutBuilder& a, std::size_t bytes) {
    auto scope = a.scope();
    (void)a.alloc_bytes(bytes);
}

std::size_t linear(const ops::LinearProfile& p, int lo, int hi) {
    return ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, lo, hi);
}
} // namespace

ops::SlidingWindowAttentionGraphProfile DraftReadout::profile(std::uint32_t bound) const {
    return ops::sliding_window_attention_graph_profile(geometry, context.capacity, {0, bound},
                                                       query.ne[2], query.ne[3]);
}

void DraftReadout::execute(std::uint32_t bound, CudaExecutionView cuda) const {
    MemoryArena arena(scratch);
    auto out = output;
    ops::sliding_window_attention(query, key, value, positions, valid_columns, geometry,
                                  context.capacity, float(1.0 / std::sqrt(geometry.head_dim)),
                                  context, {0, bound}, arena, out, cuda);
}

DraftAttentionRun::DraftAttentionRun(const Parameters& p, DraftExecution e, const DraftInputs& in,
                                     const DraftOutputs& out, Tensor residual, std::uint32_t layer,
                                     CudaExecutionView cuda)
    : parameters_(p), execution_(e), inputs_(in), residual_(residual), positions_(out.positions),
      layer_(layer), cuda_(cuda), scope_(e.workspace.scope()),
      buffers_(
          workspace::draft_attention(e.workspace, *p.profile, residual.ne[1], residual.ne[2])) {}

DraftReadout DraftAttentionRun::readout() {
    const auto& p = *parameters_.profile;
    DraftReadout out{buffers_.query,
                     buffers_.key,
                     buffers_.value,
                     positions_,
                     inputs_.valid_columns,
                     buffers_.output,
                     execution_.states.draft_layer(layer_, inputs_.state_offsets),
                     p.draft->attention,
                     {}};
    auto scope       = execution_.workspace.scope();
    const auto bytes = ops::sliding_window_attention_workspace_capacity_bytes(
        out.geometry, out.context.capacity, {0, p.config.text.max_context}, residual_.ne[1],
        residual_.ne[1], residual_.ne[2]);
    out.scratch = execution_.workspace.alloc_bytes(bytes);
    return out;
}

void DraftAttentionRun::execute(const DraftSubmit& submit) {
    const auto& p      = parameters_.draft->layers.at(layer_);
    const auto& config = *parameters_.profile->config.draft;
    const int h = residual_.ne[0], w = residual_.ne[1], b = residual_.ne[2];
    auto& arena = execution_.workspace;
    auto& v     = buffers_;
    prepare_dynamic_branch(residual_, p.input_norm, config.rms_norm_eps, p.attention_conv, v.branch,
                           arena, cuda_);
    const std::array outputs{v.key.view({v.key.ne[0] * v.key.ne[1], w * b}),
                             v.value.view({v.value.ne[0] * v.value.ne[1], w * b}),
                             v.query.view({v.query.ne[0] * v.query.ne[1], w * b})};
    ops::linear_split(v.branch.prepared.view({h, w * b}), p.kvq.weight, outputs, cuda_);
    ops::rmsnorm_rope(positions_, p.query_norm, p.key_norm, config.rms_norm_eps, config.rope_theta,
                      v.query, v.key, cuda_);
    const auto call = readout();
    if (submit)
        submit(call);
    else
        call.execute(inputs_.context_bound, cuda_);
    finish_dynamic_branch(v.output.view({v.output.ne[0] * v.output.ne[1], w, b}), p.output,
                          p.attention_conv, v.branch, residual_, execution_.parallel, cuda_);
}

DFlashRun::DFlashRun(const Parameters& p, DraftExecution e, DraftInputs in, DraftOutputs out,
                     CudaExecutionView cuda)
    : parameters_(p), execution_(e), inputs_(in), outputs_(out), cuda_(cuda),
      scope_(e.workspace.scope()),
      residual_(e.workspace.alloc(
          DType::BF16, {dimension(p.profile->config.text.hidden), out.ids.ne[0], out.ids.ne[1]})) {}

void DFlashRun::stem() {
    const auto& c = *parameters_.profile->config.draft;
    ops::prepare_masked_block(inputs_.anchors, inputs_.frontiers, inputs_.valid_columns,
                              dimension(c.mask_token_id), outputs_.ids, outputs_.positions, cuda_);
    auto flat = residual_.view({residual_.ne[0], residual_.ne[1] * residual_.ne[2]});
    ops::embedding(outputs_.ids.view({dimension(outputs_.ids.numel())}),
                   parameters_.draft->embedding, flat, cuda_);
}

DraftAttentionRun DFlashRun::attention(std::uint32_t layer) {
    return DraftAttentionRun(parameters_, execution_, inputs_, outputs_, residual_, layer, cuda_);
}

void DFlashRun::mlp(std::uint32_t layer) {
    const auto& p      = parameters_.draft->layers.at(layer);
    const auto& config = *parameters_.profile->config.draft;
    auto& arena        = execution_.workspace;
    auto scope         = arena.scope();
    const int h = residual_.ne[0], w = residual_.ne[1], b = residual_.ne[2];
    auto branch       = workspace::draft_branch(arena, *parameters_.profile, w, b);
    auto intermediate = arena.alloc(DType::BF16, {p.down.weight.k, w, b});
    prepare_dynamic_branch(residual_, p.post_attention_norm, config.rms_norm_eps, p.mlp_conv,
                           branch, arena, cuda_);
    auto flat = intermediate.view({intermediate.ne[0], w * b});
    ops::linear_swiglu(branch.prepared.view({h, w * b}), p.gate_up.weight, flat, p.gate_up.policy,
                       arena, cuda_);
    finish_dynamic_branch(intermediate, p.down, p.mlp_conv, branch, residual_, execution_.parallel,
                          cuda_);
}

void DFlashRun::finish() {
    const auto& p       = *parameters_.draft;
    const auto& profile = *parameters_.profile;
    const auto& c       = *profile.config.draft;
    const int d = outputs_.drafts.ne[0], b = outputs_.drafts.ne[1];
    auto& arena = execution_.workspace;
    auto scope  = arena.scope();
    auto v      = workspace::draft_finish(arena, profile, d, b);
    ops::rmsnorm_pack_tail(residual_, p.final_norm, c.rms_norm_eps, v.hidden, cuda_);
    ops::linear_topk(v.hidden, p.head.weight, profile.local.vocabulary.begin, profile.public_tokens,
                     p.head.policy, v.keys, arena, cuda_);
    auto projected = v.projected.view({v.projected.ne[0], d * b});
    ops::linear(v.hidden, p.selector.weight, projected, p.selector.policy, arena, cuda_);
    execution_.parallel.gather(v.keys, v.gathered, cuda_);
    ops::candidate_selector_path(v.gathered, v.projected, inputs_.anchors, p.predecessor_codebook,
                                 p.successor_codebook, profile.public_tokens, inputs_.frontiers,
                                 inputs_.sampling, outputs_.candidate_ids, outputs_.drafts,
                                 outputs_.proposal_q, arena, cuda_);
}

void draft_append_context(const Parameters& p, DraftExecution e, const Tensor& features,
                          const Tensor& positions, const Tensor& counts, const Tensor& offsets,
                          ops::ContextKVMaterializeExecutionEnvelope envelope,
                          CudaExecutionView cuda) {
    const auto& profile = *p.profile;
    const auto& draft   = *p.draft;
    auto scope          = e.workspace.scope();
    const int c = features.ne[1], b = features.ne[2];
    auto v = workspace::draft_context(e.workspace, profile, c * b);
    ops::linear(features.view({features.ne[0], c * b}), draft.feature_projection.weight,
                v.projected, draft.feature_projection.policy, e.workspace, cuda);
    e.parallel.gather(v.projected, v.gathered, cuda);
    ops::rmsnorm_gathered_features(v.gathered, draft.context_norm,
                                   profile.config.draft->rms_norm_eps, v.normalized, cuda);
    std::array<ops::ContextKVMaterializeLayerView, ops::kContextKVMaterializeLayers> layers;
    if (draft.layers.size() != layers.size())
        throw std::invalid_argument("DFlash context layer group differs from materialization");
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = draft.layers[i];
        layers[i]         = {layer.context_key, layer.context_value, layer.key_norm,
                             e.states.draft_layer(i, offsets)};
    }
    ops::context_kv_materialize(v.normalized.view({v.normalized.ne[0], c, b}), positions, counts,
                                layers, profile.draft->context, envelope, e.workspace, cuda);
}

std::size_t draft_workspace_capacity(const ModelProfile& p, int width, int max_batch) {
    WorkspaceLayoutBuilder a;
    const auto& c = *p.config.draft;
    const int h   = dimension(p.config.text.hidden);
    for (int b = 1; b <= max_batch; ++b) {
        auto batch  = a.scope();
        const int m = width * b;
        (void)a.alloc(DType::BF16, {h, width, b});
        for (const auto& layer : p.draft->layers) {
            {
                auto scope = a.scope();
                (void)workspace::draft_attention(a, p, width, b);
                scratch(a, ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                               h, width, width, b, b));
                const auto& g = p.draft->attention;
                const std::array rows{g.kv_heads * g.head_dim, g.kv_heads * g.head_dim,
                                      g.query_heads * g.head_dim};
                scratch(a, ops::linear_split_workspace_capacity_bytes({layer.kvq, {}}, rows, m, m));
                scratch(a, ops::sliding_window_attention_workspace_capacity_bytes(
                               g, c.sliding_window.value(), {0, p.config.text.max_context}, width,
                               width, b));
            }
            {
                auto scope = a.scope();
                (void)workspace::draft_branch(a, p, width, b);
                (void)a.alloc(DType::BF16, {layer.down.k, width, b});
                scratch(a, ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                               h, width, width, b, b));
                scratch(a, ops::linear_swiglu_workspace_capacity_bytes(
                               layer.gate_up.qtype, layer.gate_up.n, layer.gate_up.k,
                               layer.gate_up.policy, m, m));
            }
        }
        {
            auto scope = a.scope();
            (void)workspace::draft_finish(a, p, width - 1, b);
            const auto& head = p.draft->head;
            scratch(a, ops::linear_topk_workspace_capacity_bytes(head.qtype, head.n, head.k,
                                                                 head.policy, (width - 1) * b,
                                                                 (width - 1) * b));
            scratch(a, linear(p.draft->selector, (width - 1) * b, (width - 1) * b));
            scratch(a, ops::candidate_selector_path_workspace_capacity_bytes(
                           ops::CandidateSelectorInput::GatheredKeys, width - 1, width - 1, b, b));
        }
    }
    return a.peak_bytes();
}

std::size_t draft_context_workspace_capacity(const ModelProfile& p, const ExecutionShape& s) {
    WorkspaceLayoutBuilder a;
    for (int b = 1; b <= dimension(s.max_concurrency); ++b) {
        const int last = b == 1 ? std::max(std::min(s.prefill_chunk, p.draft->context.capacity),
                                           s.draft_tokens + 1)
                                : s.draft_tokens + 1;
        auto scope     = a.scope();
        (void)workspace::draft_context(a, p, last * b);
        scratch(a, linear(p.draft->feature_projection, 1, last * b));
        scratch(a,
                ops::context_kv_materialize_workspace_capacity_bytes(p.draft->context, b, 1, last));
    }
    return a.peak_bytes();
}
} // namespace ninfer::models::glm5next::execution
