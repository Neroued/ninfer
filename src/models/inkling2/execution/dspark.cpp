#include "models/inkling2/execution/dspark.h"

#include "core/transfer.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/markov_head.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/residual_add_rmsnorm.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/sequence.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/weighted_sum_rmsnorm.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace ninfer::models::inkling2::execution {
namespace {

std::size_t attention_bytes(const DSparkConfig& c, ops::AttentionHeadGeometry geometry, bool swa,
                            ops::ContextBucketRange buckets, int b, int w) {
    if (swa)
        return ops::block_sliding_window_attention_workspace_capacity_bytes(
            geometry, c.sliding_window, buckets, b, w, w);
    return ops::context_softmax_attention_workspace_capacity_bytes(geometry, buckets, b, w, w);
}

std::size_t linear_bytes(const ops::LinearProfile& p, int first, int last) {
    return ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, first, last);
}

struct DraftRoots {
    Tensor residual, normed, mlp;
};

struct DraftAttention {
    Tensor k, v, q, out, positions;
};

template <class Arena>
DraftRoots roots(Arena& a, const DSparkConfig& c, int columns) {
    const int h = dimension(c.hidden);
    return {a.alloc(DType::BF16, {h, columns}), a.alloc(DType::BF16, {h, columns}),
            a.alloc(DType::BF16, {h, columns})};
}

template <class Arena>
DraftAttention attention(Arena& a, const DSparkConfig& c, int w, int b) {
    const int kv = dimension(c.kv_width()), q = dimension(c.query_width());
    return {a.alloc(DType::BF16, {kv, w * b}), a.alloc(DType::BF16, {kv, w * b}),
            a.alloc(DType::BF16, {q, w * b}), a.alloc(DType::BF16, {q, w * b}),
            a.alloc(DType::I32, {w, b})};
}

DSparkReadout bind_readout(const ModelProfile& p, DSparkExecution e, const BatchView& call,
                           const DSparkProposal& proposal, const DraftAttention& a, int layer,
                           ops::ContextBucketRange buckets, CudaExecutionView cuda) {
    const auto& c = *p.config.dspark;
    const auto& l = c.layers[layer];
    const auto g  = p.dspark->geometry;
    const int w = a.positions.ne[0], b = call.rows;
    const auto bytes = attention_bytes(c, g, l.is_swa, buckets, b, w);
    auto scope       = e.arena.scope();
    return {
        g,
        c.sliding_window,
        a.q.view({g.head_dim, g.query_heads, w, b}),
        a.k.view({g.head_dim, g.kv_heads, w, b}),
        a.v.view({g.head_dim, g.kv_heads, w, b}),
        proposal.valid_columns,
        proposal.base_positions,
        call.dspark_tables,
        a.out.view({g.head_dim, g.query_heads, w, b}),
        l.is_swa ? ConstCyclicKVCacheLayerView(
                       e.states.dspark_swa_layer(l.state_slot, call.source_offsets))
                 : ConstCyclicKVCacheLayerView{},
        l.is_swa ? ConstPagedKVBatchLayerView{}
                 : ConstPagedKVBatchLayerView(e.cache.layer(l.state_slot)),
        bytes ? e.arena.alloc_bytes(bytes, 256) : MemorySpan{nullptr, 0, cuda.device()},
    };
}

void project(const Tensor& in, const LinearParameters& p, Tensor& out, MemoryArena& arena,
             CudaExecutionView cuda) {
    ops::linear(in, p.weight, out, p.policy, arena, cuda);
}

struct ContextProjection {
    Tensor k, v;
};

ContextProjection project_context(const Parameters& p, const Tensor& aux, const Tensor& positions,
                                  int i, MemoryArena& a, CudaExecutionView cuda) {
    const auto& c = *p.profile->config.dspark;
    const auto& d = *p.dspark;
    const auto& l = d.layers[i];
    const int h = dimension(c.hidden), kv = dimension(c.kv_width()), n = aux.ne[2];
    auto k       = a.alloc(DType::BF16, {kv, n});
    auto v       = a.alloc(DType::BF16, {kv, n});
    auto scope   = a.scope();
    auto context = a.alloc(DType::BF16, {h, n});
    ops::weighted_sum_rmsnorm(aux, d.alpha(i), l.dflare_norm, c.norm_epsilon, context, cuda);
    const std::array outputs{k, v};
    ops::linear_split(context, l.context_kv, outputs, a, cuda);
    const int head = dimension(c.head_dim), heads = dimension(c.kv_heads);
    auto key = k.view({head, heads, n});
    ops::rmsnorm_rope(positions.view({n}), l.context_key_norm, c.norm_epsilon, c.rope_theta, key,
                      cuda);
    return {key, v.view({head, heads, n})};
}

} // namespace

ops::OpGraphProfile DSparkReadout::profile(ops::ContextBucket bucket) const {
    if (swa.capacity)
        return ops::block_sliding_window_attention_graph_profile(geometry, window, bucket, q.ne[3],
                                                                 q.ne[2]);
    return ops::context_softmax_attention_graph_profile(geometry, bucket, q.ne[3], q.ne[2]);
}

void DSparkReadout::execute(ops::ContextBucket bucket, CudaExecutionView cuda) const {
    MemoryArena arena(scratch);
    const float scale = 1.0F / std::sqrt(float(geometry.head_dim));
    auto output       = out;
    if (swa.capacity)
        ops::block_sliding_window_attention(q, k, v, lengths, valid_columns, geometry, window,
                                            scale, swa, bucket, arena, output, cuda);
    else
        ops::context_softmax_attention(q, k, v, lengths, valid_columns, table_rows, geometry, scale,
                                       full, bucket, arena, output, cuda);
}

std::vector<DSparkReadout> plan_dspark_readouts(const Parameters& p, DSparkExecution e,
                                                const BatchView& call,
                                                const DSparkProposal& proposal,
                                                ops::ContextBucketRange buckets,
                                                CudaExecutionView cuda) {
    auto scope  = e.arena.scope();
    const int w = proposal.drafts.ne[1] + 1, b = call.rows;
    const auto& c = *p.profile->config.dspark;
    (void)roots(e.arena, c, w * b);
    const auto a = attention(e.arena, c, w, b);
    std::vector<DSparkReadout> out;
    out.reserve(c.num_layers);
    for (int i = 0; i < dimension(c.num_layers); ++i)
        out.push_back(bind_readout(*p.profile, e, call, proposal, a, i, buckets, cuda));
    return out;
}

void propose_dspark(const Parameters& p, DSparkExecution e, const BatchView& call,
                    const DSparkProposal& proposal, ops::ContextBucketRange buckets,
                    ops::ContextBucket bucket, CudaExecutionView cuda, const DSparkSubmit& submit) {
    auto& arena   = e.arena;
    const auto& c = *p.profile->config.dspark;
    const auto& d = *p.dspark;
    const int h = dimension(c.hidden), r = dimension(c.markov_rank);
    const int steps = proposal.drafts.ne[1], w = steps + 1, b = call.rows;
    auto scope = arena.scope();
    auto root  = roots(arena, c, w * b);
    {
        auto layer_scope = arena.scope();
        auto a           = attention(arena, c, w, b);
        {
            auto stem_scope = arena.scope();
            auto unused_ids = arena.alloc(DType::I32, {w, b});
            ops::prepare_masked_block(proposal.anchors, proposal.base_positions,
                                      proposal.valid_columns, 0, unused_ids, a.positions, cuda);
            auto emb     = arena.alloc(DType::BF16, {h, b});
            auto source  = arena.alloc(DType::BF16, {h, b + 1});
            auto anchors = source.slice(1, 0, b);
            ops::embedding(proposal.anchors, d.embedding, emb, cuda);
            ops::rmsnorm(emb, d.embed_norm, p.profile->config.text.norm_epsilon, false, anchors,
                         cuda);
            auto mask = source.slice(1, b, 1);
            copy_bytes_async({mask.data, mask.bytes(), cuda.device()},
                             {d.mask_embed.data, mask.bytes(), cuda.device()}, cuda);
            auto residual = root.residual.view({h, 1, w * b});
            ops::gather_batch_rows(source.view({h, 1, b + 1}), proposal.stem_indices, residual,
                                   cuda);
        }
        for (int i = 0; i < dimension(c.num_layers); ++i) {
            const auto& l = d.layers[i];
            if (i == 0)
                ops::rmsnorm(root.residual, l.input_norm, c.norm_epsilon, false, root.normed, cuda);
            else
                ops::residual_add_rmsnorm(root.residual, root.mlp, l.input_norm, c.norm_epsilon,
                                          root.residual, root.normed, cuda);
            const std::array outputs{a.k, a.v, a.q};
            ops::linear_split(root.normed, l.kvq, outputs, arena, cuda);
            auto readout = bind_readout(*p.profile, e, call, proposal, a, i, buckets, cuda);
            ops::rmsnorm_rope(a.positions, l.query_norm, l.key_norm, c.norm_epsilon, c.rope_theta,
                              readout.q, readout.k, cuda);
            if (submit)
                submit(readout);
            else
                readout.execute(bucket, cuda);
            project(a.out, l.output, root.mlp, arena, cuda);
            ops::residual_add_rmsnorm(root.residual, root.mlp, l.post_attention_norm,
                                      c.norm_epsilon, root.residual, root.normed, cuda);
            auto mlp_scope = arena.scope();
            auto mid       = arena.alloc(DType::BF16, {dimension(c.intermediate), w * b});
            ops::linear_swiglu(root.normed, l.gate_up.weight, mid, l.gate_up.policy, arena, cuda);
            project(mid, l.down, root.mlp, arena, cuda);
        }
    }
    ops::residual_add(root.mlp, root.residual, cuda);
    auto hidden = arena.alloc(DType::BF16, {h, steps * b});
    ops::rmsnorm_pack_tail(root.residual.view({h, w, b}), d.final_norm, c.norm_epsilon, hidden,
                           cuda);
    auto frnn = arena.alloc(DType::BF16, {h, b});
    ops::gather(e.states.dspark_f_rnn(call.source_offsets), proposal.rows, frnn, cuda);
    auto context  = arena.alloc(DType::FP32, {3 * r, b, steps});
    const auto& m = d.markov;
    ops::markov_rnn_context(hidden.view({h, steps, b}), frnn, m.w_hidden, m.w_frnn, m.rnn_bias,
                            arena, context, cuda);
    const int vocab = dimension(c.vocabulary);
    auto logits     = arena.alloc(DType::BF16, {vocab, b, steps});
    {
        auto head_scope = arena.scope();
        auto ordered    = hidden;
        if (b > 1) {
            ordered      = arena.alloc(DType::BF16, {h, steps * b});
            auto columns = ordered.view({h, 1, steps * b});
            ops::gather_batch_rows(hidden.view({h, 1, steps * b}), proposal.tail_indices, columns,
                                   cuda);
        }
        auto flat = logits.view({vocab, b * steps});
        project(ordered, d.output_head, flat, arena, cuda);
    }
    auto state  = arena.alloc(DType::BF16, {r, b});
    auto latent = arena.alloc(DType::BF16, {r, b});
    CUDA_CHECK(cudaMemsetAsync(state.data, 0, state.bytes(), cuda.stream()));
    auto prev        = proposal.anchors;
    const int domain = dimension(p.profile->public_tokens);
    for (int s = 0; s < steps; ++s) {
        auto step_logits   = logits.slice(2, s, 1).view({vocab, b});
        auto ids           = proposal.drafts.slice(1, s, 1).view({b});
        auto probabilities = proposal.probabilities.slice(1, s, 1).view({b});
        ops::markov_rnn_step(state, prev, context.slice(2, s, 1).view({3 * r, b}), m.w1_embed,
                             m.w_recurrent, proposal.rows, state, latent, cuda);
        ops::markov_logits_add_softcap(latent, m.fold_weight, m.fold_bias, step_logits, domain,
                                       c.final_logit_softcapping, proposal.rows, step_logits, cuda);
        ops::argmax_with_probability(step_logits, ids, probabilities, domain, arena, cuda);
        prev = ids;
    }
}

void dspark_mix_frnn(const Parameters& p, const Tensor& aux, Tensor& out, CudaExecutionView cuda) {
    const auto& c = *p.profile->config.dspark;
    ops::weighted_sum_rmsnorm(aux, p.dspark->alpha(dimension(c.num_layers)),
                              p.dspark->markov.dflare_norm, c.norm_epsilon, out, cuda);
}

void dspark_append_context(const Parameters& p, DSparkExecution e, const BatchView& call,
                           const Tensor& aux, const Tensor& counts, CudaExecutionView cuda) {
    const auto& c = *p.profile->config.dspark;
    const ops::KVCacheAppendPrefixExecutionEnvelope envelope{0, std::uint32_t(call.tokens)};
    const int head = dimension(c.head_dim), heads = dimension(c.kv_heads);
    for (int i = 0; i < dimension(c.num_layers); ++i) {
        auto scope    = e.arena.scope();
        auto kv       = project_context(p, aux, call.positions, i, e.arena, cuda);
        const auto k  = kv.k.view({head, heads, call.tokens, call.rows});
        const auto v  = kv.v.view({head, heads, call.tokens, call.rows});
        const auto& l = c.layers[i];
        if (l.is_swa)
            ops::kv_cache_append_prefix(
                k, v, call.positions, counts, envelope,
                e.states.dspark_swa_layer(l.state_slot, call.destination_offsets), cuda);
        else
            ops::kv_cache_append_prefix(k, v, call.positions, counts, call.dspark_tables, envelope,
                                        e.cache.layer(l.state_slot), cuda);
    }
}

void dspark_append_context(const Parameters& p, DSparkExecution e, const BatchView& call,
                           const Tensor& aux, const DSparkRaggedContext& meta,
                           CudaExecutionView cuda) {
    const auto& c = *p.profile->config.dspark;
    const int h = aux.ne[0], taps = aux.ne[1], n = aux.ne[2];
    auto& arena = e.arena;
    {
        auto scope   = arena.scope();
        auto swa_aux = aux;
        auto swa_pos = call.positions;
        if (meta.swa_indices.data) {
            const int count = meta.swa.total_tokens();
            swa_aux         = arena.alloc(DType::BF16, {h, taps, count});
            swa_pos         = arena.alloc(DType::I32, {count});
            auto pos        = swa_pos.view({1, 1, count});
            auto columns    = swa_aux.view({h * taps, 1, count});
            ops::gather_batch_rows(aux.view({h * taps, 1, n}), meta.swa_indices, columns, cuda);
            ops::gather_batch_rows(call.positions.view({1, 1, n}), meta.swa_indices, pos, cuda);
        }
        for (int i = 0; i < dimension(c.num_layers); ++i) {
            auto layer_scope = arena.scope();
            const auto& l    = c.layers[i];
            if (l.is_swa) {
                auto kv = project_context(p, swa_aux, swa_pos, i, arena, cuda);
                ops::kv_cache_append(
                    kv.k, kv.v, meta.swa, swa_pos,
                    e.states.dspark_swa_layer(l.state_slot, call.destination_offsets), cuda);
            } else {
                auto kv = project_context(p, aux, call.positions, i, arena, cuda);
                ops::kv_cache_append(kv.k, kv.v, meta.full, call.positions, call.dspark_tables,
                                     e.cache.layer(l.state_slot), cuda);
            }
        }
    }
    const int segments = meta.full.num_seqs();
    auto scope         = arena.scope();
    auto last          = arena.alloc(DType::BF16, {h, taps, segments});
    auto last_flat     = last.view({h * taps, 1, segments});
    ops::gather_batch_rows(aux.view({h * taps, 1, n}), meta.last_indices, last_flat, cuda);
    auto frnn = arena.alloc(DType::BF16, {h, segments});
    dspark_mix_frnn(p, last, frnn, cuda);
    ops::scatter(frnn, e.states.dspark_f_rnn(call.destination_offsets), {}, cuda);
}

std::size_t dspark_proposal_bytes(const ModelProfile& p, int b, int steps,
                                  ops::ContextBucketRange buckets) {
    const auto& c = *p.config.dspark;
    const auto& d = *p.dspark;
    const int h = dimension(c.hidden), r = dimension(c.markov_rank), w = steps + 1, n = w * b;
    const int kv = dimension(c.kv_width());
    const std::array split_rows{kv, kv, dimension(c.query_width())};
    WorkspaceLayoutBuilder a;
    const auto scratch = [&](std::size_t bytes) {
        if (bytes) {
            auto scope = a.scope();
            (void)a.alloc_bytes(bytes);
        }
    };
    (void)roots(a, c, n);
    {
        auto layer_scope = a.scope();
        (void)attention(a, c, w, b);
        {
            auto stem_scope = a.scope();
            (void)a.alloc(DType::I32, {w, b});
            (void)a.alloc(DType::BF16, {h, b});
            (void)a.alloc(DType::BF16, {h, b + 1});
        }
        for (std::size_t i = 0; i < d.layers.size(); ++i) {
            const auto& l = d.layers[i];
            scratch(ops::linear_split_workspace_capacity_bytes(l.kvq, split_rows, n, n));
            scratch(attention_bytes(c, d.geometry, c.layers[i].is_swa, buckets, b, w));
            scratch(linear_bytes(l.output, n, n));
            auto mlp_scope = a.scope();
            (void)a.alloc(DType::BF16, {dimension(c.intermediate), n});
            scratch(ops::linear_swiglu_workspace_capacity_bytes(
                l.gate_up.qtype, l.gate_up.n, l.gate_up.k, l.gate_up.policy, n, n));
            scratch(linear_bytes(l.down, n, n));
        }
    }
    (void)a.alloc(DType::BF16, {h, steps * b});
    (void)a.alloc(DType::BF16, {h, b});
    (void)a.alloc(DType::FP32, {3 * r, b, steps});
    scratch(ops::markov_rnn_context_workspace_capacity_bytes(h, r, steps, b));
    (void)a.alloc(DType::BF16, {dimension(c.vocabulary), b, steps});
    {
        auto head_scope = a.scope();
        if (b > 1) (void)a.alloc(DType::BF16, {h, steps * b});
        scratch(linear_bytes(d.output_head, steps * b, steps * b));
    }
    (void)a.alloc(DType::BF16, {r, b});
    (void)a.alloc(DType::BF16, {r, b});
    scratch(ops::argmax_with_probability_workspace_capacity_bytes(p.public_tokens, b));
    return a.peak_bytes();
}

std::size_t dspark_context_bytes(const ModelProfile& p, int cols, int segments) {
    const auto& c = *p.config.dspark;
    const int h = dimension(c.hidden), kv = dimension(c.kv_width());
    const int taps = dimension(c.target_layer_ids.size());
    const std::array rows{kv, kv};
    WorkspaceLayoutBuilder a;
    {
        auto scope = a.scope();
        if (segments && cols > dimension(c.sliding_window)) {
            const int compact = dimension(
                std::min<std::uint64_t>(cols - 1, std::uint64_t(segments) * c.sliding_window));
            (void)a.alloc(DType::BF16, {h, taps, compact});
            (void)a.alloc(DType::I32, {compact});
        }
        (void)a.alloc(DType::BF16, {kv, cols});
        (void)a.alloc(DType::BF16, {kv, cols});
        (void)a.alloc(DType::BF16, {h, cols});
        for (const auto& l : p.dspark->layers) {
            auto split = a.scope();
            const auto bytes =
                ops::linear_split_workspace_capacity_bytes(l.context_kv, rows, 1, cols);
            if (bytes) (void)a.alloc_bytes(bytes);
        }
    }
    {
        const auto count = segments ? segments : cols;
        if (segments) (void)a.alloc(DType::BF16, {h, taps, count});
        (void)a.alloc(DType::BF16, {h, count});
    }
    return a.peak_bytes();
}

} // namespace ninfer::models::inkling2::execution
