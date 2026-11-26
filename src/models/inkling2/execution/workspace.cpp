#include "models/inkling2/execution/workspace.h"

#include "models/inkling2/vision/execution.h"
#include "models/inkling2/execution/dspark.h"
#include "ninfer/ops/speculative_accept.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/loop_message.h"
#include "ninfer/ops/moe.h"
#include "ninfer/ops/relative_attention.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>

namespace ninfer::models::inkling2::execution {
namespace {

std::size_t linear_bytes(const ops::LinearProfile& p, int first, int last) {
    return ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, first, last);
}

void scratch(WorkspaceLayoutBuilder& a, std::size_t bytes) {
    if (!bytes) return;
    auto scope = a.scope();
    (void)a.alloc_bytes(bytes);
}

std::size_t main_body_bytes(const ModelProfile& p, const ExecutionShape& s, int min_width,
                            int max_width, int batch, int sm_count, bool ragged) {
    WorkspaceLayoutBuilder a;
    const auto& c           = p.config.text;
    const int h             = dimension(c.hidden);
    const int min_col       = dimension(std::uint64_t(min_width) * batch);
    const int max_col       = dimension(std::uint64_t(max_width) * batch);
    const auto full_buckets = ops::ContextBucketRange::through(s.max_context);
    const auto swa_buckets =
        ops::ContextBucketRange{ops::ContextBucket::K2, ops::ContextBucket::K2};
    const int max_segments = ragged ? dimension(s.max_concurrency) : 1;

    (void)workspace::main_roots(a, p, max_col);
    {
        auto stem_scope = a.scope();
        (void)a.alloc(DType::BF16, {h, max_col});
    }

    for (std::size_t phys = 0; phys < p.text.layers.size(); ++phys) {
        const auto& lp = p.text.layers[phys];
        for (std::size_t pass = 0; pass < lp.attention_passes.size(); ++pass) {
            const auto& ap    = lp.attention_passes[pass];
            const auto& mp    = lp.moe_passes[pass];
            const bool looped = pass > 0;
            if (looped && c.loop) {
                scratch(a, ops::loop_message_blend_rmsnorm_workspace_capacity_bytes(
                               h, dimension(c.loop->gate_rank), 3, min_col, max_col));
            }
            {
                auto attn_scope      = a.scope();
                const bool use_chunk = lp.is_swa && ragged;
                (void)workspace::attention(a, c, lp.is_swa, s.swa_kv_storage, max_width, batch,
                                           ragged, use_chunk);
                if (ap.qkv_split) {
                    scratch(a, ops::linear_split_workspace_capacity_bytes(
                                   *ap.qkv_split, ap.qkv_rows, min_col, max_col));
                } else {
                    scratch(a, ops::linear_split_workspace_capacity_bytes(ap.qrg_split, ap.qrg_rows,
                                                                          min_col, max_col));
                    scratch(a, ops::linear_pair_workspace_capacity_bytes(ap.key, ap.value, min_col,
                                                                         max_col));
                    if (ap.k_conv_proj) {
                        scratch(a, ops::linear_pair_workspace_capacity_bytes(
                                       *ap.k_conv_proj, *ap.v_conv_proj, min_col, max_col));
                    }
                }
                if (ragged) {
                    scratch(a,
                            ops::relative_softmax_attention_ragged_workspace_capacity_bytes(
                                lp.attention_geometry, lp.is_swa ? s.swa_kv_storage : s.kv_storage,
                                lp.is_swa, lp.is_swa ? swa_buckets : full_buckets, min_width,
                                max_width, 1, max_segments, sm_count));
                } else if (lp.is_swa) {
                    scratch(a, ops::relative_softmax_attention_swa_workspace_capacity_bytes(
                                   lp.attention_geometry, s.swa_kv_storage, c.sliding_window,
                                   swa_buckets, batch, min_width, max_width, sm_count));
                } else {
                    scratch(a, ops::relative_softmax_attention_full_workspace_capacity_bytes(
                                   lp.attention_geometry, s.kv_storage, full_buckets, batch,
                                   min_width, max_width, sm_count));
                }
                scratch(a, linear_bytes(ap.output, min_col, max_col));
            }
            {
                auto mlp_scope = a.scope();
                (void)a.alloc(DType::BF16, {h, max_col, 1});
                if (looped && c.loop) {
                    // The FP32 mix and its blend scratch retire before MoE output and routes.
                    auto blend_scope = a.scope();
                    (void)a.alloc(DType::FP32, {h, max_col, 1});
                    scratch(a, ops::loop_message_blend_rmsnorm_workspace_capacity_bytes(
                                   h, dimension(c.loop->gate_rank), 1, min_col, max_col));
                }
                (void)a.alloc(DType::BF16, {h, max_col});
                auto moe_scope = a.scope();
                (void)workspace::moe(a, c, max_col);
                scratch(a, ops::topk_sigmoid_route_workspace_capacity_bytes(mp.routing, min_col,
                                                                            max_col));
                scratch(a, ops::routed_swiglu_workspace_capacity_bytes(
                               mp.routed, mp.subset, DType::BF16, ops::MoeEpilogue::Write, min_col,
                               max_col));
            }
        }
    }
    return a.peak_bytes();
}

} // namespace

WorkspacePlan plan_workspace(const ModelProfile& p, const ExecutionShape& s,
                             std::int32_t sm_count, std::size_t vision_memory_budget) {
    WorkspacePlan out;
    const int chunk = dimension(s.prefill_chunk);
    out.prefill     = main_body_bytes(p, s, 1, chunk, 1, sm_count, /*ragged=*/true);

    for (std::uint32_t b = 1; b <= s.max_concurrency; ++b) {
        const int batch = dimension(b);
        out.ordinary =
            std::max(out.ordinary, main_body_bytes(p, s, 1, 1, batch, sm_count, /*ragged=*/false));
    }

    const int head_cols =
        s.purpose == EnginePurpose::CausalScoring
            ? dimension(std::min(s.prefill_chunk, 1024U))
            : dimension(std::uint64_t(s.max_concurrency) * (s.draft_tokens + 1ULL));
    out.finish = linear_bytes(p.text.head, 1, head_cols);
    if (s.purpose == EnginePurpose::Generation) {
        out.finish =
            std::max(out.finish,
                     ops::sampling_workspace_capacity_bytes(p.public_tokens, 1, s.max_concurrency));
    } else {
        WorkspaceLayoutBuilder a;
        (void)a.alloc(DType::BF16, {dimension(p.local.vocabulary.size), head_cols});
        (void)a.alloc(DType::I32, {head_cols});
        (void)a.alloc(DType::FP32, {head_cols});
        (void)a.alloc_bytes(out.finish);
        out.finish = a.peak_bytes();
    }

    out.general_bytes = std::max({out.prefill, out.ordinary, out.finish});
    if (p.dspark) {
        const int k = dimension(s.draft_tokens), batch = dimension(s.max_concurrency);
        const auto buckets = ops::ContextBucketRange::through(s.max_context);
        out.general_bytes  = std::max({
            out.general_bytes,
            dspark_context_bytes(p, chunk, std::min(chunk, batch)),
            ops::sampling_with_probability_workspace_capacity_bytes(p.public_tokens, 1, batch),
            dspark_context_bytes(p, (k + 1) * batch, 0),
            ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(p.public_tokens, 1, k, 1,
                                                                            batch),
        });
        for (int b = 1; b <= batch; ++b) {
            out.general_bytes =
                std::max(out.general_bytes, main_body_bytes(p, s, 2, k + 1, b, sm_count, false));
            for (int steps = 1; steps <= k; ++steps)
                out.general_bytes =
                    std::max(out.general_bytes, dspark_proposal_bytes(p, b, steps, buckets));
        }
    }
    out.bytes         = out.general_bytes;
    if (p.vision) {
        out.vision =
            vision::plan_resources(*p.config.vision, *p.vision, s.max_context, vision_memory_budget);
        out.bytes = std::max(out.bytes, out.vision->encoder.workspace_bytes);
    }
    return out;
}

} // namespace ninfer::models::inkling2::execution
