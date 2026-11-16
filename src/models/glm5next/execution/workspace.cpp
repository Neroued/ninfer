#include "models/glm5next/execution/workspace.h"
#include "models/glm5next/execution/dflash.h"
#include "ninfer/ops/linear_pair.h"

#include "models/glm5next/state/context.h"
#include "models/glm5next/vision/execution.h"
#include "ninfer/ops/kda_input_proj.h"
#include "ninfer/ops/kimi_delta_attention.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_accept.h"
#include "ninfer/ops/target_logprobs.h"
#include "ninfer/ops/vocabulary.h"

#include <algorithm>

namespace ninfer::models::glm5next::execution {
namespace {
std::size_t linear(const ops::LinearProfile& p, int first, int last) {
    return ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, first, last);
}

void scratch(WorkspaceLayoutBuilder& a, std::size_t bytes) {
    auto scope = a.scope();
    (void)a.alloc_bytes(bytes);
}

void mla(WorkspaceLayoutBuilder& a, const ModelProfile& p, const MlaProfile& m,
         const ExecutionShape& s, int first, int last, int local, int sm_count, bool record,
         bool frozen) {
    const auto storage = latent_storage(s.kv_storage);
    for (int mode = 0; mode < (frozen ? 2 : 1); ++mode) {
        auto scope        = a.scope();
        const bool reuse  = mode != 0;
        const int minimum = dimension(std::uint64_t(first) * local);
        const int columns = dimension(std::uint64_t(last) * local);
        (void)workspace::mla(a, p.config.text, columns, reuse, !frozen);
        const auto& profile = m.prepare.at(
            static_cast<std::size_t>(reuse ? ops::KpoolMlaPrepareProjection::QueryAndHistory
                                           : ops::KpoolMlaPrepareProjection::Full));
        scratch(a, record ? ops::kpool_mla_prepare_record_workspace_capacity_bytes(
                                profile, storage, s.index_format, local, first, last, sm_count)
                          : ops::kpool_mla_prepare_update_workspace_capacity_bytes(
                                profile, storage, s.index_format, local, first, last, sm_count));
        if (reuse)
            scratch(a, linear(m.query, minimum, columns));
        else {
            scratch(a, ops::linear_split_workspace_capacity_bytes(m.query_index, m.query_rows,
                                                                  minimum, columns));
            scratch(a, ops::kpool_indexer_workspace_capacity_bytes(m.indexer, s.index_format,
                                                                   {s.max_context}, local, first,
                                                                   last, sm_count));
        }
        scratch(a, ops::kpool_sparse_mla_workspace_capacity_bytes(
                       m.attention, storage, {s.max_context}, local, first, last, sm_count));
        scratch(a, linear(m.output, minimum, columns));
    }
}

std::size_t body(const ModelProfile& p, const ExecutionShape& s, int first, int last, int batch,
                 int sm_count, bool prefill, bool mtp, std::size_t attention_bytes) {
    WorkspaceLayoutBuilder a;
    const int minimum = dimension(std::uint64_t(first) * batch);
    const int columns = dimension(std::uint64_t(last) * batch);
    if (mtp) {
        (void)workspace::mtp_roots(a, p.config.text, columns);
        if (p.local.size > 1)
            (void)workspace::attention(a, dimension(p.config.text.hidden), columns, p.local.size);
        {
            auto scope = a.scope();
            (void)workspace::mtp_stem(a, p.config.text, columns);
            scratch(a, linear(p.mtp->stem, minimum, columns));
        }
        scratch(a, attention_bytes);
        scratch(a, ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(
                       p.mtp->moe.math, p.mtp->moe.subset, minimum, columns));
        return a.peak_bytes();
    }
    (void)workspace::roots(a, p.config.text, columns);
    if (!prefill && p.local.size > 1)
        (void)workspace::attention(a, dimension(p.config.text.hidden), columns, p.local.size);
    {
        auto scope = a.scope();
        (void)a.alloc(DType::BF16, {dimension(p.config.text.hidden), columns});
        scratch(a, ops::mhc_expand_pre_workspace_capacity_bytes(p.text.mhc, sm_count, minimum,
                                                                columns));
    }
    scratch(a, ops::mhc_pre_workspace_capacity_bytes(p.text.mhc, sm_count, minimum, columns));
    scratch(a, ops::mhc_post_pre_workspace_capacity_bytes(p.text.mhc, sm_count, minimum, columns));
    scratch(a, ops::mhc_post_mean_norm_workspace_capacity_bytes(p.text.mhc, sm_count, minimum,
                                                                columns));
    for (const auto& layer : p.text.layers) {
        if (const auto* k = std::get_if<KdaProfile>(&layer.mixer)) {
            auto scope         = a.scope();
            const bool records = !prefill && last > 1;
            (void)workspace::kda(a, p, columns, records);
            scratch(a, records ? ops::kda_input_proj_conv_record_workspace_capacity_bytes(
                                     k->qkv, batch, first, last)
                               : ops::kda_input_proj_conv_update_workspace_capacity_bytes(
                                     k->qkv, batch, first, last));
            scratch(a, ops::linear_split_workspace_capacity_bytes({k->control, {}}, k->control_rows,
                                                                  minimum, columns));
            scratch(
                a, ops::linear_pair_workspace_capacity_bytes(k->forget, k->gate, minimum, columns));
            if (prefill)
                scratch(a, ops::kimi_delta_attention_workspace_capacity_bytes(
                               dimension(p.local.kda_heads.size), dimension(p.local.kda_heads.size),
                               first, last));
            scratch(a, linear(k->output, minimum, columns));
        }
        if (const auto* d = std::get_if<DenseProfile>(&layer.ffn))
            scratch(a, ops::swiglu_workspace_capacity_bytes(d->math, minimum, columns));
        else
            scratch(a, ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(
                           std::get<MoeProfile>(layer.ffn).math,
                           std::get<MoeProfile>(layer.ffn).subset, minimum, columns));
    }
    scratch(a, attention_bytes);
    return a.peak_bytes();
}

std::size_t history(const ModelProfile& p, const ExecutionShape& s, int first, int last, int batch,
                    std::size_t attention_bytes, bool sequence = false) {
    WorkspaceLayoutBuilder a;
    const int columns = dimension(std::uint64_t(last) * batch);
    const int minimum = dimension(std::uint64_t(first) * batch);
    (void)a.alloc(DType::BF16, {dimension(p.config.text.hidden), columns});
    (void)a.alloc(DType::BF16, {dimension(p.config.text.hidden), columns});
    if (!sequence && p.local.size > 1)
        (void)a.alloc(DType::BF16, {dimension(p.config.text.hidden), columns});
    {
        auto scope = a.scope();
        (void)workspace::mtp_stem(a, p.config.text, columns);
        scratch(a, linear(p.mtp->stem, minimum, columns));
    }
    scratch(a, attention_bytes);
    return a.peak_bytes();
}

std::size_t attention_workspace(const ModelProfile& p, const ExecutionShape& s, int first, int last,
                                int batch, int sm_count, bool record, bool mtp) {
    WorkspaceLayoutBuilder a;
    if (mtp) {
        mla(a, p, p.mtp->attention, s, first, last, batch, sm_count, false, true);
    } else {
        for (const auto& layer : p.text.layers)
            if (const auto* m = std::get_if<MlaProfile>(&layer.mixer))
                mla(a, p, *m, s, first, last, batch, sm_count, record, false);
    }
    return a.peak_bytes();
}

std::size_t history_attention_workspace(const ModelProfile& p, const ExecutionShape& s, int first,
                                        int last, int batch, int sm_count) {
    const auto& profile = p.mtp->attention.prepare.at(
        static_cast<std::size_t>(ops::KpoolMlaPrepareProjection::History));
    return ops::kpool_mla_prepare_update_workspace_capacity_bytes(
        profile, latent_storage(s.kv_storage), s.index_format, batch, first, last, sm_count);
}

} // namespace

WorkspacePlan plan_workspace(const ModelProfile& p, const ExecutionShape& s, int sm_count,
                             std::size_t vision_budget) {
    WorkspacePlan out;
    const int chunk                = dimension(s.prefill_chunk);
    out.prefill                    = body(p, s, 1, chunk, 1, sm_count, true, false,
                                          attention_workspace(p, s, 1, chunk, 1, sm_count, false, false));
    std::size_t ordinary_attention = 0, verify_attention = 0, draft_attention = 0,
                history_attention = 0;
    // Local attention geometry is independent of the global roots. Accumulate its
    // high-water mark once; every local row count <= the current global count is covered.
    for (std::uint32_t b = 1; b <= s.max_concurrency; ++b) {
        const int batch    = dimension(b);
        ordinary_attention = std::max(
            ordinary_attention, attention_workspace(p, s, 1, 1, batch, sm_count, false, false));
        out.ordinary = std::max(
            out.ordinary, body(p, s, 1, 1, batch, sm_count, false, false, ordinary_attention));
        if (p.mtp || p.draft) {
            verify_attention =
                std::max(verify_attention,
                         attention_workspace(p, s, 2, dimension(std::uint64_t(s.draft_tokens) + 1),
                                             batch, sm_count, true, false));
            out.verify =
                std::max(out.verify, body(p, s, 2, dimension(std::uint64_t(s.draft_tokens) + 1),
                                          batch, sm_count, false, false, verify_attention));
        }
        if (p.mtp) {
            draft_attention = std::max(
                draft_attention, attention_workspace(p, s, 1, 1, batch, sm_count, false, true));
            out.draft         = std::max(out.draft,
                                         body(p, s, 1, 1, batch, sm_count, false, true, draft_attention));
            history_attention = std::max(
                history_attention,
                history_attention_workspace(p, s, 1, dimension(s.draft_tokens), batch, sm_count));
            out.history = std::max(
                out.history, history(p, s, 1, dimension(s.draft_tokens), batch, history_attention));
        }
    }
    if (p.mtp)
        out.history = std::max(
            out.history, history(p, s, 1, chunk, 1,
                                 history_attention_workspace(p, s, 1, chunk, 1, sm_count), true));
    const int columns = s.purpose == EnginePurpose::CausalScoring
                            ? dimension(std::min(s.prefill_chunk, 1024U))
                            : dimension(std::uint64_t(s.max_concurrency) * (s.draft_tokens + 1ULL));
    out.finish        = linear(p.text.head, 1, columns);
    if (p.mtp)
        out.finish = std::max(out.finish, linear(p.mtp->head, 1, dimension(s.max_concurrency)));
    if (s.purpose == EnginePurpose::Generation) {
        out.finish =
            std::max(out.finish,
                     ops::sampling_workspace_capacity_bytes(p.public_tokens, 1, s.max_concurrency));
        if (p.local.size > 1)
            out.finish = std::max(out.finish, ops::vocabulary_candidates_workspace_capacity_bytes(
                                                  p.local.vocabulary.size, columns, 20));
        if (p.mtp)
            out.finish =
                std::max(out.finish, ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                         p.public_tokens, 1, s.draft_tokens, 1, s.max_concurrency));
    } else {
        WorkspaceLayoutBuilder a;
        (void)a.alloc(DType::BF16, {dimension(p.local.vocabulary.size), columns});
        (void)a.alloc(DType::I32, {columns});
        (void)a.alloc(DType::FP32, {columns});
        if (p.local.size > 1) {
            (void)a.alloc(DType::FP32, {3, columns});
            (void)a.alloc(DType::FP32, {3, columns, dimension(p.local.size)});
            out.finish =
                std::max(out.finish, ops::vocabulary_logprob_stats_workspace_capacity_bytes(
                                         p.local.vocabulary.size, columns));
        }
        (void)a.alloc_bytes(out.finish);
        out.finish = a.peak_bytes();
    }
    if (p.draft) {
        out.draft   = draft_workspace_capacity(p, dimension(s.draft_tokens + 1),
                                               dimension(s.max_concurrency));
        out.history = draft_context_workspace_capacity(p, s);
    }
    out.general_bytes =
        std::max({out.prefill, out.ordinary, out.verify, out.draft, out.history, out.finish});
    out.bytes = out.general_bytes;
    if (p.vision) {
        out.vision =
            vision::plan_resources(*p.config.vision, *p.vision, s.max_context, vision_budget);
        out.bytes = std::max(out.bytes, out.vision->encoder.workspace_bytes);
    }
    return out;
}

} // namespace ninfer::models::glm5next::execution
