#include "models/qwen4_exp/execution/workspace.h"
#include "ninfer/ops/gdn_control.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_accept.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::execution {
namespace {
std::size_t linear(const ops::LinearProfile& p, int first, int last) {
    return ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, first, last);
}

std::size_t body(const ModelProfile& p, const EngineOptions& options, int first, int last,
                 int batch, bool prefill, bool mtp, int sm_count, const moe::Plan* offload) {
    WorkspaceLayoutBuilder a;
    const auto columns = last * batch, minimum = first * batch;
    (void)workspace::roots(a, p.config.text, columns);
    const auto scratch = [&](std::size_t bytes) {
        auto scope = a.scope();
        (void)a.alloc_bytes(bytes);
    };
    if (mtp) {
        auto scope = a.scope();
        (void)workspace::mtp_stem(a, p.config.text, columns);
        scratch(linear(p.mtp->fc_embedding, minimum, columns));
        scratch(linear(p.mtp->fc_hidden,
                       dimension(std::uint64_t(p.config.text.hc.streams) * minimum),
                       dimension(std::uint64_t(p.config.text.hc.streams) * columns)));
    } else {
        auto scope = a.scope();
        (void)workspace::ple(a, p.config.text, columns, prefill);
        scratch(linear(p.text.ple, minimum, columns));
    }
    const auto layer = [&](const BlockProfile& layer) {
        scratch(ops::hc_mix_workspace_capacity_bytes(layer.attention_hc, minimum, columns));
        {
            auto scope = a.scope();
            if (const auto* g = std::get_if<GdnProfile>(&layer.mixer)) {
                (void)workspace::gdn(a, p.config.text.gdn, columns, prefill);
                scratch(
                    ops::gdn_control_proj_workspace_capacity_bytes(g->control, minimum, columns));
                if (prefill) {
                    scratch(ops::linear_split_workspace_capacity_bytes(g->input, g->input_rows,
                                                                       first, last));
                    scratch(ops::gated_delta_net_workspace_capacity_bytes(
                        dimension(p.config.text.gdn.key_heads),
                        dimension(p.config.text.gdn.value_heads), first, last));
                } else if (last == 1) {
                    scratch(ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                        g->input, batch, 1, 1));
                } else {
                    scratch(ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                        g->input, batch, first, last));
                }
                scratch(linear(g->output, minimum, columns));
            } else {
                const auto& q = std::get<QsaProfile>(layer.mixer);
                (void)workspace::qsa(a, p.config.text.qsa, columns, prefill || mtp);
                scratch(qsa_projection_workspace_bytes(q, minimum, columns));
                scratch(ops::qsa_indexer_workspace_capacity_bytes(first, last, batch,
                                                                  {options.max_context}));
                scratch(ops::qsa_sparse_gqa_workspace_capacity_bytes(
                    options.kv_cache, dimension(p.config.text.qsa.query_heads),
                    dimension(p.config.text.qsa.kv_heads), first, last, batch,
                    {options.max_context}, sm_count));
                scratch(linear(q.output, minimum, columns));
            }
        }
        scratch(ops::hc_mix_workspace_capacity_bytes(layer.moe_hc, minimum, columns));
        scratch(moe::workspace_bytes(layer.moe.math, ops::MoeEpilogue::Write, minimum, columns));
        if (!mtp && offload) {
            scratch(offload->workspace_bytes(layer.moe.bank, minimum, columns));
        }
    };
    if (mtp)
        layer(p.mtp->layer);
    else
        for (const auto& block : p.text.layers) layer(block);
    scratch(ops::hc_mix_workspace_capacity_bytes(mtp ? p.mtp->final_hc : p.text.final_hc, minimum,
                                                 columns));
    if (!mtp && !prefill && options.purpose == EnginePurpose::Generation)
        scratch(linear(p.text.head, minimum, columns));
    return a.peak_bytes();
}
} // namespace

WorkspacePlan plan_workspace(const ModelProfile& p, const EngineOptions& o, std::uint32_t chunk,
                             std::int32_t sm_count, const moe::Plan* offload) {
    WorkspacePlan out;
    out.prefill = body(p, o, 1, chunk, 1, true, false, sm_count, offload);
    for (std::uint32_t b = 1; b <= o.max_concurrency; ++b) {
        out.ordinary = std::max(out.ordinary, body(p, o, 1, 1, b, false, false, sm_count, offload));
        if (p.mtp) {
            out.verify = std::max(out.verify, body(p, o, 2, o.speculative.draft_tokens + 1, b,
                                                   false, false, sm_count, offload));
            out.mtp    = std::max(out.mtp, body(p, o, 1, o.speculative.draft_tokens + 1, b, false,
                                                true, sm_count, nullptr));
        }
    }
    if (p.mtp) out.mtp = std::max(out.mtp, body(p, o, 1, chunk, 1, true, true, sm_count, nullptr));
    const auto head_columns = o.purpose == EnginePurpose::CausalScoring
                                  ? std::min(chunk, 1024U)
                                  : o.max_concurrency * (o.speculative.draft_tokens + 1);
    out.finish              = linear(p.text.head, 1, head_columns);
    if (o.purpose == EnginePurpose::Generation) {
        out.finish =
            std::max(out.finish,
                     ops::sampling_workspace_capacity_bytes(p.public_tokens, 1, o.max_concurrency));
        if (p.mtp)
            out.finish =
                std::max(out.finish,
                         ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                             p.public_tokens, 1, o.speculative.draft_tokens, 1, o.max_concurrency));
    }
    if (o.purpose == EnginePurpose::CausalScoring) {
        WorkspaceLayoutBuilder a;
        (void)workspace::score(a, p.config.text.vocabulary, head_columns);
        (void)a.alloc_bytes(out.finish);
        out.finish = a.peak_bytes();
    }
    out.general_bytes = std::max({out.prefill, out.ordinary, out.verify, out.mtp, out.finish});
    out.bytes         = out.general_bytes;
    if (p.vision) {
        out.vision = qwen::VisionContext::plan_workspace(
            *p.config.vision, *p.vision,
            std::min<std::uint64_t>(o.max_context, qwen::kMaximumVisionItemTokens),
            out.general_bytes);
        out.bytes = out.vision->capacity_bytes;
    }
    return out;
}
} // namespace ninfer::models::qwen4_exp::execution
