#include "models/qwen4_exp/execution/workspace.h"
#include "core/platform.h"
#include "ninfer/ops/gdn_control.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_accept.h"
#include "ninfer/ops/vocabulary.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::execution {
SequenceParallelPolicy sequence_parallel_policy(std::uint32_t ranks) {
    return NINFER_TARGET_SPARK && ranks == 2 ? SequenceParallelPolicy{2, 129}
                                             : SequenceParallelPolicy{};
}

namespace {
std::size_t linear(const ops::LinearProfile& p, int first, int last) {
    return ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, first, last);
}

std::size_t body(const ModelProfile& p, const EngineOptions& options, int first, int last,
                 int batch, bool prefill, bool mtp, int sm_count, const moe::Plan* offload) {
    const auto policy = sequence_parallel_policy(p.local.size);
    if (prefill && !mtp && policy.minimum_columns && first < policy.minimum_columns &&
        last >= policy.minimum_columns)
        return std::max(
            body(p, options, first, policy.minimum_columns - 1, batch, prefill, mtp, sm_count,
                 offload),
            body(p, options, policy.minimum_columns, last, batch, prefill, mtp, sm_count, offload));
    WorkspaceLayoutBuilder a;
    const auto columns = dimension(std::uint64_t(last) * batch),
               minimum = dimension(std::uint64_t(first) * batch);
    const auto partition =
        prefill && !mtp ? policy.partition(columns, p.local.rank) : TokenPartition(columns);
    const auto local_minimum =
        partition.split() ? policy.partition(minimum, p.local.rank).count : minimum;
    (void)workspace::roots(a, p.config.text, columns, partition.padded);
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
    const auto layer = [&](const BlockProfile& layer, bool local) {
        const int lo = local ? local_minimum : minimum;
        const int hi = local ? partition.count : columns;
        scratch(ops::hc_mix_workspace_capacity_bytes(layer.attention_hc, lo, hi));
        {
            auto scope = a.scope();
            if (const auto* g = std::get_if<GdnProfile>(&layer.mixer)) {
                (void)workspace::gdn(a, p.local.gdn, columns, prefill);
                scratch(
                    ops::gdn_control_proj_workspace_capacity_bytes(g->control, minimum, columns));
                if (prefill) {
                    scratch(ops::linear_split_workspace_capacity_bytes(g->input, g->input_rows,
                                                                       first, last));
                    scratch(ops::gated_delta_net_workspace_capacity_bytes(
                        dimension(p.local.gdn.key_heads), dimension(p.local.gdn.value_heads), first,
                        last));
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
                (void)workspace::qsa(a, p.local.qsa, columns, prefill || mtp);
                scratch(qsa_projection_workspace_bytes(q, minimum, columns));
                scratch(ops::qsa_indexer_workspace_capacity_bytes(first, last, batch,
                                                                  {options.max_context}));
                scratch(ops::qsa_sparse_gqa_workspace_capacity_bytes(
                    options.kv_cache, dimension(p.local.qsa.query_heads),
                    dimension(p.local.qsa.kv_heads), first, last, batch, {options.max_context},
                    sm_count));
                scratch(linear(q.output, minimum, columns));
            }
        }
        scratch(ops::hc_mix_workspace_capacity_bytes(layer.moe_hc, lo, hi));
        scratch(moe::workspace_bytes(layer.moe.math, ops::MoeEpilogue::Write, minimum, columns,
                                     layer.moe.subset));
        if (!mtp && offload) {
            scratch(offload->workspace_bytes(layer.moe.bank, minimum, columns, !prefill));
        }
    };
    if (mtp)
        layer(p.mtp->layer, false);
    else
        for (std::size_t i = 0; i < p.text.layers.size(); ++i)
            layer(p.text.layers[i], partition.split() && i >= p.config.text.ple.layer);
    scratch(ops::hc_mix_workspace_capacity_bytes(mtp ? p.mtp->final_hc : p.text.final_hc,
                                                 local_minimum, partition.count));
    if (!mtp && !prefill && options.purpose == EnginePurpose::Generation)
        scratch(linear(p.text.head, minimum, columns));
    return a.peak_bytes();
}

std::size_t history(const ModelProfile& p, int first, int last, int batch,
                    bool partitioned = false) {
    WorkspaceLayoutBuilder a;
    const auto columns = dimension(std::uint64_t(last) * batch),
               minimum = dimension(std::uint64_t(first) * batch);
    const auto& mtp    = *p.mtp;
    const TokenPartition partition(columns, partitioned ? p.local.size : 1,
                                   partitioned ? p.local.rank : 0);
    const int local_minimum = TokenPartition(minimum, partition.ranks, partition.rank).count;
    (void)workspace::history_roots(a, p.config.text, partition.count, partition.padded);
    const auto scratch = [&](std::size_t bytes) {
        auto scope = a.scope();
        (void)a.alloc_bytes(bytes);
    };
    {
        auto scope = a.scope();
        (void)workspace::mtp_stem(a, p.config.text, partition.count, columns);
        scratch(linear(mtp.fc_embedding, local_minimum, partition.count));
        scratch(linear(mtp.fc_hidden,
                       dimension(std::uint64_t(p.config.text.hc.streams) * local_minimum),
                       dimension(std::uint64_t(p.config.text.hc.streams) * partition.count)));
    }
    scratch(ops::hc_mix_workspace_capacity_bytes(mtp.history.mix, local_minimum, partition.count));
    {
        auto scope = a.scope();
        (void)workspace::qsa_history(a, p.local.qsa, columns);
        scratch(ops::linear_split_workspace_capacity_bytes(
            {mtp.history.input, {}}, mtp.history.output_rows, minimum, columns));
    }
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
            out.verify  = std::max(out.verify, body(p, o, 2, o.speculative.draft_tokens + 1, b,
                                                    false, false, sm_count, offload));
            out.mtp     = std::max(out.mtp, body(p, o, 1, 1, b, false, true, sm_count, nullptr));
            out.history = std::max(out.history, history(p, 1, o.speculative.draft_tokens, b));
        }
    }
    if (p.mtp) {
        const auto policy    = sequence_parallel_policy(p.local.size);
        const int replicated = policy.minimum_columns
                                   ? std::min(dimension(chunk), policy.minimum_columns - 1)
                                   : dimension(chunk);
        out.history          = std::max(out.history, history(p, 1, replicated, 1));
        if (policy.minimum_columns && chunk >= std::uint32_t(policy.minimum_columns))
            out.history =
                std::max(out.history, history(p, policy.minimum_columns - 1, chunk, 1, true));
    }
    const auto head_columns =
        o.purpose == EnginePurpose::CausalScoring
            ? std::min(chunk, 1024U)
            : dimension(std::uint64_t(o.max_concurrency) * (o.speculative.draft_tokens + 1));
    out.finish = linear(p.text.head, 1, head_columns);
    if (p.mtp) out.finish = std::max(out.finish, linear(p.mtp->head, 1, o.max_concurrency));
    if (o.purpose == EnginePurpose::Generation) {
        if (p.local.size > 1)
            out.finish = std::max(out.finish, ops::vocabulary_candidates_workspace_capacity_bytes(
                                                  p.local.vocabulary, head_columns, 20));
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
        (void)workspace::score(a, p.local.vocabulary, head_columns);
        if (p.local.size > 1) {
            (void)a.alloc(DType::FP32, {3, static_cast<int>(head_columns)});
            (void)a.alloc(DType::FP32,
                          {3, static_cast<int>(head_columns), dimension(p.local.size)});
            out.finish =
                std::max(out.finish, ops::vocabulary_logprob_stats_workspace_capacity_bytes(
                                         p.local.vocabulary, head_columns));
        }
        (void)a.alloc_bytes(out.finish);
        out.finish = a.peak_bytes();
    }
    out.general_bytes =
        std::max({out.prefill, out.ordinary, out.verify, out.mtp, out.history, out.finish});
    out.bytes = out.general_bytes;
    if (p.vision) {
        out.vision = qwen::VisionContext::plan_resources(*p.config.vision, *p.vision, o.max_context,
                                                         o.vision_memory_budget_bytes);
        out.bytes  = std::max(out.bytes, out.vision->encoder.workspace_bytes);
    }
    return out;
}
} // namespace ninfer::models::qwen4_exp::execution
