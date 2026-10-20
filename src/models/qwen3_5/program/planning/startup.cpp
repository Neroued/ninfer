#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/dense_mlp.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/common/moe/execution.h"
#include "models/common/context/storage.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/planning.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/common/qwen/vision/vision.h"
#include "models/qwen3_5/execution/workspace.h"
#include "core/device.h"
#include "core/platform.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/gdn_control.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_accept.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {

using execution::dimension;
namespace workspace = execution::workspace;

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

ResourceLayout persistent_layout(const SequencePlanImpl& plan) {
    const auto& profile = *plan.profile;
    const auto& config  = profile.config.text;
    if (!plan.context_cache.extra_state_slots)
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    ResourceSpec spec;
    spec.decoder = {
        .full_attention_layers     = config.full_attention_layers,
        .mtp_layers                = 1,
        .capacity                  = plan.capacity,
        .kv_heads                  = dimension(config.attention->num_key_value_heads),
        .attention_head_dim        = dimension(config.attention->head_dim),
        .kv_storage                = plan.kv_storage,
        .enable_mtp                = plan.features.mtp(),
        .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
        .text_physical_page_groups = plan.main_page_groups,
        .mtp_physical_page_groups  = plan.features.mtp() ? plan.main_page_groups : 0,
    };
    spec.state_images = {
        .slot_count = checked_i32(static_cast<std::uint64_t>(plan.max_concurrency) +
                                      *plan.context_cache.extra_state_slots,
                                  "Qwen3.5 StateImage slot count exceeds int32"),
        .linear =
            {
                .layers        = config.linear_attention_layers,
                .conv_channels = config.gdn ? dimension(config.gdn->conv_channels()) : 0,
                .conv_width    = config.gdn ? dimension(config.gdn->linear_conv_kernel_dim - 1) : 0,
                .value_heads   = config.gdn ? dimension(config.gdn->linear_num_value_heads) : 0,
                .value_head_dim = config.gdn ? dimension(config.gdn->linear_value_head_dim) : 0,
                .key_head_dim   = config.gdn ? dimension(config.gdn->linear_key_head_dim) : 0,
                .conv_dtype     = DType::BF16,
            },
        .hidden = dimension(config.hidden_size),
    };
    if (plan.speculative_backend != SpeculativeBackend::None) {
        spec.replay_records = GdnReplayRecordSpec{
            .layers          = dimension(config.linear_attention_layers),
            .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
            .width           = static_cast<std::int32_t>(plan.draft_window + 1),
            .conv_channels   = config.gdn ? dimension(config.gdn->conv_channels()) : 0,
            .qk_heads        = config.gdn ? dimension(config.gdn->linear_num_key_heads) : 0,
            .value_heads     = config.gdn ? dimension(config.gdn->linear_num_value_heads) : 0,
            .key_dim         = config.gdn ? dimension(config.gdn->linear_key_head_dim) : 0,
            .value_dim       = config.gdn ? dimension(config.gdn->linear_value_head_dim) : 0,
        };
    }
    if (plan.features.masked_draft()) {
        const auto& draft              = *profile.config.draft;
        spec.state_images.dflash_local = DFlashLocalStateSpec{
            .layers   = draft.local_layer_count(),
            .capacity = draft.sliding_window.value_or(0),
            .kv_heads = dimension(draft.attention.num_key_value_heads),
            .head_dim = dimension(draft.attention.head_dim),
        };
        spec.dflash = DFlashPersistentSpec{
            .full_layers  = draft.full_layer_count(),
            .kv_heads     = dimension(draft.attention.num_key_value_heads),
            .head_dim     = dimension(draft.attention.head_dim),
            .feature_rows = dimension(config.hidden_size *
                                      static_cast<std::uint64_t>(draft.target_layer_ids.size())),
        };
    }
    spec.frame = {
        .hidden         = dimension(config.hidden_size),
        .output_rows    = dimension(config.vocab_size),
        .public_tokens  = dimension(profile.public_tokens),
        .prefill_chunk  = std::min(plan.prefill_chunk, plan.capacity),
        .batch_capacity = plan.max_concurrency,
        .draft_window   = plan.draft_window,
        .backend        = plan.speculative_backend,
        .causal_scoring = plan.causal_scoring,
        .vision         = plan.features.vision,
    };
    spec.host_context_bytes = plan.context_cache.host_capacity_bytes.value_or(0);
    return plan_resources(spec);
}

WorkspaceSpec build_workspace_plan(const SequencePlanImpl& plan) {
    const auto& profile = *plan.profile;
    const auto& config  = profile.config.text;

    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    const ops::CausalAttentionExecutionEnvelope text_envelope{1, plan.capacity};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace::text_prefill_roots(layout, config, tokens, plan.features.vision ? 3 : 0,
                                            plan.features.vision ? tokens : 0);
    };
    const auto linear_scratch = [&](WorkspaceLayoutBuilder& layout, const ops::LinearProfile& p,
                                    int first, int last) {
        scratch(layout,
                ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, first, last));
    };
    const auto add_scratch = [&](WorkspaceLayoutBuilder& layout, const ops::LinearProfile& p,
                                 int first, int last) {
        scratch(layout,
                ops::linear_add_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, first, last));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, TextPhase phase, GdnWorkspacePath path,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width,
                                 ops::CausalAttentionExecutionEnvelope envelope) {
        for (const auto& block : profile.text.layers) {
            {
                auto stage = layout.scope();
                if (const auto* attention =
                        std::get_if<execution::AttentionProfile>(&block.mixer)) {
                    (void)workspace::text_attention_projection(layout, config, last);
                    scratch(layout, execution::attention_projection_workspace_bytes(*attention,
                                                                                    first, last));
                    (void)workspace::text_attention_results(layout, config, last);
                    scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                                        {dimension(config.attention->head_dim),
                                         dimension(config.attention->num_attention_heads),
                                         dimension(config.attention->num_key_value_heads)},
                                        plan.kv_storage, envelope, batch_size, min_width, max_width,
                                        plan.multiprocessor_count));
                    add_scratch(layout, attention->output, first, last);
                } else {
                    const auto& gdn = std::get<execution::GdnProfile>(block.mixer);
                    (void)workspace::gdn_control(layout, config, last);
                    scratch(layout, ops::gdn_norm_control_proj_workspace_capacity_bytes(
                                        gdn.control, first, last));
                    (void)workspace::gdn_projection(layout, config, last);
                    if (path == GdnWorkspacePath::Snapshot) {
                        scratch(layout, execution::gdn_snapshot_workspace_bytes(
                                            gdn, batch_size, min_width, max_width));
                    } else if (path == GdnWorkspacePath::ReplayRecord) {
                        scratch(layout, execution::gdn_record_workspace_bytes(
                                            gdn, batch_size, min_width, max_width));
                    } else {
                        (void)workspace::gdn_prefill_conv(layout, config, last);
                        scratch(layout,
                                execution::gdn_projection_workspace_bytes(gdn, first, last));
                    }
                    (void)workspace::gdn_recurrent_output(layout, config, last);
                    if (path == GdnWorkspacePath::Prefill) {
                        scratch(layout,
                                ops::gated_delta_net_workspace_capacity_bytes(
                                    dimension(config.gdn->linear_num_key_heads),
                                    dimension(config.gdn->linear_num_value_heads), first, last));
                    }
                    (void)workspace::gdn_normalized_output(layout, config, last);
                    add_scratch(layout, gdn.output, first, last);
                }
            }
            auto stage = layout.scope();
            (void)workspace::post_mixer_hidden(layout, config, last);
            if (const auto* dense = std::get_if<execution::DenseProfile>(&block.ffn)) {
                scratch(layout, execution::dense_mlp_workspace_bytes(*dense, first, last,
                                                                     execution::DenseMlpUse::Text));
            } else {
                const auto& p = std::get<execution::MoeLayerProfile>(block.ffn);
                auto bytes =
                    moe::workspace_bytes(p.math, ops::MoeEpilogue::AddResidual, first, last);
                if (plan.moe) {
                    bytes = std::max(bytes, plan.moe->workspace_bytes(p.bank, first, last));
                }
                scratch(layout, bytes);
            }
        }
        if (!plan.causal_scoring) { linear_scratch(layout, profile.text.output_head, first, last); }
    };
    const auto mtp_post_mixer = [&](WorkspaceLayoutBuilder& layout, int first, int last) {
        linear_scratch(layout, profile.mtp->output, first, last);
        if (const auto* dense = std::get_if<execution::DenseProfile>(&profile.mtp->ffn)) {
            scratch(layout, execution::dense_mlp_workspace_bytes(*dense, first, last,
                                                                 execution::DenseMlpUse::Mtp));
        } else {
            scratch(layout, moe::workspace_bytes(
                                std::get<execution::MoeLayerProfile>(profile.mtp->ffn).math,
                                ops::MoeEpilogue::AddResidual, first, last));
        }
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, dimension(profile.proposal->rows), columns);
            linear_scratch(layout, profile.proposal->head, columns, columns);
        } else {
            linear_scratch(layout, profile.mtp->output_head, columns, columns);
        }
    };

    WorkspaceSpec out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    target_body(text_prefill, 1, chunk, qwen3_5::TextPhase::Prefill, GdnWorkspacePath::Prefill, 1,
                1, chunk, text_envelope);
    if (!plan.causal_scoring) {
        scratch(text_prefill,
                ops::sampling_workspace_capacity_bytes(dimension(profile.public_tokens), 1, 1));
    }
    out.text_prefill = finish(text_prefill);

    if (plan.causal_scoring) {
        WorkspaceLayoutBuilder causal_score;
        matrix(causal_score, DType::BF16, dimension(config.vocab_size),
               static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::I32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::FP32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        linear_scratch(causal_score, profile.text.output_head, 1, kCausalScoreTile);
        out.causal_score = finish(causal_score);
    }

    if (!plan.causal_scoring) {
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            WorkspaceLayoutBuilder ordinary;
            matrix(ordinary, DType::BF16, dimension(config.hidden_size), batch);
            target_body(ordinary, batch, batch, qwen3_5::TextPhase::Verify,
                        GdnWorkspacePath::Snapshot, batch, 1, 1, text_envelope);
            scratch(ordinary, ops::sampling_workspace_capacity_bytes(
                                  dimension(profile.public_tokens), batch, batch));
            out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
        }
    }

    if (plan.features.mtp()) {
        const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                  std::int32_t last) {
            (void)workspace::mtp_stem(layout, config, last, true);
            linear_scratch(layout, profile.mtp->input_projection, first, last);
        };
        const auto mtp_history = [&](std::int32_t first, std::int32_t last) {
            WorkspaceLayoutBuilder layout;
            mtp_stem(layout, first, last);
            const auto key_width = dimension(config.attention->key_width());
            matrix(layout, DType::BF16, key_width, last);
            matrix(layout, DType::BF16, key_width, last);
            scratch(layout, execution::mtp_kv_workspace_bytes(profile.mtp->projection,
                                                              *config.attention, first, last));
            matrix(layout, DType::BF16, key_width, last);
            return finish(layout);
        };
        const auto mtp_draft = [&](std::int32_t batch) {
            WorkspaceLayoutBuilder layout;
            mtp_stem(layout, batch, batch);
            (void)workspace::mtp_attention_projection(layout, config, batch);
            scratch(layout, execution::mtp_projection_workspace_bytes(profile.mtp->projection,
                                                                      batch, batch));
            (void)workspace::mtp_attention_results(layout, config, batch);
            scratch(layout,
                    ops::causal_softmax_attention_workspace_capacity_bytes(
                        {dimension(config.attention->head_dim),
                         dimension(config.attention->num_attention_heads),
                         dimension(config.attention->num_key_value_heads)},
                        plan.kv_storage, text_envelope, batch, 1, 1, plan.multiprocessor_count));
            (void)workspace::mtp_post_attention(layout, config, batch);
            mtp_post_mixer(layout, batch, batch);
            return finish(layout);
        };

        // Main, history writes, and draft steps reuse the workspace in sequence. Their inputs,
        // conditioning, and outputs live in the execution frame.
        out.mtp_prefill = std::max(out.text_prefill, mtp_history(1, chunk));
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);
            target_body(target, aggregate, aggregate, qwen3_5::TextPhase::Verify,
                        GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);

            out.ordinary_round = std::max(out.ordinary_round, mtp_history(batch, batch));
            for (std::int32_t width = 1; width <= drafts; ++width) {
                const auto columns = batch * width;
                out.mtp_round      = std::max(out.mtp_round, mtp_history(columns, columns));
            }
            WorkspaceLayoutBuilder proposal;
            proposal_scratch(proposal, batch);
            const auto accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                dimension(profile.public_tokens), drafts, drafts, batch, batch);
            out.mtp_round = std::max(
                {out.mtp_round, finish(target), mtp_draft(batch), finish(proposal), accept});
        }
    }

    if (plan.features.masked_draft()) {
        {
            const auto* draft                  = &*profile.config.draft;
            const auto dflash_context_capacity = [&](std::int32_t width, std::int32_t batch,
                                                     bool compact_input) {
                const auto tokens = width * batch;
                WorkspaceLayoutBuilder layout;
                if (batch == 1 && width > dimension(draft->sliding_window.value_or(0)))
                    matrix(layout, DType::I32, 1, 1);
                if (compact_input) {
                    matrix(layout, DType::BF16,
                           dimension(config.hidden_size *
                                     std::uint64_t(draft->target_layer_ids.size())),
                           tokens);
                }
                if (draft->dflash2.has_value()) {
                    const auto local_width =
                        std::min(width, dimension(draft->sliding_window.value_or(0)));
                    (void)workspace::dflash_context(layout, config, *draft, local_width * batch);
                    linear_scratch(layout, profile.draft->feature_projection, local_width * batch,
                                   local_width * batch);
                    scratch(layout, ops::context_kv_materialize_workspace_capacity_bytes(
                                        batch, local_width, local_width));
                    return finish(layout);
                }
                (void)workspace::dflash_context(layout, config, *draft, tokens);
                linear_scratch(layout, profile.draft->feature_projection, tokens, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace::dflash_context_layer(layout, config, *draft, tokens);
                }
                return finish(layout);
            };
            const auto dflash_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens = width * batch;
                matrix(layout, DType::BF16, dimension(config.hidden_size), tokens);
                if (draft->dflash2.has_value()) {
                    const auto prepare = [&] {
                        (void)workspace::dflash2_branch(layout, config, *draft, width, batch);
                        scratch(layout,
                                ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                                    width, width, batch, batch));
                    };
                    {
                        auto attention = layout.scope();
                        prepare();
                        matrix(layout, DType::BF16, dimension(draft->attention.query_width()),
                               tokens);
                        matrix(layout, DType::BF16, dimension(draft->attention.key_width()),
                               tokens);
                        matrix(layout, DType::BF16, dimension(draft->attention.key_width()),
                               tokens);
                        matrix(layout, DType::BF16, dimension(draft->attention.query_width()),
                               tokens);
                        scratch(layout, ops::sliding_window_attention_workspace_capacity_bytes(
                                            {dimension(draft->attention.head_dim),
                                             dimension(draft->attention.num_attention_heads),
                                             dimension(draft->attention.num_key_value_heads)},
                                            dimension(draft->sliding_window.value_or(0)),
                                            {0, plan.capacity}, width, width, batch));
                        scratch(layout,
                                ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                    dimension(draft->attention.query_width()), width, width, batch,
                                    batch));
                    }
                    {
                        auto mlp = layout.scope();
                        prepare();
                        matrix(layout, DType::BF16, dimension(draft->intermediate_size), tokens);
                        for (const auto& block : profile.draft->layers) {
                            const auto& p = block.mlp.gate_up;
                            scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                                p.qtype, p.n, p.k, p.policy, tokens, tokens));
                        }
                        scratch(
                            layout,
                            ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                dimension(draft->intermediate_size), width, width, batch, batch));
                    }
                    const auto mask_columns = drafts * batch;
                    matrix(layout, DType::BF16, dimension(config.hidden_size), mask_columns);
                    matrix(layout, DType::FP32, dimension(draft->dflash2->selector_top_k),
                           mask_columns);
                    const auto& head = plan.proposal_head == ProposalHead::Optimized
                                           ? profile.proposal->head
                                           : profile.draft->output_head;
                    scratch(layout, ops::linear_topk_workspace_capacity_bytes(
                                        head.qtype, head.n, head.k, mask_columns, mask_columns));
                    matrix(layout, DType::BF16, dimension(draft->dflash2->selector_rank),
                           mask_columns);
                    linear_scratch(layout, profile.draft->selector->hidden_projection, mask_columns,
                                   mask_columns);
                    scratch(layout, ops::candidate_selector_path_workspace_capacity_bytes(
                                        drafts, drafts, batch, batch));
                    return finish(layout);
                }
                {
                    auto attention = layout.scope();
                    (void)workspace::dflash_attention(layout, config, *draft, tokens);
                    scratch(layout,
                            std::max(ops::sliding_window_attention_workspace_capacity_bytes(
                                         {dimension(draft->attention.head_dim),
                                          dimension(draft->attention.num_attention_heads),
                                          dimension(draft->attention.num_key_value_heads)},
                                         dimension(draft->sliding_window.value_or(0)),
                                         {0, plan.capacity}, width, width, batch),
                                     ops::context_softmax_attention_workspace_capacity_bytes(
                                         {dimension(draft->attention.head_dim),
                                          dimension(draft->attention.num_attention_heads),
                                          dimension(draft->attention.num_key_value_heads)},
                                         {0, plan.capacity}, width, width, batch)));
                    for (const auto& block : profile.draft->layers) {
                        add_scratch(layout, block.output, tokens, tokens);
                    }
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace::dflash_mlp(layout, config, *draft, tokens);
                    for (const auto& block : profile.draft->layers) {
                        const auto& p = block.mlp.gate_up;
                        scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                            p.qtype, p.n, p.k, p.policy, tokens, tokens));
                    }
                    for (const auto& block : profile.draft->layers) {
                        add_scratch(layout, block.mlp.down, tokens, tokens);
                    }
                }
                matrix(layout, DType::BF16, dimension(config.hidden_size), drafts * batch);
                matrix(layout, DType::BF16, dimension(config.hidden_size), drafts * batch);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, dimension(profile.proposal->rows), drafts * batch);
                } else {
                    matrix(layout, DType::BF16, dimension(config.vocab_size), drafts * batch);
                }
                const auto& head = plan.proposal_head == ProposalHead::Optimized
                                       ? profile.proposal->head
                                       : profile.draft->output_head;
                linear_scratch(layout, head, drafts * batch, drafts * batch);
                return finish(layout);
            };

            out.dflash_context = dflash_context_capacity(chunk, 1, false);
            for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
                 ++batch) {
                const std::int32_t aggregate = verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);
                target_body(target, aggregate, aggregate, qwen3_5::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);
                const std::size_t accept =
                    draft->dflash2.has_value()
                        ? ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                              dimension(profile.public_tokens), {false}, drafts, drafts, batch,
                              batch)
                        : ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                              dimension(profile.public_tokens), drafts, drafts, batch, batch);
                const std::size_t proposal = dflash_proposal_capacity(verify, batch);
                WorkspaceLayoutBuilder accepted_hidden;
                matrix(accepted_hidden, DType::BF16, dimension(config.hidden_size), batch);
                out.dflash_round =
                    std::max({out.dflash_round, finish(target), accept, finish(accepted_hidden),
                              dflash_context_capacity(verify, batch, true), proposal});
            }
        }
    }

    out.general_capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_round, out.causal_score});
    out.capacity = out.general_capacity;
    if (plan.features.vision) {
        const std::uint32_t merged = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(plan.capacity, qwen::kMaximumVisionItemTokens));
        out.vision   = qwen::VisionContext::plan_workspace(*profile.config.vision, *profile.vision,
                                                           merged, out.general_capacity);
        out.capacity = std::max(out.capacity, out.vision->capacity_bytes);
    }
    return out;
}

void validate_target_options(const execution::ModelProfile& profile, DeviceContext& device,
                             const EngineOptions& options) {
    if (!profile.config.text.attention || profile.config.text.full_attention_layers == 0) {
        throw std::invalid_argument("Qwen3.5 Program requires at least one full-attention layer");
    }
    if (profile.options != models::load_options(options)) {
        throw std::invalid_argument(
            "loaded components do not match the requested execution options");
    }
    if (profile.draft && options.max_context > profile.config.draft->max_position_embeddings) {
        throw std::invalid_argument("max_context exceeds the selected draft position capacity");
    }
    if (options.max_context == 0 ||
        options.max_context > profile.config.text.max_position_embeddings) {
        throw std::invalid_argument("max_context exceeds the configured position capacity");
    }
    if (options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens < options.max_context) {
            throw std::invalid_argument("kv_capacity must be at least max_context");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages) {
            throw std::invalid_argument("kv_capacity cannot cover the required execution pages");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    case SpeculativeBackend::Mtp:
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMtpDecodeMaximumDrafts) {
            throw std::invalid_argument("MTP draft window must be in [1," +
                                        std::to_string(kMtpDecodeMaximumDrafts) + "]");
        }
        break;
    case SpeculativeBackend::DFlash:
    case SpeculativeBackend::DFlash2:
        if (!profile.draft || (options.speculative.backend == SpeculativeBackend::DFlash2) !=
                                  profile.config.draft->dflash2.has_value()) {
            throw std::invalid_argument(
                "selected masked draft backend is not supported by this target");
        }
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kDFlashDecodeMaximumDrafts) {
            throw std::invalid_argument("masked draft window must be in [1," +
                                        std::to_string(kDFlashDecodeMaximumDrafts) + "]");
        }
        break;
    }
    if (device.compute_capability() != 120) {
        throw std::invalid_argument("Qwen3.5 family runtime requires compute capability 12.0");
    }
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                               = std::make_unique<SequencePlanImpl>();
    impl->profile                           = inputs.profile;
    impl->moe                               = inputs.moe;
    impl->capacity                          = inputs.capacity;
    impl->main_page_groups                  = main_page_groups;
    impl->kv_capacity                       = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency                   = inputs.max_concurrency;
    impl->prefill_chunk                     = inputs.prefill_chunk;
    impl->draft_window                      = inputs.draft_window;
    impl->speculative_backend               = inputs.speculative_backend;
    impl->proposal_head                     = inputs.proposal_head;
    impl->features                          = inputs.features;
    impl->gpu_timing                        = inputs.gpu_timing;
    impl->use_cuda_graph                    = inputs.use_cuda_graph;
    impl->causal_scoring                    = inputs.causal_scoring;
    impl->multiprocessor_count              = inputs.multiprocessor_count;
    impl->context_cache                     = inputs.context_cache;
    impl->kv_storage                        = inputs.kv_storage;
    impl->persistent                        = persistent_layout(*impl);
    impl->context_cache.host_capacity_bytes = context::host_context_capacity(
        impl->context_cache.host_capacity_bytes, impl->persistent.state_images.images.image_stride);
    impl->persistent.host_context_bytes = impl->context_cache.host_capacity_bytes.value();
    impl->workspace                     = build_workspace_plan(*impl);
    impl->device_reservation_bytes =
        checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence device capacity");
    if constexpr (NINFER_SHARED_PHYSICAL_MEMORY != 0)
        impl->device_reservation_bytes =
            checked_add(impl->device_reservation_bytes, impl->persistent.pinned_bytes(),
                        "sequence unified memory capacity");
    if (impl->moe) {
        impl->device_reservation_bytes =
            checked_add(impl->device_reservation_bytes, impl->moe->description().device_bytes,
                        "MoE device capacity");
    }
    return impl;
}

} // namespace

std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(std::shared_ptr<const execution::ModelProfile> profile,
                           DeviceContext& device, const EngineOptions& options,
                           std::shared_ptr<const moe::Plan> moe) {
    validate_target_options(*profile, device, options);
    SequencePlanningInputs inputs{
        .profile         = std::move(profile),
        .moe             = std::move(moe),
        .capacity        = options.max_context,
        .max_concurrency = options.max_concurrency,
        .prefill_chunk =
            std::min(options.prefill_chunk ? options.prefill_chunk : 1024U, options.max_context),
        .draft_window         = options.speculative.draft_tokens,
        .speculative_backend  = options.speculative.backend,
        .kv_storage           = options.kv_cache,
        .proposal_head        = options.speculative.proposal_head,
        .features             = models::load_options(options),
        .gpu_timing           = options.observability.gpu_timing,
        .use_cuda_graph       = options.use_cuda_graph,
        .causal_scoring       = options.purpose == EnginePurpose::CausalScoring,
        .multiprocessor_count = device.multiprocessor_count(),
        .context_cache        = options.context_cache,
    };
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    const std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    const auto maximum_pages          = static_cast<std::uint32_t>(INT32_MAX / kPagedKVPageSize);
    inputs.context_cache.extra_state_slots = 0;

    auto planner     = std::make_unique<qwen3_5::detail::SequencePlannerImpl>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::ContextCapacityCurve{
          .primary_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_primary_page_groups             = minimum_pages,
          .maximum_primary_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes        = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_primary_page_group = 0,
          .minimum_state_slots                     = inputs.max_concurrency,
          .bytes_per_additional_state_slot =
            planner->minimum->persistent.state_images.images.image_stride,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3.5 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_primary_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
make_graph_probe_plan(std::shared_ptr<const execution::ModelProfile> profile, DeviceContext& device,
                      const EngineOptions& options, std::shared_ptr<const moe::Plan> moe) {
    auto planner = make_sequence_planner_impl(std::move(profile), device, options, std::move(moe));
    auto inputs  = std::move(planner->inputs);
    inputs.context_cache.host_capacity_bytes = 0;
    auto plan         = build_sequence_candidate(inputs, inputs.max_concurrency);
    plan->graph_probe = true;
    return plan;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            runtime::ContextCapacity capacity) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3.5 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(capacity);
    std::unique_ptr<SequencePlanImpl> plan;
    if (capacity.primary_page_groups == planner->curve.minimum_primary_page_groups &&
        capacity.device_state_slots == planner->curve.minimum_state_slots) {
        plan = std::move(planner->minimum);
    } else {
        planner->inputs.context_cache.extra_state_slots =
            capacity.device_state_slots - planner->inputs.max_concurrency;
        plan = build_sequence_candidate(planner->inputs, capacity.primary_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3.5 physical sequence layout is not affine in context capacity");
    }
    return plan;
}

} // namespace ninfer::models::qwen3_5::detail
