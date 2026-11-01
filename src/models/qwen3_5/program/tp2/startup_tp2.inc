// Translation-unit fragment of src/models/qwen3_5/program/planning/startup.cpp, included once at
// its end (after its closing namespace): the two-device (tp 2) workspace plan and the tp 2 option
// checks, sharing startup.cpp's private reserve_* helpers. Not a header. Our file; see
// docs/maintainer/upstream-merge.md.

#include "models/qwen3_5/execution/tp2/draft_split.h"
#include "ninfer/ops/tp2/sliding_window_attention.h"

namespace ninfer::models::qwen3_5::detail {
namespace {

std::size_t dflash2_candidates_workspace_tp2(const SequencePlanImpl& plan, std::int32_t columns) {
    const auto& parameters                    = *plan.parameters;
    const execution::ProposalParameters& head = *parameters.proposal;
    if (!head.split()) {
        return ops::linear_topk_workspace_capacity_bytes(head.head.weight.qtype, head.head.weight.n,
                                                         head.head.weight.k, columns, columns);
    }
    // Rank 0 ranks its block and rank 1 its own; both hold the broadcast hidden, the candidates
    // and their staging (execution::dflash2_candidates_split).
    return execution::dflash2_candidates_split_workspace_bytes(
        head, dimension(parameters.model.config().text.hidden_size), columns, true);
}

bool dflash2_drafter_split_tp2(const SequencePlanImpl& plan) {
    return plan.parameters != nullptr && execution::dflash2_drafter_split(*plan.parameters);
}

// One rank's proposal of the split drafter over `batch` rows of `width` masked-block columns, in
// the allocation order of execution/tp2/draft_tp2.inc (propose_dflash2_batch_tp2). Both ranks run
// the same layers at their shard shapes and rank with their head blocks; the selecting rank (rank 1
// by default, load/sharding.cpp) also projects and selects, so the plan includes that step and
// bounds both ranks.
std::size_t dflash_proposal_workspace_split(const SequencePlanImpl& plan, std::int32_t width,
                                            std::int32_t batch) {
    const auto& parameters     = *plan.parameters;
    const auto& config         = parameters.model.config().text;
    const auto& draft          = *parameters.model.config().draft;
    const auto drafts          = static_cast<std::int32_t>(plan.draft_window);
    const std::int32_t hidden  = dimension(config.hidden_size);
    const std::int32_t columns = width * batch;
    const std::int32_t mask    = drafts * batch;
    const ops::AttentionHeadGeometry heads{128, 16, 4};
    WorkspaceLayoutBuilder layout;
    reserve_matrix(layout, DType::BF16, hidden, columns); // residual
    reserve_matrix(layout, DType::BF16, hidden, columns); // all-reduce staging
    const auto prepare = [&] {
        (void)workspace::dflash2_branch(layout, config, draft, width, batch);
        reserve_scratch(layout, ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                                    width, width, batch, batch));
    };
    for (const auto& block : parameters.draft->layers) {
        {
            auto attention = layout.scope();
            prepare();
            reserve_matrix(layout, DType::BF16, 128 * heads.query_heads, columns);
            reserve_matrix(layout, DType::BF16, 128 * heads.kv_heads, columns);
            reserve_matrix(layout, DType::BF16, 128 * heads.kv_heads, columns);
            reserve_matrix(layout, DType::BF16, 128 * heads.query_heads, columns);
            reserve_scratch(layout,
                            ops::sliding_window_attention_head_block_workspace_capacity_bytes(
                                heads, dimension(draft.sliding_window), {0, plan.capacity}, width,
                                width, batch));
            reserve_matrix(layout, DType::BF16, hidden, columns);
            reserve_linear(layout, block.output, columns, columns);
        }
        {
            auto mlp = layout.scope();
            prepare();
            const auto& gate_up = block.mlp.gate_up;
            if (execution::dflash2_split_mlp_fused(gate_up)) {
                reserve_matrix(layout, DType::BF16, gate_up.weight.n / 2, columns);
                reserve_scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                            gate_up.weight.qtype, gate_up.weight.n,
                                            gate_up.weight.k, gate_up.policy, columns, columns));
            } else {
                reserve_matrix(layout, DType::BF16, gate_up.weight.n, columns);
                reserve_linear(layout, gate_up, columns, columns);
                reserve_matrix(layout, DType::BF16, gate_up.weight.n / 2, columns);
            }
            reserve_matrix(layout, DType::BF16, hidden, columns);
            reserve_linear(layout, block.mlp.down, columns, columns);
        }
    }
    reserve_matrix(layout, DType::BF16, hidden, mask);
    reserve_matrix(layout, DType::FP32, dimension(draft.dflash2.selector_top_k), mask);
    const execution::ProposalParameters& head = *parameters.proposal;
    reserve_scratch(
        layout,
        head.split()
            ? execution::dflash2_candidates_split_workspace_bytes(head, hidden, mask, false)
            : ops::linear_topk_workspace_capacity_bytes(head.head.weight.qtype, head.head.weight.n,
                                                        head.head.weight.k, mask, mask));
    // The selecting rank's projection (rank 1's with the candidate selector there).
    const execution::Parameters& selecting =
        parameters.draft->selector || plan.peer_parameters == nullptr ? parameters
                                                                      : *plan.peer_parameters;
    if (!selecting.draft || !selecting.draft->selector) {
        throw std::logic_error("tensor-parallel DFlash2 planning found no candidate selector");
    }
    reserve_matrix(layout, DType::BF16, dimension(draft.dflash2.selector_rank), mask);
    reserve_linear(layout, selecting.draft->selector->hidden_projection, mask, mask);
    reserve_scratch(layout, ops::candidate_selector_path_workspace_capacity_bytes(drafts, drafts,
                                                                                  batch, batch));
    return layout.peak_bytes(1);
}

// One rank's transient workspace of a dense model's Qwen3.5 heads at tensor-parallel width 2: the
// call roots (ids, positions, residual, all-reduce staging) and vocabulary-split logits of each
// phase, the MTP prompt alignment, forwards and proposals (execution/tp2/text_tp2.inc), and the
// DFlash2 drafter (execution/draft.cpp); the Text layers are the Qwen4 forward's
// (build_qwen4_workspace_plan). Per-layer MTP stages use the rank's share of the config and the
// rank's shard Parameters; the call roots keep the replicated hidden width. Both ranks allocate
// this capacity: where the ranks differ (rank 0's logits gather staging, sampling, token
// embedding, optimized proposal head and drafter, rank 1's last hidden column), the plan covers
// both. It is built from rank 0's Parameters, whose shard shapes equal rank 1's except for the
// heads and the drafter only rank 0 holds.
WorkspacePlan build_qwen3_5_head_workspace_plan_tp2(const SequencePlanImpl& plan) {
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;
    const TextConfig shard = execution::shard_text_config(config, plan.tp);
    const bool mtp         = plan.speculative_backend == SpeculativeBackend::Mtp;
    if (mtp && (!parameters.mtp || !shard.attention)) {
        throw std::invalid_argument("tensor-parallel MTP workspace requires the MTP parameters");
    }

    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto hidden = dimension(config.hidden_size);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    const ops::CausalAttentionExecutionEnvelope text_envelope{1, plan.capacity};
    const std::int32_t public_tokens = dimension(parameters.model.resources().public_token_count);

    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };
    const auto attention_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t batch_size,
                                       std::int32_t min_width, std::int32_t max_width) {
        reserve_scratch(layout,
                        ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(shard.attention->head_dim),
                             dimension(shard.attention->num_attention_heads),
                             dimension(shard.attention->num_key_value_heads)},
                            plan.kv_storage, text_envelope, batch_size, min_width, max_width));
    };
    // Vocabulary-split head over `columns` final hidden columns: this rank's rows, rank 0's
    // staging of rank 1's rows, and the column-parallel projection's scratch.
    const auto split_logits = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        const execution::LinearParameters& head = parameters.text.output_head;
        const std::int32_t peer_rows            = dimension(config.vocab_size) - head.weight.n;
        auto call                               = layout.scope();
        (void)workspace::tp_logits(layout, head.weight.n, peer_rows, columns, true);
        reserve_scratch(layout,
                        execution::output_head_split_workspace_bytes(head, columns, columns));
    };
    // The MTP proposal over `columns` hidden columns: rank 0's optimized head alone, that head
    // split by vocabulary with its argmax candidates, or the vocabulary-split output head.
    const auto proposal = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        auto call = layout.scope();
        if (plan.proposal_head == ProposalHead::Optimized && parameters.proposal->split()) {
            (void)workspace::tp_proposal_argmax(layout, parameters.proposal->head.weight.n,
                                                columns);
            reserve_linear(layout, parameters.proposal->head, columns, columns);
        } else if (plan.proposal_head == ProposalHead::Optimized) {
            reserve_matrix(layout, DType::BF16, dimension(parameters.proposal->rows), columns);
            reserve_linear(layout, parameters.proposal->head, columns, columns);
        } else {
            split_logits(layout, columns);
        }
    };
    // TextContext::mtp_forward_core_tp2 over `tokens` columns, as `batch_size` rows of `width`.
    const auto mtp_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              std::int32_t batch_size, std::int32_t width) {
        const execution::MtpParameters& p = *parameters.mtp;
        auto core                         = layout.scope();
        (void)workspace::tp_call_roots(layout, config, tokens);
        (void)workspace::mtp_stem_split(layout, config, tokens, true);
        reserve_linear(layout, p.input_projection, tokens, tokens);
        (void)workspace::mtp_attention_projection(layout, shard, tokens);
        reserve_scratch(
            layout, execution::mtp_projection_split_workspace_bytes(p.projection, tokens, tokens));
        (void)workspace::mtp_attention_results(layout, shard, tokens);
        attention_scratch(layout, batch_size, width, width);
        (void)workspace::mtp_post_attention(layout, config, tokens);
        reserve_linear(layout, p.output, tokens, tokens);
        reserve_scratch(layout, execution::mtp_ffn_split_workspace_bytes(p.ffn, tokens, tokens));
    };
    // TextContext::mtp_prefill_chunk_tp2 over a `tokens`-column chunk, the prompt's last one when
    // `last_chunk`.
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                       bool last_chunk) {
        const execution::MtpParameters& p = *parameters.mtp;
        const auto key_width              = dimension(shard.attention->key_width());
        const auto query_width            = dimension(shard.attention->query_width());
        auto call                         = layout.scope();
        (void)workspace::tp_call_roots(layout, config, tokens);
        if (last_chunk) {
            for (int i = 0; i < 3; ++i) { reserve_matrix(layout, DType::BF16, hidden, 1); }
        }
        {
            auto bulk = layout.scope();
            (void)workspace::mtp_stem_split(layout, config, tokens, true);
            reserve_linear(layout, p.input_projection, tokens, tokens);
            reserve_matrix(layout, DType::BF16, key_width, tokens);
            reserve_matrix(layout, DType::BF16, key_width, tokens);
            reserve_scratch(layout, execution::mtp_kv_split_workspace_bytes(
                                        p.projection, *shard.attention, tokens, tokens));
            reserve_matrix(layout, DType::BF16, key_width, tokens);
        }
        if (!last_chunk) { return; }
        reserve_matrix(layout, DType::BF16, query_width, 1);
        reserve_matrix(layout, DType::BF16, query_width, 1);
        reserve_scratch(layout, execution::mtp_query_gate_split_workspace_bytes(
                                    p.projection, *shard.attention, 1, 1));
        reserve_matrix(layout, DType::BF16, query_width, 1);
        reserve_matrix(layout, DType::BF16, query_width, 1);
        reserve_matrix(layout, DType::BF16, hidden, 1);
        reserve_matrix(layout, DType::BF16, hidden, 1);
        if (plan.features.vision) { reserve_matrix(layout, DType::I32, 1, 3); }
        attention_scratch(layout, 1, 1, 1);
        reserve_linear(layout, p.output, 1, 1);
        reserve_scratch(layout, execution::mtp_ffn_split_workspace_bytes(p.ffn, 1, 1));
        proposal(layout, 1);
    };

    WorkspacePlan out;
    // The text prefill chunk up to its sampled token; the MTP prompt alignment continues from it.
    const auto text_prefill = [&](WorkspaceLayoutBuilder& layout) {
        // A multimodal chunk carries three-axis RoPE positions and its visual scatter indices on
        // both ranks; a text chunk continuing a multimodal prefix carries a one-axis RoPE offset.
        (void)workspace::text_prefill_roots(layout, config, chunk, plan.features.vision ? 3 : 1,
                                            plan.features.vision ? chunk : 0);
        (void)workspace::tp_call_roots(layout, config, chunk);
        reserve_matrix(layout, DType::BF16, hidden, 1);
        split_logits(layout, 1);
        reserve_scratch(layout, ops::sampling_workspace_capacity_bytes(public_tokens, 1, 1));
    };
    {
        WorkspaceLayoutBuilder layout;
        text_prefill(layout);
        out.text_prefill = finish(layout);
    }
    // Speculative verification of `batch` rows of K+1 columns: the call roots and the gathered
    // logits in rank 0's frame (the Text forward is build_qwen4_workspace_plan's).
    const auto verification = [&](std::int32_t batch) {
        const std::int32_t aggregate = batch * verify;
        WorkspaceLayoutBuilder target;
        reserve_matrix(target, DType::BF16, hidden, aggregate);
        (void)workspace::tp_call_roots(target, config, aggregate);
        {
            auto logits             = target.scope();
            const std::int32_t rows = parameters.text.output_head.weight.n;
            (void)workspace::tp_logits(target, rows, dimension(config.vocab_size) - rows, aggregate,
                                       true);
            reserve_scratch(target, execution::output_head_split_workspace_bytes(
                                        parameters.text.output_head, aggregate, aggregate));
        }
        return finish(target);
    };
    if (plan.causal_scoring) {
        // ProgramImpl::causal_score's flush of one score tile: rank 0 holds the gathered logits,
        // the targets, the logprobs, its head half and the staging of rank 1's; rank 1 its head
        // half. Its hidden copy lives in its persistent score_hidden.
        const execution::LinearParameters& head = parameters.text.output_head;
        const std::int32_t rows1                = dimension(config.vocab_size) - head.weight.n;
        const auto tile                         = static_cast<std::int32_t>(kCausalScoreTile);
        const std::size_t head_scratch =
            execution::output_head_split_workspace_bytes(head, 1, tile);
        WorkspaceLayoutBuilder rank0;
        reserve_matrix(rank0, DType::BF16, dimension(config.vocab_size), tile);
        reserve_matrix(rank0, DType::I32, 1, tile);
        reserve_matrix(rank0, DType::FP32, 1, tile);
        (void)workspace::tp_logits(rank0, head.weight.n, rows1, tile, true);
        reserve_scratch(rank0, head_scratch);
        WorkspaceLayoutBuilder rank1;
        (void)workspace::tp_logits(rank1, rows1, head.weight.n, tile, false);
        reserve_scratch(rank1, head_scratch);
        out.causal_score = std::max(finish(rank0), finish(rank1));
    } else if (plan.speculative_backend == SpeculativeBackend::None) {
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            WorkspaceLayoutBuilder layout;
            reserve_matrix(layout, DType::BF16, hidden, batch);
            (void)workspace::tp_call_roots(layout, config, batch);
            reserve_matrix(layout, DType::BF16, hidden, batch);
            split_logits(layout, batch);
            reserve_scratch(layout,
                            ops::sampling_workspace_capacity_bytes(public_tokens, batch, batch));
            out.ordinary_round = std::max(out.ordinary_round, finish(layout));
        }
    } else if (mtp) {
        {
            WorkspaceLayoutBuilder layout;
            text_prefill(layout);
            reserve_matrix(layout, DType::I32, 1, chunk);
            if (plan.features.vision) {
                // Rank 0's composed MTP input embedding and its shifted visual scatter indices.
                reserve_matrix(layout, DType::BF16, hidden, chunk);
                (void)workspace::visual_scatter_indices(layout, chunk);
            }
            {
                auto final_chunk = layout.scope();
                mtp_prefill_chunk(layout, chunk, true);
            }
            // The prompt proposal steps after the final chunk: the next hidden, the rope
            // position, one MTP column and its proposal.
            for (std::int32_t step = 1; step < drafts; ++step) {
                auto ar_step = layout.scope();
                reserve_matrix(layout, DType::BF16, hidden, 1);
                reserve_matrix(layout, DType::I32, 1, 1);
                mtp_core(layout, 1, 1, 1);
                proposal(layout, 1);
            }
            out.mtp_prefill = finish(layout);
        }
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder alignment;
            mtp_core(alignment, aggregate, batch, verify);
            WorkspaceLayoutBuilder ar;
            mtp_core(ar, batch, batch, 1);
            WorkspaceLayoutBuilder proposals;
            proposal(proposals, batch);
            const std::size_t accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    public_tokens, drafts, drafts, batch, batch);
            out.mtp_round = std::max({out.mtp_round, verification(batch), finish(alignment),
                                      finish(ar), finish(proposals), accept});
        }
    } else {
        // DFlash2: rank 0 appends the prefill features to the drafter's context after each chunk's
        // text schedule, and a round runs rank 0's context append and proposal, the two-device
        // verification and rank 0's sparse acceptance, each from a reset arena.
        out.dflash_context = dflash_context_workspace(plan, chunk, 1, false);
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            out.dflash_round = std::max({out.dflash_round, verification(batch),
                                         dflash_accept_workspace(plan, batch),
                                         dflash_context_workspace(plan, verify, batch, true),
                                         dflash2_drafter_split_tp2(plan)
                                             ? dflash_proposal_workspace_split(plan, verify, batch)
                                             : dflash_proposal_workspace(plan, verify, batch)});
        }
    }
    out.general_capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_round, out.causal_score});
    out.capacity = out.general_capacity;
    if (plan.features.vision) {
        // The rank that holds the tower encodes each item after its general prefix, as on one
        // device; the other rank only receives the merged embeddings.
        out.vision = execution::VisionContext::plan_workspace(
            *parameters.model.config().vision, *plan.vision_parameters->vision,
            vision_item_tokens(plan), out.general_capacity);
        out.vision_receiver   = execution::VisionContext::plan_receiver(*out.vision);
        out.capacity          = std::max(out.capacity, out.vision->capacity_bytes);
        out.receiver_capacity = std::max(out.general_capacity, out.vision_receiver->capacity_bytes);
    }
    return out;
}

// The tensor-parallel checks of validate_target_options(), in their original order.
void validate_tensor_parallel_options(const execution::Parameters& parameters,
                                      const execution::Parameters* peer_parameters,
                                      const EngineOptions& options) {
    if (options.tp != 1 && options.tp != execution::kTensorParallelWidth) {
        throw std::invalid_argument("Qwen3.5 tensor-parallel width must be 1 or 2");
    }
    if (options.tp != 1) {
        if (parameters.model.device_count() != options.tp || parameters.device != 0 ||
            peer_parameters == nullptr || &peer_parameters->model != &parameters.model ||
            peer_parameters->device != 1) {
            throw std::invalid_argument("tensor-parallel planning requires rank 0 and rank 1 "
                                        "Parameters of one two-device Model");
        }
        // The full output head is split by vocabulary rows across the ranks, and the DFlash2
        // candidate ranking has a vocabulary-split form only for the optimized head.
        if (options.speculative.backend == SpeculativeBackend::DFlash2 &&
            options.speculative.proposal_head != ProposalHead::Optimized) {
            throw std::invalid_argument(
                "tensor-parallel DFlash2 requires the optimized proposal head (--lm-head-draft)");
        }
        if (options.kv_cache != KvCacheStorage::BFloat16 &&
            options.kv_cache != KvCacheStorage::Int8Group64 &&
            options.kv_cache != KvCacheStorage::Nvfp4Group16) {
            throw std::invalid_argument(
                "tensor-parallel attention supports only bf16, int8 and nvfp4 KV caches");
        }
        // A candidate selector on rank 1 selects among the candidates of the split proposal head,
        // which both ranks merge; a head whole on rank 0 leaves rank 1 without them.
        if (options.speculative.backend == SpeculativeBackend::DFlash2 && peer_parameters->draft &&
            peer_parameters->draft->selector &&
            (!parameters.proposal || !parameters.proposal->split())) {
            throw std::invalid_argument(
                "tensor-parallel DFlash2: the candidate selector on rank 1 requires the "
                "vocabulary-split proposal head (set NINFER_TP_SELECTOR=primary)");
        }
        // Rejects extents the width does not divide.
        (void)execution::shard_text_config(parameters.model.config().text, options.tp);
    }
}

} // namespace
} // namespace ninfer::models::qwen3_5::detail
