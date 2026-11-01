#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/execution/linear.h"
#include <cmath>
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/execution/workspace.h"
#include "models/qwen3_5/execution/tp2/draft_split.h"

#include "core/nvtx.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/tp2/context_kv_materialize.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <optional>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

using detail::DFlashPersistentState;

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr || !state.execution.parameters.draft.has_value()) {
        throw std::logic_error("masked draft schedule requires its weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    {
        require_dflash_state(state);
        const auto& target = state.execution.parameters.model.config().text;
        const auto& config = *state.execution.parameters.model.config().draft;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = std::span<const std::uint32_t>(config.target_layer_ids),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    {
        const auto& target = state.execution.parameters.model.config().text;
        const auto& config = *state.execution.parameters.model.config().draft;
        return DFlashFeatureSink{
            .batch_features      = &dflash_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const std::uint32_t>(config.target_layer_ids),
        };
    }
}

template <class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    {
        const auto& target         = state.execution.parameters.model.config().text;
        const auto& config         = *state.execution.parameters.model.config().draft;
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
            commit_counts.ne[0] != batch || lanes.dtype != DType::I32 || lanes.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window = batch == 1 && width > dimension(config.sliding_window);
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset =
            replace_local_window ? width - dimension(config.sliding_window) : 0;
        const int local_width = replace_local_window ? dimension(config.sliding_window) : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window ? static_cast<std::uint32_t>(dimension(config.sliding_window))
                                 : envelope.min_count,
            replace_local_window ? static_cast<std::uint32_t>(dimension(config.sliding_window))
                                 : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->local_append_count;
            ops::set_i32_scalar(local_counts, dimension(config.sliding_window),
                                state.execution.device.stream);
        }

        const Tensor input =
            replace_local_window ? features.slice(1, local_offset, local_width) : features;
        const auto roots =
            workspace::dflash_context(state.execution.work, target, config, local_width * batch);
        Tensor projected = roots.projected;
        Tensor context   = roots.normalized;
        project(input.view(
                    {dimension(target.hidden_size * std::uint64_t(config.target_layer_ids.size())),
                     local_width * batch}),
                state.execution.parameters.draft->feature_projection, projected,
                state.execution.work, state.execution.device.stream);
        ops::rmsnorm(projected, state.execution.parameters.draft->context_norm, config.rms_norm_eps,
                     false, context, state.execution.device.stream);

        std::array<ops::ContextKVMaterializeLayerView, ops::kContextKVMaterializeLayers> layers;
        if (config.num_hidden_layers != layers.size() || config.rms_norm_eps != 1e-6F ||
            config.rope_theta != 1e7F) {
            throw std::invalid_argument(
                "masked draft context: config is outside the fused materialization contract");
        }
        for (int layer = 0; layer < dimension(config.num_hidden_layers); ++layer) {
            const auto& weights = state.execution.parameters.draft->layers[layer];
            layers[layer]       = {weights.context_key.weight, weights.context_value.weight,
                                   weights.key_norm, dflash_state(state).local_layer(layer)};
        }
        const Tensor local_positions =
            replace_local_window ? positions.slice(0, local_offset, local_width) : positions;
        // The split drafter at tp 2 holds a rank's four KV heads (execution/tp2/draft_split.h).
        const auto materialize =
            layers.front().key_weight.n == dimension(config.attention.key_width())
                ? &ops::context_kv_materialize
                : &ops::context_kv_materialize_head_block;
        materialize(context.view({dimension(target.hidden_size), local_width, batch}),
                    local_positions, local_counts, lanes, layers,
                    {local_envelope.min_count, local_envelope.max_count}, state.execution.work,
                    state.execution.device.stream);
    }
}

void prepare_dynamic_branch(ExecutionCore& execution, const Tensor& residual, const Tensor& norm,
                            float eps, const DynamicConvParameters& weights,
                            workspace::DFlash2BranchRoots& branch) {
    auto scope      = execution.work.scope();
    const int width = residual.ne[1], batch = residual.ne[2];
    WorkspaceArena scratch(execution.work.alloc_bytes(
        ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(width, width, batch,
                                                                           batch)));
    ops::rmsnorm_dynamic_grouped_conv_prepare(
        residual, norm, eps, weights.base_kernel, weights.kernel_projection.weight, branch.prepared,
        branch.finish_delta, scratch, execution.device.stream);
}

void finish_dynamic_branch(ExecutionCore& execution, const Tensor& input,
                           const LinearParameters& projection, const DynamicConvParameters& weights,
                           const Tensor& finish_delta, Tensor& residual) {
    auto scope      = execution.work.scope();
    const int width = input.ne[1], batch = input.ne[2];
    WorkspaceArena scratch(
        execution.work.alloc_bytes(ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
            input.ne[0], width, width, batch, batch)));
    ops::linear_dynamic_grouped_conv_add(input, projection.weight, weights.base_kernel,
                                         finish_delta, residual, scratch, execution.device.stream);
}

void propose_dflash2_batch(DFlashBatchContext& state, qwen3_5::DFlashDecodeState& frame, int batch,
                           int k, DFlashEnvelopes envelopes) {
    const auto& target = state.execution.parameters.model.config().text;
    const auto& config = *state.execution.parameters.model.config().draft;
    if (config.rms_norm_eps != 1e-6F || config.rope_theta != 1e7F) {
        throw std::invalid_argument("DFlash2: fused norm/RoPE requires epsilon=1e-6 and theta=1e7");
    }

    const int width                = k + 1;
    const int columns              = width * batch;
    const int mask_columns         = k * batch;
    auto& work                     = state.execution.work;
    const cudaStream_t stream      = state.execution.device.stream;
    const DraftParameters& weights = *state.execution.parameters.draft;
    Tensor anchors                 = frame.anchors.slice(0, 0, batch);
    Tensor frontiers               = frame.execution_frontiers.slice(0, 0, batch);
    Tensor valid_columns           = frame.proposal_valid_columns.slice(0, 0, batch);
    Tensor state_slots             = frame.state_destination_slots.slice(0, 0, batch);
    Tensor ids                     = frame.proposal_ids.slice(1, 0, batch);
    Tensor positions               = frame.proposal_positions.slice(1, 0, batch);
    work.reset();
    ops::prepare_masked_block(anchors, frontiers, valid_columns, dimension(config.mask_token_id),
                              ids, positions, stream);
    Tensor residual      = work.alloc(DType::BF16, {dimension(target.hidden_size), width, batch});
    Tensor flat_residual = residual.view({dimension(target.hidden_size), columns});
    ops::embedding(ids.view({columns}), state.execution.parameters.text.token_embedding,
                   flat_residual, stream);
    for (std::size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
        const auto& layer = weights.layers[layer_index];
        nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash, layer_index);
        {
            auto scope  = work.scope();
            auto branch = workspace::dflash2_branch(work, target, config, width, batch);
            prepare_dynamic_branch(state.execution, residual, layer.input_norm, config.rms_norm_eps,
                                   *layer.attention_conv, branch);
            Tensor query      = work.alloc(DType::BF16, {dimension(config.attention.head_dim),
                                                         dimension(config.attention.num_attention_heads),
                                                         width, batch});
            Tensor key        = work.alloc(DType::BF16, {dimension(config.attention.head_dim),
                                                         dimension(config.attention.num_key_value_heads),
                                                         width, batch});
            Tensor value      = work.alloc(DType::BF16, {dimension(config.attention.head_dim),
                                                         dimension(config.attention.num_key_value_heads),
                                                         width, batch});
            Tensor query_flat = query.view({dimension(config.attention.query_width()), columns});
            Tensor key_flat   = key.view({dimension(config.attention.key_width()), columns});
            Tensor value_flat = value.view({dimension(config.attention.key_width()), columns});
            ops::attn_input_proj(branch.prepared.view({dimension(target.hidden_size), columns}),
                                 layer.query_key_value.weight, query_flat, key_flat, value_flat,
                                 stream);
            ops::rmsnorm_rope(positions, layer.query_norm, layer.key_norm, query, key, stream);
            Tensor attention = work.alloc(
                DType::BF16, {dimension(config.attention.head_dim),
                              dimension(config.attention.num_attention_heads), width, batch});
            ops::sliding_window_attention(
                query, key, value, positions, valid_columns, state_slots,
                {dimension(config.attention.head_dim),
                 dimension(config.attention.num_attention_heads),
                 dimension(config.attention.num_key_value_heads)},
                dimension(config.sliding_window),
                static_cast<float>(1.0 / std::sqrt(double(config.attention.head_dim))),
                state.dflash.local_layer(static_cast<std::uint32_t>(layer_index)), envelopes.local,
                work, attention, stream);
            finish_dynamic_branch(
                state.execution,
                attention.view({dimension(config.attention.query_width()), width, batch}),
                layer.output, *layer.attention_conv, branch.finish_delta, residual);
        }
        {
            auto scope  = work.scope();
            auto branch = workspace::dflash2_branch(work, target, config, width, batch);
            prepare_dynamic_branch(state.execution, residual, layer.post_attention_norm,
                                   config.rms_norm_eps, *layer.mlp_conv, branch);
            Tensor intermediate =
                work.alloc(DType::BF16, {dimension(config.intermediate_size), width, batch});
            Tensor intermediate_flat =
                intermediate.view({dimension(config.intermediate_size), columns});
            project_swiglu(branch.prepared.view({dimension(target.hidden_size), columns}),
                           layer.mlp.gate_up, intermediate_flat, work, stream);
            finish_dynamic_branch(state.execution, intermediate, layer.mlp.down, *layer.mlp_conv,
                                  branch.finish_delta, residual);
        }
    }
    Tensor hidden = work.alloc(DType::BF16, {dimension(target.hidden_size), mask_columns});
    ops::rmsnorm_pack_tail(residual, weights.final_norm, hidden, stream);
    Tensor candidates = frame.candidate_ids.slice(2, 0, batch);
    Tensor ids_flat   = candidates.view({dimension(config.dflash2.selector_top_k), mask_columns});
    Tensor scores =
        work.alloc(DType::FP32, {dimension(config.dflash2.selector_top_k), mask_columns});
    if (state.execution.proposal_head == ProposalHead::Full) {
        ops::linear_topk(hidden, state.execution.parameters.draft->output_head.weight,
                         dimension(state.execution.parameters.model.resources().public_token_count),
                         ids_flat, scores, work, stream);
    } else {
        const auto& head      = *state.execution.parameters.proposal;
        const TpExecution* tp = state.execution.tp;
        if (tp != nullptr && head.split()) {
            // Vocabulary-split head (tp 2): both ranks rank their blocks, rank 0 merges.
            dflash2_candidates_split(
                {hidden, hidden},
                {&head, tp->parameters->proposal ? &*tp->parameters->proposal : nullptr}, true,
                ids_flat, scores, {&work, tp->work}, *tp->execution, *tp->events);
        } else if (head.token_ids) {
            ops::linear_topk(hidden, head.head.weight, *head.token_ids, ids_flat, scores, work,
                             stream);
        } else {
            ops::linear_topk(
                hidden, head.head.weight,
                dimension(state.execution.parameters.model.resources().public_token_count),
                ids_flat, scores, work, stream);
        }
    }
    Tensor projected =
        work.alloc(DType::BF16, {dimension(config.dflash2.selector_rank), mask_columns});
    project(hidden, weights.selector->hidden_projection, projected, work, stream);
    Tensor drafts     = frame.draft_tokens.slice(1, 0, batch);
    Tensor proposal_q = frame.proposal_q.slice(2, 0, batch);
    ops::candidate_selector_path(
        candidates, scores.view({dimension(config.dflash2.selector_top_k), k, batch}),
        projected.view({dimension(config.dflash2.selector_rank), k, batch}), anchors,
        weights.selector->predecessor_codebook, weights.selector->successor_codebook, frontiers,
        frame.sampling, drafts, proposal_q, work, stream);
    work.reset();
}

// One rank's exact-B verification window into its own DFlashDecodeState; both ranks are sliced by
// the same function. Only the primary (rank 0) view carries the proposal distribution, the
// sampling configs and the feature sink: the drafter and acceptance are rank 0's.
TargetVerifyFrameView dflash_verify_view(qwen3_5::DFlashDecodeState& frame, std::int32_t batch_size,
                                         const GdnReplayRecords* replay_records, bool primary,
                                         DFlashFeatureSink* sink) {
    return TargetVerifyFrameView{
        .ids                     = frame.verify_ids.slice(1, 0, batch_size),
        .cache_positions         = frame.verify_positions.slice(1, 0, batch_size),
        .rope_positions          = frame.target_rope_positions.slice(1, 0, batch_size),
        .valid_columns           = frame.target_valid_columns.slice(0, 0, batch_size),
        .kv_table_rows           = frame.text_kv_table_rows.slice(0, 0, batch_size),
        .state_source_slots      = frame.state_source_slots.slice(0, 0, batch_size),
        .state_destination_slots = frame.state_destination_slots.slice(0, 0, batch_size),
        .target_hidden           = frame.target_hidden.slice(2, 0, batch_size),
        .target_logits           = frame.target_logits.slice(2, 0, batch_size),
        .target_tokens           = frame.target_argmax.slice(1, 0, batch_size),
        .drafts                  = frame.draft_tokens.slice(1, 0, batch_size),
        .current_extents         = frame.proposal_extents.slice(0, 0, batch_size),
        .candidate_ids           = primary && frame.candidate_ids.data != nullptr
                                       ? frame.candidate_ids.slice(2, 0, batch_size)
                                       : Tensor{},
        .proposal_q              = primary && frame.proposal_q.data != nullptr
                                       ? frame.proposal_q.slice(2, 0, batch_size)
                                       : Tensor{},
        .frontiers               = frame.execution_frontiers.slice(0, 0, batch_size),
        .anchors                 = frame.anchors.slice(0, 0, batch_size),
        .licensed_tokens         = frame.licensed_tokens.slice(1, 0, batch_size),
        .licensed_counts         = frame.licensed_counts.slice(0, 0, batch_size),
        .accepted_drafts         = frame.accepted_drafts.slice(0, 0, batch_size),
        .selected_hidden         = frame.target_continuation_hidden.slice(1, 0, batch_size),
        .replay_records          = replay_records,
        .sampling                = primary ? frame.sampling : nullptr,
        .feature_sink            = sink,
    };
}

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DFlashEnvelopes envelopes,
                              ops::CausalAttentionExecutionEnvelope target_envelope) {
    return [&state, batch_size, k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        qwen3_5::DFlashDecodeState& frame = state.frame;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        const cudaStream_t stream         = state.execution.device.stream;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   stream));

        // Tensor-parallel width 2: rank 1 receives its own upload of the same ingress record,
        // takes part in the split drafter (or only verifies, with the drafter on rank 0 alone), and
        // derives its verification inputs from the round's drafts; acceptance is rank 0's, and
        // target_verify_accept copies its accepted counts to rank 1. Rank 1 samples only the
        // proposal, when the candidate selector is its (the record's temperature and seed).
        const TpExecution* tp      = state.execution.tp;
        const DeviceContext* rank1 = tp != nullptr ? &*tp->execution->dev[1] : nullptr;
        if (tp != nullptr) {
            if (state.peer_frame == nullptr || state.peer_continuation_hidden_store == nullptr ||
                tp->replay_records == nullptr) {
                throw std::logic_error("tensor-parallel DFlash decode requires rank 1's frame");
            }
            const ScopedCurrentDevice scope(rank1->device);
            CUDA_CHECK(cudaMemcpyAsync(state.peer_frame->ingress.data, &state.host_ingress,
                                       sizeof(qwen3_5::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                       rank1->stream));
        }

        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts     = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents            = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor active_lanes       = frame.active_lanes.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor append_positions   = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts      = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.verify_positions.slice(1, 0, batch_size);

        // The draft phase's own collectives stay on the staged copies while the pinned-host mailbox
        // carries the target forward below: this phase interleaves cross-device copies with its
        // layer sums, and under WSL2 such a copy stalls behind the peer's spinning exchange kernel
        // (issue #1). A full DFlash2 round hung the mailbox; see
        // ProgramImpl::degrade_peer_mailbox().
        std::optional<ops::PeerEvents::StagedScope> staged_draft;
        if (tp != nullptr && tp->staged_draft_collectives) { staged_draft.emplace(*tp->events); }
        if (tp != nullptr && tp->dflash != nullptr) {
            // The split drafter (tp 2): both ranks append and propose (execution/tp2/).
            dflash2_draft_round_tp2(state, batch_size, k, envelopes);
        } else {
            state.execution.work.reset();
            Tensor compact_features = state.execution.work.alloc(
                DType::BF16,
                {dimension(state.execution.parameters.draft->feature_projection.weight.k), width,
                 batch_size});
            ops::prepare_ragged_prefix(dflash_state(state).pending_features, active_lanes,
                                       context_starts, frontiers, compact_features,
                                       append_positions, append_counts, stream);
            append_context_impl(state, compact_features, append_positions, append_counts,
                                state_destinations, envelopes.append);

            nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                             static_cast<std::uint64_t>(k + 1U) * batch_size);
            propose_dflash2_batch(state, frame, batch_size, k, envelopes);
        }
        // Tensor-parallel width 2: the drafts reach the other rank, which derives its verification
        // inputs from them (execution/tp2/draft_tp2.inc).
        if (tp != nullptr) { dflash2_share_drafts_tp2(state, batch_size); }
        staged_draft.reset();
        ops::speculative_prepare_verify_inputs(anchors, drafts, frontiers, extents, verify_ids,
                                               target_positions, stream);

        TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                         {}, state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, nullptr, tp);
        // The Program's Text-forward runtime.
        card.set_qwen4_runtime(state.execution.qwen4);
        DFlashFeatureSink sink =
            batch_feature_sink_impl(state, active_lanes, valid_columns, width, batch_size);
        // The split drafter captures rank 1's own features from its copy of the residual.
        const Tensor peer_lanes =
            tp != nullptr ? state.peer_frame->active_lanes.slice(0, 0, batch_size) : Tensor{};
        const Tensor peer_valid =
            tp != nullptr ? state.peer_frame->target_valid_columns.slice(0, 0, batch_size)
                          : Tensor{};
        std::optional<DFlashFeatureSink> peer_sink;
        if (tp != nullptr && tp->dflash != nullptr) {
            peer_sink =
                dflash_peer_batch_sink(*tp->dflash, peer_lanes, peer_valid, width, batch_size);
        }
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeDFlashTarget, nvtx::Category::DFlash,
                                           static_cast<std::uint64_t>(width) * batch_size);
            const TargetVerifyFrameView primary =
                dflash_verify_view(frame, batch_size, state.execution.replay_records, true, &sink);
            if (tp != nullptr) {
                target_verify_accept(
                    state.execution, state.continuation_hidden_store, card, primary,
                    dflash_verify_view(*state.peer_frame, batch_size, tp->replay_records, false,
                                       peer_sink ? &*peer_sink : nullptr),
                    *state.peer_continuation_hidden_store, target_envelope);
            } else {
                target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                     primary, target_envelope);
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_5::DFlashDecodeEgress), cudaMemcpyDeviceToHost,
                                   stream));
    };
}

} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl(state, features, positions, commit_counts, lanes, envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl(state, features, positions, commit_counts, lanes, envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::CausalAttentionExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    capture_graph(state, definition, body);
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes,
                         ops::CausalAttentionExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::models::qwen3_5::execution

#include "models/qwen3_5/execution/tp2/draft_tp2.inc"
