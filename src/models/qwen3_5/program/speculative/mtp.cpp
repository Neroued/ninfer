#include "ninfer/ops/speculative_round.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/execution/qwen4_tfdump.h"
#include "core/nvtx.h"
#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/target_logprobs.h"
#include <cuda_runtime.h>
#include <array>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

// One rank's exact-B window into its own MtpDecodeState. Both ranks are sliced by the same
// function, so a shape mistake cannot apply to one device only.
struct MtpRoundView {
    Tensor anchors;
    Tensor frontiers;
    Tensor budgets;
    Tensor current_extents;
    Tensor target_valid;
    Tensor current_drafts;
    Tensor target_rope;
    Tensor text_rows;
    Tensor mtp_rows;
    Tensor state_sources;
    Tensor state_destinations;
    Tensor rope_deltas;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_tokens;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor selected_hidden;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted;
    Tensor next_extents;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    Tensor ar_hidden;
    Tensor next_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;
    Tensor next_drafts;
    Tensor next_draft_logprobs;
    Tensor proposal_logits;
    Tensor forced_targets;
    const ops::SamplingConfig* sampling = nullptr;
};

MtpRoundView slice_mtp_frame(qwen3_5::MtpDecodeState& frame, std::int32_t batch_size) {
    MtpRoundView out;
    out.anchors             = frame.anchors.slice(0, 0, batch_size);
    out.frontiers           = frame.base_frontiers.slice(0, 0, batch_size);
    out.budgets             = frame.remaining_budgets.slice(0, 0, batch_size);
    out.current_extents     = frame.current_extents.slice(0, 0, batch_size);
    out.target_valid        = frame.target_valid_columns.slice(0, 0, batch_size);
    out.current_drafts      = frame.current_drafts.slice(1, 0, batch_size);
    out.target_rope         = frame.target_rope_positions.slice(1, 0, batch_size);
    out.text_rows           = frame.text_kv_table_rows.slice(0, 0, batch_size);
    out.mtp_rows            = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
    out.state_sources       = frame.state_source_slots.slice(0, 0, batch_size);
    out.state_destinations  = frame.state_destination_slots.slice(0, 0, batch_size);
    out.rope_deltas         = frame.rope_deltas.slice(0, 0, batch_size);
    out.verify_ids          = frame.verify_ids.slice(1, 0, batch_size);
    out.target_positions    = frame.target_positions.slice(1, 0, batch_size);
    out.target_tokens       = frame.target_argmax.slice(1, 0, batch_size);
    out.target_logits       = frame.target_logits.slice(2, 0, batch_size);
    out.target_hidden       = frame.target_hidden.slice(2, 0, batch_size);
    out.selected_hidden     = frame.target_continuation_hidden.slice(1, 0, batch_size);
    out.licensed_tokens     = frame.licensed_tokens.slice(1, 0, batch_size);
    out.licensed_counts     = frame.licensed_counts.slice(0, 0, batch_size);
    out.accepted            = frame.accepted_drafts.slice(0, 0, batch_size);
    out.next_extents        = frame.next_extents.slice(0, 0, batch_size);
    out.alignment_ids       = frame.alignment_ids.slice(1, 0, batch_size);
    out.alignment_hidden    = frame.alignment_hidden.slice(2, 0, batch_size);
    out.ar_hidden           = frame.ar_hidden.slice(1, 0, batch_size);
    out.next_hidden         = frame.next_hidden.slice(1, 0, batch_size);
    out.ar_positions        = frame.ar_positions.slice(0, 0, batch_size);
    out.ar_rope_positions   = frame.ar_rope_positions.slice(0, 0, batch_size);
    out.ar_valid_columns    = frame.ar_valid_columns.slice(0, 0, batch_size);
    out.next_drafts         = frame.next_drafts.slice(0, 0, batch_size);
    out.next_draft_logprobs = frame.next_draft_logprobs.slice(0, 0, batch_size);
    out.proposal_logits     = frame.proposal_logits.slice(1, 0, batch_size);
    if (qwen4_replay_enabled()) {
        out.forced_targets = frame.forced_targets.slice(1, 0, batch_size);
    }
    out.sampling = frame.sampling;
    return out;
}

TargetVerifyFrameView verify_view(const MtpRoundView& view, const GdnReplayRecords* records) {
    return TargetVerifyFrameView{
        .ids                     = view.verify_ids,
        .cache_positions         = view.target_positions,
        .rope_positions          = view.target_rope,
        .valid_columns           = view.target_valid,
        .kv_table_rows           = view.text_rows,
        .state_source_slots      = view.state_sources,
        .state_destination_slots = view.state_destinations,
        .target_hidden           = view.target_hidden,
        .target_logits           = view.target_logits,
        .target_tokens           = view.target_tokens,
        .drafts                  = view.current_drafts,
        .current_extents         = view.current_extents,
        .frontiers               = view.frontiers,
        .anchors                 = view.anchors,
        .licensed_tokens         = view.licensed_tokens,
        .licensed_counts         = view.licensed_counts,
        .accepted_drafts         = view.accepted,
        .selected_hidden         = view.selected_hidden,
        .forced_targets          = view.forced_targets,
        .replay_records          = records,
        .sampling                = view.sampling,
    };
}

void prepare_next_round(const MtpRoundView& view, std::int32_t max_context, cudaStream_t stream) {
    Tensor alignment_ids     = view.alignment_ids;
    Tensor next_extents      = view.next_extents;
    Tensor ar_positions      = view.ar_positions;
    Tensor ar_rope_positions = view.ar_rope_positions;
    Tensor ar_valid_columns  = view.ar_valid_columns;
    ops::mtp_prepare_next_round(view.verify_ids, view.anchors, view.accepted, view.frontiers,
                                view.budgets, view.licensed_counts, view.rope_deltas, alignment_ids,
                                next_extents, ar_positions, ar_rope_positions, ar_valid_columns,
                                max_context, stream);
}

// The bridge over both ranks (program/tp2/mtp_bridge_tp2.inc).
void mtp_bridge_and_propose_tp2(PrefillContext& state, const Tensor& next_token,
                                const Tensor& previous_hidden, const Tensor& peer_previous_hidden,
                                std::int32_t position, std::span<const std::int32_t> rope_position,
                                bool build_proposal, const Tensor* next_embedding);

} // namespace

void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, const Tensor* peer_previous_hidden,
                            std::int32_t position, std::span<const std::int32_t> rope_position,
                            bool build_proposal, const Tensor* next_embedding) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    if ((state.execution.tp != nullptr) != (peer_previous_hidden != nullptr)) {
        throw std::logic_error("MTP bridge needs rank 1's retained hidden exactly at tp 2");
    }
    if (state.execution.tp != nullptr) {
        mtp_bridge_and_propose_tp2(state, next_token, previous_hidden, *peer_previous_hidden,
                                   position, rope_position, build_proposal, next_embedding);
        return;
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden = state.execution.io.mtp->ar_hidden;
    Tensor logits     = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0     = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    // Qwen4Exp runs a text position (the same on the three M-RoPE axes) one-axis, as its decode.
    const bool one_axis = state.execution.parameters.qwen4_mtp.has_value() &&
                          rope_position[1] == rope_position[0] &&
                          rope_position[2] == rope_position[0];
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, one_axis ? 1 : 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position_view.bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::CausalAttentionExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent >
            static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.execution.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                 envelope, next_hidden, logits, next_draft);
        CUDA_CHECK(::ninfer::memcpy_device_async(
            state.execution.io.mtp->ar_hidden.data, next_hidden.data,
            state.execution.io.mtp->ar_hidden.bytes(), state.execution.device.stream));
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpCausalAttentionEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3_5::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        // Rows of the carried hidden (Qwen4Exp: the wide residual).
        const std::int32_t hidden_size = frame.ar_hidden.ne[0];
        const auto max_context         = static_cast<std::int32_t>(state.text_cache.max_context());
        cudaStream_t stream            = state.execution.device.stream;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   stream));
        if (const Qwen4Runtime* qwen4 = state.execution.qwen4;
            qwen4 != nullptr && qwen4->decode_ple_rows.data != nullptr &&
            qwen4->decode_ple_staged == nullptr) {
            // Qwen4Exp: the n-gram rows of the verified window, gathered on the host before the
            // launch (one upload node). Deferred rows (decode_ple_staged) are uploaded by the PLE
            // layer itself once the host has gathered them after the launch.
            const Tensor rows = qwen4->decode_ple_rows.slice(1, 0, width * batch_size);
            CUDA_CHECK(cudaMemcpyAsync(rows.data, qwen4->decode_ple_host, rows.bytes(),
                                       cudaMemcpyHostToDevice, stream));
            if (const TpExecution* tp_rows = state.execution.tp;
                tp_rows != nullptr && tp_rows->qwen4 != nullptr &&
                tp_rows->qwen4->decode_ple_rows.data != nullptr) {
                // tp 2: rank 1 uploads the same staging (replicated PLE).
                const Tensor peer_rows =
                    tp_rows->qwen4->decode_ple_rows.slice(1, 0, width * batch_size);
                const DeviceContext& rank1_ple = *tp_rows->execution->dev[1];
                const ScopedCurrentDevice scope(rank1_ple.device);
                CUDA_CHECK(cudaMemcpyAsync(peer_rows.data, qwen4->decode_ple_host,
                                           peer_rows.bytes(), cudaMemcpyHostToDevice,
                                           rank1_ple.stream));
            }
        }

        // Tensor-parallel width 2: rank 1 runs the round from its own upload of the same ingress
        // record. Everything else it needs is derived from that record by the same Ops on rank 1,
        // except the acceptance results, which only rank 0 computes and which are copied to rank
        // 1 (accepted counts in target_verify_accept; anchors, frontiers and licensed counts
        // below). Rank 1 never samples or accepts, so the record's sampling configs are never
        // read there.
        const TpExecution* tp      = state.execution.tp;
        const DeviceContext* rank1 = tp != nullptr ? &*tp->execution->dev[1] : nullptr;
        std::optional<MtpRoundView> peer;
        if (tp != nullptr) {
            if (state.peer_frame == nullptr || state.peer_continuation_hidden_store == nullptr) {
                throw std::logic_error("tensor-parallel MTP decode requires rank 1's frame");
            }
            const ScopedCurrentDevice scope(rank1->device);
            CUDA_CHECK(cudaMemcpyAsync(state.peer_frame->ingress.data, &state.host_ingress,
                                       sizeof(qwen3_5::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                       rank1->stream));
            peer.emplace(slice_mtp_frame(*state.peer_frame, batch_size));
        }

        TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                         {}, state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, &state.mtp_cache, tp);
        card.set_qwen4_runtime(state.execution.qwen4);
        const MtpRoundView round = slice_mtp_frame(frame, batch_size);
        {
            Tensor verify_ids       = round.verify_ids;
            Tensor target_positions = round.target_positions;
            ops::speculative_prepare_verify_inputs(round.anchors, round.current_drafts,
                                                   round.frontiers, round.current_extents,
                                                   verify_ids, target_positions, stream);
        }
        if (peer) {
            const ScopedCurrentDevice scope(rank1->device);
            Tensor verify_ids       = peer->verify_ids;
            Tensor target_positions = peer->target_positions;
            ops::speculative_prepare_verify_inputs(peer->anchors, peer->current_drafts,
                                                   peer->frontiers, peer->current_extents,
                                                   verify_ids, target_positions, rank1->stream);
        }
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp,
                                           static_cast<std::uint64_t>(width) * batch_size);
            if (peer) {
                target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                     verify_view(round, state.execution.replay_records),
                                     verify_view(*peer, tp->replay_records),
                                     *state.peer_continuation_hidden_store,
                                     envelopes.target_verify);
            } else {
                target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                     verify_view(round, state.execution.replay_records),
                                     envelopes.target_verify);
            }
        }

        {
            nvtx::ScopedRange draft_range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp,
                                          static_cast<std::uint64_t>(k) * batch_size);
            // From here on the round interleaves cross-device copies (acceptance to rank 1,
            // proposal logits to rank 0) with the MTP head's all-reduces.
            std::optional<ops::PeerEvents::StagedScope> staged_draft;
            if (peer && tp->staged_draft_collectives) { staged_draft.emplace(*tp->events); }
            prepare_next_round(round, max_context, stream);
            if (peer) {
                // Rank 0's acceptance updated anchors, frontiers and licensed counts in place;
                // rank 1 pulls them and prepares its copy of the next round from the same inputs.
                const ops::PeerEvents& events = *tp->events;
                CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), stream));
                const ScopedCurrentDevice scope(rank1->device);
                CUDA_CHECK(cudaStreamWaitEvent(rank1->stream, events.inputs_ready(0), 0));
                for (const auto& [destination, source] :
                     {std::pair{peer->anchors, round.anchors},
                      std::pair{peer->frontiers, round.frontiers},
                      std::pair{peer->licensed_counts, round.licensed_counts}}) {
                    CUDA_CHECK(::ninfer::memcpy_device_async(destination.data, source.data,
                                                             source.bytes(), rank1->stream));
                }
                prepare_next_round(*peer, max_context, rank1->stream);
            }

            Tensor draft0 = round.next_drafts.slice(1, 0, 1).view({batch_size});
            // T41: each proposal step's softmax log-probability of its argmax over the proposal
            // vocabulary (the full head, rank 0 holding the gathered logits at tp 2, or T43's
            // draft shortlist), written by the proposal itself.
            std::array<Tensor, kMtpDecodeMaximumDrafts> logprob_views;
            const auto draft_logprob = [&](std::int32_t step) -> Tensor* {
                if (!state.draft_logprobs) { return nullptr; }
                logprob_views[static_cast<std::size_t>(step)] =
                    round.next_draft_logprobs.slice(1, step, 1).view({batch_size});
                return &logprob_views[static_cast<std::size_t>(step)];
            };
            if (peer) {
                // The MTP head is split at tp 2: rank 0 contracts the embedding of the alignment
                // ids and rank 1 its copy of the target hidden.
                const TextContext::RankTensors target_hidden{round.target_hidden,
                                                             peer->target_hidden};
                const TextContext::RankTensors target_positions{round.target_positions,
                                                                peer->target_positions};
                const TextContext::RankTensors target_rope{round.target_rope, peer->target_rope};
                const TextContext::RankTensors licensed_counts{round.licensed_counts,
                                                               peer->licensed_counts};
                const TextContext::RankTensors mtp_rows{round.mtp_rows, peer->mtp_rows};
                const TextContext::RankTensors alignment_hidden{round.alignment_hidden,
                                                                peer->alignment_hidden};
                card.mtp_forward_decode_batch(round.alignment_ids, target_hidden, target_positions,
                                              target_rope, licensed_counts, mtp_rows,
                                              envelopes.batch, alignment_hidden);
            } else {
                Tensor alignment_hidden = round.alignment_hidden;
                card.mtp_forward_decode_batch(round.alignment_ids, round.target_hidden,
                                              round.target_positions, round.target_rope,
                                              round.licensed_counts, round.mtp_rows,
                                              envelopes.batch, alignment_hidden);
            }
            {
                Tensor ar_hidden = round.ar_hidden;
                ops::speculative_select_accepted_hidden(round.alignment_hidden, round.accepted,
                                                        ar_hidden, stream);
            }
            if (peer) {
                const ScopedCurrentDevice scope(rank1->device);
                Tensor ar_hidden = peer->ar_hidden;
                ops::speculative_select_accepted_hidden(peer->alignment_hidden, peer->accepted,
                                                        ar_hidden, rank1->stream);
            }

            if (peer) {
                const TextContext::RankTensors ar_hidden{round.ar_hidden, peer->ar_hidden};
                Tensor proposal_logits = round.proposal_logits;
                card.mtp_propose_batch(ar_hidden, proposal_logits, draft0, draft_logprob(0));
            } else {
                Tensor proposal_logits = round.proposal_logits;
                card.mtp_propose_batch(round.ar_hidden, proposal_logits, draft0, draft_logprob(0));
            }
            for (std::uint32_t step = 0; step + 1 < k; ++step) {
                const auto s        = static_cast<std::int32_t>(step);
                Tensor previous     = round.next_drafts.slice(1, s, 1).view({1, batch_size});
                Tensor next         = round.next_drafts.slice(1, s + 1, 1).view({batch_size});
                Tensor position     = round.ar_positions.slice(1, s, 1).view({1, batch_size});
                Tensor rope         = round.ar_rope_positions.slice(1, s, 1).view({1, batch_size});
                Tensor valid        = round.ar_valid_columns.slice(1, s, 1).view({batch_size});
                Tensor hidden_batch = round.ar_hidden.view({hidden_size, 1, batch_size});
                Tensor next_hidden_batch = round.next_hidden.view({hidden_size, 1, batch_size});
                if (peer) {
                    const TextContext::RankTensors hidden{
                        hidden_batch, peer->ar_hidden.view({hidden_size, 1, batch_size})};
                    const TextContext::RankTensors positions{
                        position, peer->ar_positions.slice(1, s, 1).view({1, batch_size})};
                    const TextContext::RankTensors ropes{
                        rope, peer->ar_rope_positions.slice(1, s, 1).view({1, batch_size})};
                    const TextContext::RankTensors valids{
                        valid, peer->ar_valid_columns.slice(1, s, 1).view({batch_size})};
                    const TextContext::RankTensors mtp_rows{round.mtp_rows, peer->mtp_rows};
                    const TextContext::RankTensors next_hidden{
                        next_hidden_batch, peer->next_hidden.view({hidden_size, 1, batch_size})};
                    card.mtp_forward_decode_batch(previous, hidden, positions, ropes, valids,
                                                  mtp_rows, envelopes.ar[step], next_hidden);
                    const TextContext::RankTensors proposal_hidden{round.next_hidden,
                                                                   peer->next_hidden};
                    Tensor proposal_logits = round.proposal_logits;
                    card.mtp_propose_batch(proposal_hidden, proposal_logits, next,
                                           draft_logprob(s + 1));
                    const ScopedCurrentDevice scope(rank1->device);
                    CUDA_CHECK(
                        ::ninfer::memcpy_device_async(peer->ar_hidden.data, peer->next_hidden.data,
                                                      peer->ar_hidden.bytes(), rank1->stream));
                } else {
                    card.mtp_forward_decode_batch(previous, hidden_batch, position, rope, valid,
                                                  round.mtp_rows, envelopes.ar[step],
                                                  next_hidden_batch);
                    Tensor proposal_logits = round.proposal_logits;
                    card.mtp_propose_batch(round.next_hidden, proposal_logits, next,
                                           draft_logprob(s + 1));
                }
                CUDA_CHECK(::ninfer::memcpy_device_async(
                    round.ar_hidden.data, round.next_hidden.data, round.ar_hidden.bytes(), stream));
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_5::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   stream));
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpCausalAttentionEnvelopes envelopes,
                              DecodeGraphDefinition& definition) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpCausalAttentionEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
}

} // namespace ninfer::models::qwen3_5::execution

#include "models/qwen3_5/program/tp2/mtp_bridge_tp2.inc"
