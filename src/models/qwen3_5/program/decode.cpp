#include <string_view>
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/execution/qwen4_runtime.h"
#include "models/qwen3_5/execution/qwen4_tfdump.h"
#include "core/nvtx.h"
#include "core/device.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

auto ordinary_batch_body(OrdinaryBatchContext& state, std::int32_t batch_size,
                         ops::CausalAttentionExecutionEnvelope envelope) {
    return [&state, batch_size, envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3_5::OrdinaryDecodeState& ordinary = state.frame;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));
        if (state.execution.tp != nullptr) {
            // Rank 1 reads its own upload of the same record: the same tokens, positions, KV rows
            // (its execution tables mirror rank 0's row for row) and StateImage slots. It never
            // samples, so the record's rank-0 sampling configs are never read there.
            if (state.peer_frame == nullptr) {
                throw std::logic_error("tensor-parallel decode requires rank 1's ordinary frame");
            }
            const DeviceContext& rank1 = *state.execution.tp->execution->dev[1];
            const ScopedCurrentDevice scope(rank1.device);
            CUDA_CHECK(cudaMemcpyAsync(state.peer_frame->ingress.data, &state.host_ingress,
                                       sizeof(qwen3_5::OrdinaryDecodeIngress),
                                       cudaMemcpyHostToDevice, rank1.stream));
        }

        TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                         {}, state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, nullptr, state.execution.tp);
        card.set_qwen4_runtime(state.execution.qwen4);
        if (const Qwen4Runtime* qwen4 = state.execution.qwen4;
            qwen4 != nullptr && qwen4->decode_ple_rows.data != nullptr &&
            qwen4->decode_ple_staged == nullptr) {
            // The round's n-gram rows, gathered on the host before the launch (one upload node).
            // Deferred rows (decode_ple_staged) are uploaded by the PLE layer itself.
            const Tensor rows = qwen4->decode_ple_rows.slice(1, 0, batch_size);
            CUDA_CHECK(cudaMemcpyAsync(rows.data, qwen4->decode_ple_host, rows.bytes(),
                                       cudaMemcpyHostToDevice, state.execution.device.stream));
            if (state.execution.tp != nullptr) {
                // tp 2: rank 1 uploads the same host rows into its own copy (replicated PLE).
                const Qwen4Runtime* peer = state.execution.tp->qwen4;
                if (peer == nullptr || peer->decode_ple_rows.data == nullptr) {
                    throw std::logic_error("Qwen4Exp tp 2 decode has no rank 1 PLE rows");
                }
                const DeviceContext& rank1 = *state.execution.tp->execution->dev[1];
                const ScopedCurrentDevice scope(rank1.device);
                const Tensor peer_rows = peer->decode_ple_rows.slice(1, 0, batch_size);
                CUDA_CHECK(cudaMemcpyAsync(peer_rows.data, peer->decode_ple_host, peer_rows.bytes(),
                                           cudaMemcpyHostToDevice, rank1.stream));
            }
        }

        Tensor tokens             = ordinary.tokens.slice(0, 0, batch_size);
        Tensor cache_positions    = ordinary.cache_positions.slice(0, 0, batch_size);
        Tensor rope_positions     = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows            = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = ordinary.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = ordinary.state_destination_slots.slice(0, 0, batch_size);
        Tensor hidden             = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits             = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled            = ordinary.sampled_tokens.slice(0, 0, batch_size);

        card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows, state_sources,
                                   state_destinations, envelope, hidden, logits);
        ops::scatter(hidden, state_destinations, state.continuation_hidden_store,
                     state.execution.device.stream);
        ops::sample(logits, sampled,
                    dimension(state.execution.parameters.model.resources().public_token_count),
                    ordinary.sampling, cache_positions, ops::kSamplePurposeDecode,
                    state.execution.work, state.execution.device.stream);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, ordinary.egress.data,
                                   sizeof(qwen3_5::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    capture_graph(state, definition, body);
}

void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::CausalAttentionExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::models::qwen3_5::execution

namespace ninfer::models::qwen3_5::detail {

namespace {

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label);

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label);

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label);

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier;
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label) {
    const auto it = std::find_if(family.topologies.begin(), family.topologies.end(),
                                 [topology_class](const DecodeGraphTopology& topology) {
                                     return topology.topology_class == topology_class;
                                 });
    if (it == family.topologies.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph topology is unavailable");
    }
    return *it;
}

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label) {
    DecodeGraphTopology& topology   = select_graph_topology(family, profile.topology_class, label);
    const std::size_t profile_index = static_cast<std::size_t>(&profile - family.profiles.data());
    if (topology.installed_profile != profile_index) {
        topology.executable.update(profile.definition);
        topology.installed_profile = profile_index;
    }
    return topology.executable;
}

// Observation only: the entry is appended within the capacity reserved by install_sampling.
void record_accepted_length(RequestControl& request, std::uint8_t length) noexcept {
    std::vector<std::uint8_t>& lengths = request.speculative_stats.accepted_lengths;
    if (lengths.size() < request.accepted_length_steps) { lengths.push_back(length); }
}

} // namespace

void ProgramImpl::install_sampling(SequenceState& sequence, RequestControl& request,
                                   const ops::SamplingConfig& config,
                                   std::uint32_t accepted_length_steps,
                                   std::uint32_t output_tokens) {
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(sequence.lane), 1)
                        .view({dimension(parameters.model.resources().public_token_count)});
    request.sampling_host     = config;
    request.speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
    };
    // Every decode step commits at least one of the output tokens, so the bound below is never
    // exceeded and recording never reallocates inside a round.
    request.accepted_length_steps = speculative_backend == SpeculativeBackend::None
                                        ? 0U
                                        : std::min(accepted_length_steps, output_tokens);
    if (request.accepted_length_steps != 0) {
        request.speculative_stats.accepted_lengths.reserve(request.accepted_length_steps);
    }
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    if (penalties) { CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream)); }
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &request.sampling_host,
                               sizeof(request.sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

void ProgramImpl::copy_tail(SequenceState& sequence, std::int32_t column) {
    // T42: a Qwen4Exp chunk wider than prefill_hidden ran on lent workspace and left its last
    // column in prefill_hidden's last column (qwen4_text.cpp keep_last_column).
    if (qwen4_prompt_base_chunk > 0 && column >= prefill_hidden.ne[1] &&
        column < static_cast<std::int32_t>(prefill_chunk)) {
        column = prefill_hidden.ne[1] - 1;
    }
    if (column < 0 || column >= prefill_hidden.ne[1] || sequence.tail_hidden.data == nullptr) {
        throw std::logic_error("target tail hidden column is invalid");
    }
    const Tensor source = prefill_hidden.slice(1, column, 1);
    CUDA_CHECK(::ninfer::memcpy_device_async(sequence.tail_hidden.data, source.data,
                                             sequence.tail_hidden.bytes(), device.stream));
    if (peer_retains_hidden()) {
        // The hidden axis is replicated, so rank 1's final-normed chunk holds the same column:
        // a local copy on rank 1's stream, after the chunk that wrote it.
        const Tensor peer_source      = peer->prefill_hidden.slice(1, column, 1);
        const Tensor peer_destination = peer_tail_hidden(sequence);
        const ScopedCurrentDevice rank1(peer->device.device);
        CUDA_CHECK(::ninfer::memcpy_device_async(peer_destination.data, peer_source.data,
                                                 peer_destination.bytes(), peer->device.stream));
    }
    sequence.tail_hidden_valid = true;
}

void ProgramImpl::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImpl::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

void ProgramImpl::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                std::span<const std::uint32_t> starts,
                                                std::span<const std::uint32_t> counts) {
    if (speculative_backend != SpeculativeBackend::DFlash2 || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }

    std::uint32_t minimum_count = draft_window + 1U;
    std::uint32_t maximum_count = 0;
    *dflash_host_ingress        = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || counts[row] == 0 || counts[row] > draft_window + 1U ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("DFlash context append contains an invalid row");
        }
        SequenceState& sequence   = active_sequence(lane);
        const std::uint32_t start = starts[row];
        const std::uint64_t end64 = static_cast<std::uint64_t>(start) + counts[row];
        const std::uint32_t end   = static_cast<std::uint32_t>(end64);
        if (!sequence.kv || text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            end64 > capacity) {
            throw std::logic_error("DFlash context append is outside retained target storage");
        }
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(start, "DFlash append context frontier");
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(end, "DFlash append target frontier");
        dflash_host_ingress->active_lanes[row]            = static_cast<std::int32_t>(lane);
        const StateImageSelectors selectors               = state_selectors(sequence);
        dflash_host_ingress->state_source_slots[row]      = selectors.source;
        dflash_host_ingress->state_destination_slots[row] = selectors.destination;
        // Context append writes only the drafter's fixed cyclic state. Target execution owns Main
        // KV coverage, including any uncommitted suffix that survives until the round is settled.
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    qwen3_5::DFlashDecodeState& frame = *io.dflash_decode;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                               sizeof(qwen3_5::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                               device.stream));
    const auto batch                = static_cast<std::int32_t>(lanes.size());
    Tensor active_lane_tensor       = frame.active_lanes.slice(0, 0, batch);
    Tensor state_destination_tensor = frame.state_destination_slots.slice(0, 0, batch);
    Tensor device_starts            = frame.context_frontiers.slice(0, 0, batch);
    Tensor device_ends              = frame.execution_frontiers.slice(0, 0, batch);
    Tensor positions                = frame.append_positions.slice(1, 0, batch);
    Tensor device_counts            = frame.append_counts.slice(0, 0, batch);

    work.reset();
    Tensor features =
        work.alloc(DType::BF16, {dimension(parameters.draft->feature_projection.weight.k),
                                 static_cast<std::int32_t>(draft_window + 1U), batch});
    ops::prepare_ragged_prefix(dflash->pending_features, active_lane_tensor, device_starts,
                               device_ends, features, positions, device_counts, device.stream);

    execution::DFlashAppendContext state{{device, parameters, work, state_images->linear(),
                                          replay_records ? &*replay_records : nullptr, io,
                                          prefill_hidden, prefill_chunk, proposal_head},
                                         *dflash};
    mark_workspace_usage(workspace_plan.dflash_context);
    execution::dflash_append_context(state, features, positions, device_counts,
                                     state_destination_tensor, {minimum_count, maximum_count});
    // The split drafter at tp 2 catches rank 1's context up from its own pending features.
    if (peer && peer->drafter) {
        execution::dflash_append_context_peer(*peer->drafter, *dflash_host_ingress, batch,
                                              draft_window, {minimum_count, maximum_count});
    }
}

void ProgramImpl::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= dimension(parameters.model.resources().public_token_count)) {
            throw std::runtime_error("target returned a token outside the public token domain");
        }
    }
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                                   std::span<const runtime::RoundBudget> budgets,
                                   runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeOrdinaryRound, nvtx::Category::Decode,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::None) {
        throw std::logic_error("ordinary batch execution requires the ordinary backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("ordinary batch membership is invalid");
    }

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("ordinary batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto start = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeOrdinarySubmit, nvtx::Category::Decode,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable = nullptr;
        ops::CausalAttentionExecutionEnvelope envelope{maximum_frontier + 1, maximum_frontier + 1};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(ordinary_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "ordinary batch");
            executable = &install_graph_profile(ordinary_graphs, profile, "ordinary batch");
            envelope   = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
        }
        if (qwen4_ple_gather) { begin_qwen4_ple_round(); }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence            = active_sequence(lanes[row]);
            const RequestControl& request      = requests[lanes[row]];
            const std::uint32_t frontier       = sequence.execution_frontier;
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "ordinary batch position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            const StateImageSelectors selectors                 = state_selectors(sequence);
            ordinary_host_ingress->state_source_slots[row]      = selectors.source;
            ordinary_host_ingress->state_destination_slots[row] = selectors.destination;
            ordinary_host_ingress->sampling[row]                = request.sampling_host;
            // Structured output: the row's mask, uploaded before the launch (D2).
            ordinary_host_ingress->sampling[row].mask = fill_grammar_mask(row, lanes[row], {});
            ensure_sequence_kv_mapped(sequence, frontier + 1, 0);
            if (qwen4_ple_gather) {
                // Qwen4Exp n-gram embedding of the row's token at `frontier` (its context is the
                // ledger before it).
                queue_qwen4_ple_rows(lanes[row], frontier,
                                     std::span<const TokenId>(&sequence.ledger[frontier], 1), row);
            }
        }

        execution::OrdinaryBatchContext schedule_state{
            {device, parameters, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, tp_binding(), graph_peer_bridge(), qwen4_runtime()},
            decoder->text_kv,
            *io.ordinary,
            *ordinary_host_ingress,
            *ordinary_host_egress,
            state_images->continuation_hidden_store(),
            peer ? &*peer->io.ordinary : nullptr};

        // An eager round gathers its n-gram rows before the body; a graph round after the launch
        // (on the stager thread).
        if (qwen4_ple_gather && (executable == nullptr || !qwen4_ple_stager)) {
            stage_qwen4_ple_round();
        }
        mark_workspace_usage(workspace_plan.ordinary_round);
        qwen4_begin_window(execution::Qwen4Window::Decode);
        execution::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                         envelope, executable);
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            // The n-gram rows are gathered on the stager thread while the device runs the layers
            // before the PLE layer.
            const bool ple_staging = qwen4_ple_gather && executable != nullptr && qwen4_ple_stager;
            if (ple_staging) { qwen4_ple_stager->start(); }
            // The expert cache policy reads the round's routing while the device runs it.
            qwen4_observe_window();
            synchronize_devices();
            if (ple_staging) { qwen4_ple_stager->finish(); }
        }
        timing.end_wait();
        // T53 (eager rounds): the next round's n-gram rows (the sampled token's) start their page
        // reads now; a graph round gathers them after its launch instead.
        if (executable == nullptr) { prefetch_qwen4_ple_next_round(lanes, false); }
        qwen4_end_window();
        if (!qwen4_forced_tokens.empty()) {
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                force_qwen4_tokens(active_sequence(lanes[row]),
                                   &ordinary_host_egress->sampled_tokens[row], 1);
            }
        }
        if (qwen4_replay) {
            // T59 replay: the reference token (teacher forcing through the ordinary route).
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                const SequenceState& sequence = active_sequence(lanes[row]);
                if (const execution::Qwen4ReplaySequence* ref =
                        qwen4_replay->match(sequence.ledger)) {
                    if (ordinary_host_egress->sampled_tokens[row] !=
                        ref->ids[sequence.ledger.size()]) {
                        ++qwen4_replay->argmax_disagreements;
                    }
                    ordinary_host_egress->sampled_tokens[row] = ref->ids[sequence.ledger.size()];
                    ++qwen4_replay->forced_tokens;
                } else {
                    ++qwen4_replay->rounds_unmatched;
                }
            }
        }
        if (qwen4_top2_file) {
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                write_qwen4_top2(io.ordinary->logits.slice(1, static_cast<std::int32_t>(row), 1),
                                 lanes[row], active_sequence(lanes[row]).execution_frontier, 0);
            }
        }
        if (qwen4_tfdump && qwen4_tfdump->logits_enabled()) {
            // Validation dump of the round's logits (position = the consumed token's position).
            const Tensor logits =
                io.ordinary->logits.slice(1, 0, static_cast<std::int32_t>(lanes.size()));
            std::vector<std::uint16_t> host(static_cast<std::size_t>(logits.numel()));
            CUDA_CHECK(
                cudaMemcpy(host.data(), logits.data, logits.bytes(), cudaMemcpyDeviceToHost));
            std::vector<std::int32_t> positions;
            for (const std::uint32_t lane : lanes) {
                positions.push_back(
                    static_cast<std::int32_t>(active_sequence(lane).execution_frontier));
            }
            qwen4_tfdump->write_logits(host, logits.ne[0], positions);
        }

        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = active_sequence(lanes[row]);
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            log_qwen4_round('o', base_E, 0, 0, seconds);
            TokenId token = ordinary_host_egress->sampled_tokens[row];
            if (qwen4_tfdump) {
                // Validation: teacher-forced decode (NINFER_QWEN4_TFDUMP_FORCE).
                if (const std::int32_t forced = qwen4_tfdump->forced_token(sequence.ledger.size());
                    forced >= 0) {
                    token                                     = forced;
                    ordinary_host_egress->sampled_tokens[row] = token;
                }
            }
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            sequence.text_kv_valid = base_E + 1;
            commit_sequence_kv(sequence, sequence.text_kv_valid, 0);
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            sequence.prefix_digests.append_generated(std::span<const TokenId>(&token, 1),
                                                     sequence.rope_delta);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = 1};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens =
                std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(), lanes.size()),
            .timing = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            raise_qwen4_ple_staged(); // a launched round may wait for its n-gram rows
            synchronize_devices();
            if (qwen4_ple_stager) { qwen4_ple_stager->finish(); }
        } catch (...) {}
        timing.end_wait();
        qwen4_abandon_window();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_mtp_batch(std::span<const std::uint32_t> lanes,
                              std::span<const runtime::RoundBudget> budgets,
                              runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeMtpRound, nvtx::Category::Mtp,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        decoder->mtp_cache() == nullptr) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("MTP batch membership is invalid");
    }

    const std::uint32_t width      = draft_window + 1;
    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("MTP batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            backend_kv_addresses->bound_row(*sequence.kv->backend) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > draft_window) {
            throw std::logic_error("MTP batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto started = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeMtpSubmit, nvtx::Category::Mtp,
                             static_cast<std::uint64_t>(lanes.size()));
        std::optional<nvtx::ScopedRange> prepare_range;
        prepare_range.emplace(nvtx::Name::DecodeMtpPrepare, nvtx::Category::Mtp,
                              static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable = nullptr;
        execution::MtpCausalAttentionEnvelopes envelopes =
            mtp_causal_attention_envelopes(maximum_frontier, draft_window, capacity);
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(mtp_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "MTP batch");
            executable = &install_graph_profile(mtp_graphs, profile, "MTP batch");
            envelopes = mtp_causal_attention_envelopes(profile.max_execution_frontier, draft_window,
                                                       capacity);
        }
        if (qwen4_ple_gather) { begin_qwen4_ple_round(); }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = active_sequence(lanes[row]);
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            // T59 replay: the recorded window of this frontier (extent, drafts) and the reference.
            const execution::Qwen4ReplaySequence* replay_ref =
                qwen4_replay ? qwen4_replay->match(sequence.ledger) : nullptr;
            const execution::Qwen4ReplayWindow* replay_window = nullptr;
            if (replay_ref != nullptr) {
                const auto it = replay_ref->windows.find(frontier);
                if (it != replay_ref->windows.end()) { replay_window = &it->second; }
                ++(replay_window != nullptr ? qwen4_replay->windows_replayed
                                            : qwen4_replay->windows_unlisted);
            } else if (qwen4_replay) {
                ++qwen4_replay->rounds_unmatched;
            }
            // T121c: NINFER_QWEN4_REPLAY_MODEL_DRAFTS=1 keeps the model's own drafts (and extent)
            // where the replay has no recorded window: forced targets with realistic rejections.
            static const bool replay_model_drafts = [] {
                const char* v = std::getenv("NINFER_QWEN4_REPLAY_MODEL_DRAFTS");
                return v != nullptr && std::string_view(v) == "1";
            }();
            const std::uint32_t proposed_extent = replay_window != nullptr ? replay_window->extent
                                                  : replay_ref != nullptr && !replay_model_drafts
                                                      ? draft_window
                                                      : sequence.mtp_draft_count;
            const std::uint32_t extent =
                qwen4_mtp_no_drafts
                    ? 0U
                    : std::min({proposed_extent, draft_window, max_by_budget, qwen4_mtp_extent_cap,
                                capacity - sequence.execution_frontier - 1});
            mtp_host_ingress->anchors[row]        = sequence.ledger.back();
            mtp_host_ingress->base_frontiers[row] = checked_i32(frontier, "MTP batch frontier");
            mtp_host_ingress->remaining_budgets[row] =
                checked_i32(budgets[row].generated_tokens_remaining, "MTP batch remaining budget");
            mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
            mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
            for (std::uint32_t j = 0; j < draft_window; ++j) {
                mtp_host_ingress->current_drafts[row * draft_window + j] =
                    j < extent ? sequence.mtp_drafts[j] : sequence.ledger.back();
            }
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::uint32_t position = frontier + std::min(j, extent);
                mtp_host_ingress->target_rope_positions[row * width + j] =
                    checked_i32(position, "MTP batch RoPE position") + sequence.rope_delta;
            }
            mtp_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            mtp_host_ingress->mtp_kv_table_rows[row] =
                backend_kv_addresses->bound_row(*sequence.kv->backend);
            const StateImageSelectors selectors            = state_selectors(sequence);
            mtp_host_ingress->state_source_slots[row]      = selectors.source;
            mtp_host_ingress->state_destination_slots[row] = selectors.destination;
            mtp_host_ingress->rope_deltas[row]             = sequence.rope_delta;
            mtp_host_ingress->sampling[row]                = request.sampling_host;
            ensure_sequence_kv_mapped(sequence, frontier + extent + 1,
                                      std::min(capacity, frontier + extent + draft_window));
            if (!qwen4_mtp_oracle.empty()) {
                // Validation oracle drafts: the reference continuation of the matching sequence.
                for (const std::vector<TokenId>& line : qwen4_mtp_oracle) {
                    if (line.size() <= sequence.ledger.size() ||
                        !std::equal(sequence.ledger.begin(), sequence.ledger.end(), line.begin())) {
                        continue;
                    }
                    for (std::uint32_t j = 0; j < extent && frontier + 1U + j < line.size(); ++j) {
                        mtp_host_ingress->current_drafts[row * draft_window + j] =
                            line[frontier + 1U + j];
                    }
                    break;
                }
            }
            if (replay_ref != nullptr) {
                // T59 replay: recorded drafts (or the reference itself where no window was
                // recorded), and the reference continuation as every column's target, so the
                // device accepts the recorded count and commits the reference tokens.
                const auto& ids = replay_ref->ids;
                for (std::uint32_t j = 0; j < extent; ++j) {
                    const std::size_t at = frontier + 1U + j;
                    if (replay_window == nullptr && replay_model_drafts) { continue; } // T121c
                    mtp_host_ingress->current_drafts[row * draft_window + j] =
                        replay_window != nullptr && j < replay_window->drafts.size()
                            ? replay_window->drafts[j]
                            : ids[std::min(at, ids.size() - 1)];
                }
                for (std::uint32_t j = 0; j < width; ++j) {
                    const std::size_t at = frontier + 1U + j;
                    mtp_host_ingress->forced_targets[row * width + j] =
                        ids[std::min(at, ids.size() - 1)];
                }
            }
            // Structured output: masks of the K+1 verified columns from the host drafts of this
            // round (lookahead, matcher unchanged); columns past an illegal draft carry a safe
            // nonempty mask. Uploaded before the launch (D2).
            mtp_host_ingress->sampling[row].mask = fill_grammar_mask(
                row, lanes[row],
                std::span<const TokenId>(&mtp_host_ingress->current_drafts[row * draft_window],
                                         extent));
            if (qwen4_ple_gather) {
                // Qwen4Exp n-gram rows of the verified window: the anchor, then the drafts (the
                // padding columns past the extent repeat the anchor and are never committed).
                std::array<TokenId, kMtpDecodeMaximumWidth> window{};
                window[0] = sequence.ledger.back();
                for (std::uint32_t j = 0; j < draft_window; ++j) {
                    window[j + 1] = mtp_host_ingress->current_drafts[row * draft_window + j];
                }
                queue_qwen4_ple_rows(lanes[row], frontier,
                                     std::span<const TokenId>(window.data(), width), row * width);
            }
        }

        execution::MtpBatchContext schedule_state{
            {device, parameters, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, tp_binding(), graph_peer_bridge(), qwen4_runtime()},
            decoder->text_kv,
            *decoder->mtp_cache(),
            *io.mtp_decode,
            *mtp_host_ingress,
            *mtp_host_egress,
            state_images->continuation_hidden_store(),
            peer ? &*peer->io.mtp_decode : nullptr,
            peer ? &peer->state_images->continuation_hidden_store() : nullptr,
            mtp_min_p > 0.0F || qwen4_mtp_calib != nullptr};

        // An eager round gathers its n-gram rows before the body; a graph round after the launch
        // (on the stager thread).
        if (qwen4_ple_gather && (executable == nullptr || !qwen4_ple_stager)) {
            stage_qwen4_ple_round();
        }
        prepare_range.reset();
        mark_workspace_usage(workspace_plan.mtp_round);
        // One expert window per MTP round: the verification over the Text layers, then the MTP
        // layer's alignment and draft calls (the offload backend's layer num_hidden_layers).
        qwen4_begin_window(execution::Qwen4Window::Decode);
        execution::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                    draft_window, envelopes, executable);
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            // The n-gram rows are gathered on the stager thread while the device runs the layers
            // before the PLE layer, which waits for them (Qwen4Runtime::decode_ple_staged).
            const bool ple_staging = qwen4_ple_gather && executable != nullptr && qwen4_ple_stager;
            if (ple_staging) { qwen4_ple_stager->start(); }
            // The expert cache policy reads the round's routing while the device runs it.
            qwen4_observe_window();
            synchronize_devices();
            if (ple_staging) { qwen4_ple_stager->finish(); }
        }
        timing.end_wait();
        // T53 (eager rounds): the next round's n-gram rows are known now; their page reads
        // overlap the window end, the commit and the next round's preparation. A graph round
        // gathers them after its launch, off the host's critical path, so it skips this.
        if (executable == nullptr) { prefetch_qwen4_ple_next_round(lanes, true); }
        qwen4_end_window();
        if (!qwen4_forced_tokens.empty()) {
            // Only the round's last committed token is not consumed by the round itself, so only
            // single-token commits (NINFER_QWEN4_MTP_NO_DRAFTS) can be forced.
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (mtp_host_egress->licensed_counts[row] != 1) {
                    throw std::logic_error(
                        "NINFER_QWEN4_FORCE_TOKENS requires one token per round");
                }
                force_qwen4_tokens(active_sequence(lanes[row]),
                                   mtp_host_egress->licensed_tokens.data() + row * width, 1);
            }
        }
        if (qwen4_tfdump && qwen4_tfdump->logits_enabled()) {
            // Validation dump of the target logits of every committed verification column
            // (position = the consumed token's position, as for ordinary rounds).
            const Tensor logits = io.mtp_decode->target_logits;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                const std::int32_t count = mtp_host_egress->licensed_counts[row];
                const Tensor columns =
                    logits.slice(2, static_cast<std::int32_t>(row), 1).slice(1, 0, count);
                std::vector<std::uint16_t> host(static_cast<std::size_t>(columns.numel()));
                CUDA_CHECK(
                    cudaMemcpy(host.data(), columns.data, columns.bytes(), cudaMemcpyDeviceToHost));
                std::vector<std::int32_t> positions;
                for (std::int32_t j = 0; j < count; ++j) {
                    positions.push_back(
                        static_cast<std::int32_t>(active_sequence(lanes[row]).execution_frontier +
                                                  static_cast<std::uint32_t>(j)));
                }
                qwen4_tfdump->write_logits(host, logits.ne[0], positions);
            }
        }
        if (qwen4_top2_file) {
            // Validation: the target's top-2 of every committed verification column.
            const Tensor logits = io.mtp_decode->target_logits;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                const std::int32_t count = mtp_host_egress->licensed_counts[row];
                for (std::int32_t j = 0; j < count; ++j) {
                    write_qwen4_top2(logits.slice(2, static_cast<std::int32_t>(row), 1)
                                         .slice(1, j, 1)
                                         .view({logits.ne[0], 1}),
                                     lanes[row],
                                     active_sequence(lanes[row]).execution_frontier +
                                         static_cast<std::uint32_t>(j),
                                     1);
                }
            }
        }

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = active_sequence(lanes[row]);
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i     = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(draft_window) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("MTP batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(mtp_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (peer && execution::tp_rank_probe_enabled()) { // T123 rank-consistency probe
                execution::tp_rank_probe_check(base_E, static_cast<std::int32_t>(row), count_i,
                                               row_tokens.data());
            }
            const std::uint32_t pcur =
                static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row]);
            log_qwen4_round('m', base_E, pcur, accepted_i, seconds);
            if (qwen4_replay_record != nullptr) {
                std::fprintf(qwen4_replay_record, "win %u %u %d", base_E, pcur, accepted_i);
                for (std::uint32_t j = 0; j < draft_window; ++j) {
                    std::fprintf(qwen4_replay_record, " %d",
                                 mtp_host_ingress->current_drafts[row * draft_window + j]);
                }
                std::fputc('\n', qwen4_replay_record);
            }
            if (qwen4_replay) {
                if (const execution::Qwen4ReplaySequence* ref =
                        qwen4_replay->match(sequence.ledger)) {
                    for (std::int32_t j = 0; j < count_i; ++j) {
                        const std::size_t at = sequence.ledger.size() + static_cast<std::size_t>(j);
                        if (at >= ref->ids.size() ||
                            ref->ids[at] != row_tokens[static_cast<std::size_t>(j)]) {
                            ++qwen4_replay->commit_mismatches;
                            break;
                        }
                    }
                }
            }
            if (pcur == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += pcur;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            if (request.accepted_length_steps != 0) {
                record_accepted_length(request, pcur == 0 ? kSpeculativeFallbackStep
                                                          : static_cast<std::uint8_t>(accepted_i));
            }
            request.pending = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width,
            .timing     = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            raise_qwen4_ple_staged(); // a launched round may wait for its n-gram rows
            synchronize_devices();
            if (qwen4_ple_stager) { qwen4_ple_stager->finish(); }
        } catch (...) {}
        timing.end_wait();
        qwen4_abandon_window();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_dflash_batch(std::span<const std::uint32_t> lanes,
                                 std::span<const runtime::RoundBudget> budgets,
                                 runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeDFlashRound, nvtx::Category::DFlash,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::DFlash2 || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("DFlash batch membership is invalid");
    }

    const std::uint32_t width           = draft_window + 1U;
    std::uint32_t maximum_frontier      = 0;
    std::uint32_t maximum_target_tokens = 1;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("DFlash batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            sequence.dflash_context_frontier > sequence.execution_frontier ||
            sequence.execution_frontier - sequence.dflash_context_frontier > width ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("DFlash batch row is not decode-ready");
        }
        const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                ? budgets[row].generated_tokens_remaining - 1U
                                                : 0U;
        // D3: a constrained row drafts nothing (its drafts are born on the device, after the
        // host masks): an ordinary round for that row, counted as a fallback step.
        const std::uint32_t extent = row_constrained(row)
                                         ? 0U
                                         : std::min({draft_window, max_by_budget,
                                                     capacity - sequence.execution_frontier - 1U});
        maximum_frontier           = std::max(maximum_frontier, sequence.execution_frontier);
        maximum_target_tokens =
            std::max(maximum_target_tokens, sequence.execution_frontier + extent + 1U);
    }

    const auto started = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeDFlashSubmit, nvtx::Category::DFlash,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable    = nullptr;
        execution::DFlashEnvelopes envelopes = dflash_envelopes(0, maximum_frontier, draft_window);
        ops::CausalAttentionExecutionEnvelope target_envelope{1, maximum_target_tokens};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(dflash_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "DFlash batch");
            executable      = &install_graph_profile(dflash_graphs, profile, "DFlash batch");
            envelopes       = dflash_envelopes(profile.min_execution_frontier,
                                               profile.max_execution_frontier, draft_window);
            target_envelope = {
                1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                       capacity, static_cast<std::uint64_t>(profile.max_execution_frontier) +
                                     draft_window + 1ULL))};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = active_sequence(lanes[row]);
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1U
                                                    : 0U;
            const std::uint32_t extent =
                row_constrained(row)
                    ? 0U
                    : std::min({draft_window, max_by_budget, capacity - frontier - 1U});
            dflash_host_ingress->anchors[row] = sequence.ledger.back();
            dflash_host_ingress->execution_frontiers[row] =
                checked_i32(frontier, "DFlash batch frontier");
            dflash_host_ingress->context_frontiers[row] =
                checked_i32(sequence.dflash_context_frontier, "DFlash context frontier");
            dflash_host_ingress->proposal_valid_columns[row] = static_cast<std::int32_t>(width);
            dflash_host_ingress->proposal_extents[row]       = static_cast<std::int32_t>(extent);
            dflash_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1U);
            for (std::uint32_t column = 0; column < width; ++column) {
                const std::uint32_t position = frontier + std::min(column, extent);
                dflash_host_ingress->target_rope_positions[row * width + column] =
                    checked_i32(position, "DFlash target RoPE position") + sequence.rope_delta;
            }
            dflash_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            dflash_host_ingress->active_lanes[row]       = static_cast<std::int32_t>(sequence.lane);
            const StateImageSelectors selectors          = state_selectors(sequence);
            dflash_host_ingress->state_source_slots[row] = selectors.source;
            dflash_host_ingress->state_destination_slots[row] = selectors.destination;
            dflash_host_ingress->sampling[row]                = request.sampling_host;
            dflash_host_ingress->sampling[row].mask = fill_grammar_mask(row, lanes[row], {});
            ensure_sequence_kv_mapped(sequence, frontier + extent + 1U, 0U);
        }

        execution::DFlashBatchContext schedule_state{
            {device, parameters, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, tp_binding(), graph_peer_bridge()},
            decoder->text_kv,
            *dflash,
            *io.dflash_decode,
            *dflash_host_ingress,
            *dflash_host_egress,
            state_images->continuation_hidden_store(),
            peer ? &*peer->io.dflash_decode : nullptr,
            peer ? &peer->state_images->continuation_hidden_store() : nullptr};

        mark_workspace_usage(workspace_plan.dflash_round);
        execution::dflash_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                       draft_window, envelopes, target_envelope, executable);
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            synchronize_devices();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = active_sequence(lanes[row]);
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = dflash_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = dflash_host_egress->accepted_drafts[row];
            const std::uint32_t extent =
                static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]);
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || accepted_i > static_cast<std::int32_t>(extent) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("DFlash batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(dflash_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (extent == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += extent;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            if (request.accepted_length_steps != 0) {
                record_accepted_length(request, extent == 0
                                                    ? kSpeculativeFallbackStep
                                                    : static_cast<std::uint8_t>(accepted_i));
            }
            sequence.dflash_context_frontier = base_E;
            request.pending                  = PendingCandidate{
                                 .kind          = PendingKind::Speculative,
                                 .base_E        = base_E,
                                 .base_S        = base_S,
                                 .prompt_tokens = 0,
                                 .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(dflash_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(dflash_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width,
            .timing     = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            synchronize_devices();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_raw(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed_timing) {
    if (speculative_backend == SpeculativeBackend::None) {
        return decode_ordinary_batch(lanes, budgets, failed_timing);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        return decode_mtp_batch(lanes, budgets, failed_timing);
    }
    return decode_dflash_batch(lanes, budgets, failed_timing);
}

runtime::ExecutionTiming ProgramImpl::resolve_non_speculative_pending(
    SequenceState& sequence, RequestControl& request, std::uint32_t accepted_tokens, bool terminal,
    std::optional<std::uint32_t> prefix_execution_split_after,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (request.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("pending resolution requires a pending generated round");
    }
    if ((request.pending.kind != PendingKind::Begin &&
         request.pending.kind != PendingKind::Ordinary) ||
        request.pending.produced != 1 || accepted_tokens != 1) {
        throw std::logic_error("non-speculative pending round must commit its single token");
    }

    const std::uint32_t base_ledger_frontier = request.pending.kind == PendingKind::Begin
                                                   ? request.pending.prompt_tokens
                                                   : request.pending.base_S;
    commit_generated_prefix_identity(
        sequence, base_ledger_frontier,
        std::span<const TokenId>(sequence.ledger).subspan(base_ledger_frontier, accepted_tokens),
        prefix_execution_split_after);

    switch (request.pending.kind) {
    case PendingKind::Begin:
        sequence.execution_frontier = request.pending.prompt_tokens;
        sequence.ledger_frontier    = request.pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
        advance_rebuild_work(sequence, request.pending.base_E + request.pending.produced,
                             prefill_chunk);
        sequence.execution_frontier = request.pending.base_E + request.pending.produced;
        sequence.ledger_frontier    = request.pending.base_S + request.pending.produced;
        break;
    case PendingKind::Speculative:
    case PendingKind::None:
        throw std::logic_error("non-speculative pending round has an invalid kind");
    }
    if (sequence.ledger_frontier != sequence.execution_frontier + 1 ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier ||
        sequence.prefix_digests.size() != sequence.ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    // Begin publishes a sampled token but does not execute it through the target. An exact-hit
    // Fork therefore still names an immutable read source and an unwritten destination here; the
    // first state-mutating decode commit closes it. A suffix prefill already closed its Fork at
    // the committed prefill frontier.
    if (request.pending.kind == PendingKind::Begin && terminal && sequence.state.fork_pending) {
        const StateImageSelectors selectors = state_selectors(sequence);
        timing.resume_submit();
        state_images->copy_slot(selectors.source, selectors.destination, device.stream);
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        settle_state_fork(sequence);
    } else if (request.pending.kind == PendingKind::Ordinary) {
        settle_state_fork(sequence);
    }
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    if (terminal) { sequence.mtp_draft_count = 0; }
    request.lifecycle = terminal ? Lifecycle::Finishable : Lifecycle::Active;
    request.pending   = {};
    return timing.finish();
}


} // namespace ninfer::models::qwen3_5::detail
