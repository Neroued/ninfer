#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/reasoning_eos_guard.h"
#include "core/device.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <cstdlib>
#include <string_view>

namespace {
// NINFER_QWEN4_REWRITE_SPLITS=1 keeps the rewrite-frontier prefill splits of a routed-expert model.
bool qwen4_rewrite_splits() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_QWEN4_REWRITE_SPLITS");
        return v != nullptr && std::string_view(v) == "1";
    }();
    return on;
}
} // namespace

namespace ninfer::models::qwen3_5::execution {
namespace {

DFlashFeatureSink make_dflash_prefill_sink(PrefillContext& state) {
    if (!state.execution.io.dflash_prefill || state.dflash_prefill_host_ingress == nullptr) {
        throw std::logic_error("DFlash prefill controls are unavailable");
    }
    return dflash_feature_sink(
        state, [&state](const Tensor& features, const Tensor& positions, bool rewrite_checkpoint) {
            auto& frame = *state.execution.io.dflash_prefill;
            // Target execution and draft append use the same chunk bindings. Decode may have
            // used another compact row, or a checkpoint may have forked the destination slot.
            *state.dflash_prefill_host_ingress = {
                .append_count           = features.ne[1],
                .state_destination_slot = state.state_destination_slot,
            };
            CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, state.dflash_prefill_host_ingress,
                                       sizeof(qwen3_5::DFlashPrefillIngress),
                                       cudaMemcpyHostToDevice, state.execution.device.stream));
            const auto exact = static_cast<std::uint32_t>(features.ne[1]);
            dflash_append_context(state, features, positions, frame.append_count,
                                  frame.state_destination_slot, {exact, exact});
            (void)rewrite_checkpoint;
        });
}

// The vocabulary-split output head over one retained hidden column (program/tp2/prefill_tp2.inc).
void project_split_output_head(PrefillContext& state, const Tensor& hidden, Tensor& logits);

} // namespace

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t state_source_slot,
                         std::int32_t state_destination_slot, std::uint32_t mtp_proposal_extent) {
    card.set_sampling(sampling);
    card.set_qwen4_runtime(execution.qwen4);
    card.set_linear_state_slots(state_source_slot, state_destination_slot);
    card.set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    card.set_mtp_proposal_extent(mtp_proposal_extent);
    if (execution.proposal_head == ProposalHead::Full) {
        card.set_proposal_head(nullptr, nullptr, 0);
        return;
    }
    if (card.proposal_head() == nullptr || card.proposal_head_n() <= 0) {
        throw std::runtime_error("optimized proposal head is unavailable");
    }
}

PrefillChunkResult prefill_text_chunk(PrefillContext& state, std::span<const TokenId> ids,
                                      std::uint32_t nominal_length,
                                      std::optional<std::uint32_t> split_frontier,
                                      bool finalize_at_end) {
    TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                     state.execution.tp, state.rope_delta);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_reasoning_guard(state.reasoning_guard);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden,
                                              state.peer_rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    const std::span<const int> prompt(ids.data(), ids.size());
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        // The split drafter at tp 2 captures and appends rank 1's context too.
        std::optional<DFlashFeatureSink> peer_sink;
        if (state.execution.tp != nullptr && state.execution.tp->dflash != nullptr) {
            peer_sink = dflash_peer_prefill_sink(*state.execution.tp->dflash,
                                                 state.dflash_prefill_host_ingress);
            card.set_peer_feature_sink(&*peer_sink);
        }
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                                  sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end);
}

PrefillChunkResult prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                                            VisionPrefillSession& vision,
                                            std::uint32_t nominal_length,
                                            std::optional<std::uint32_t> split_frontier,
                                            bool finalize_at_end) {
    TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                     state.execution.tp, state.rope_delta);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_reasoning_guard(state.reasoning_guard);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden,
                                              state.peer_rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        std::optional<DFlashFeatureSink> peer_sink;
        if (state.execution.tp != nullptr && state.execution.tp->dflash != nullptr) {
            peer_sink = dflash_peer_prefill_sink(*state.execution.tp->dflash,
                                                 state.dflash_prefill_host_ingress);
            card.set_peer_feature_sink(&*peer_sink);
        }
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision,
                                  finalize_at_end, sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision, finalize_at_end);
}

void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge) {
    if (!state.mtp_kv.valid() || bridge.previous_hidden == nullptr || state.text_kv_base == 0 ||
        bridge.position < 0 ||
        static_cast<std::uint32_t>(bridge.position) + 1 != state.text_kv_base) {
        throw std::logic_error("multimodal MTP bridge does not match the reusable frontier");
    }

    Tensor bridge_token = state.execution.io.mtp->target_input_ids.slice(0, 0, 1);
    const TokenId token = prompt.token_ids[state.text_kv_base];
    CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token), cudaMemcpyHostToDevice,
                               state.execution.device.stream));

    Tensor visual_embedding;
    const Tensor* composed_embedding = nullptr;
    if (prompt.token_types[state.text_kv_base] != 0) {
        const VisionChunk chunk = vision.prepare_chunk(state.text_kv_base, 1);
        if (chunk.control == nullptr) {
            throw std::logic_error("visual MTP bridge has no encoded Vision item");
        }
        const auto& scatter = chunk.control->scatter_indices;
        const auto column   = std::lower_bound(scatter.begin(), scatter.end(),
                                               static_cast<std::int32_t>(state.text_kv_base));
        if (column == scatter.end() || *column != static_cast<std::int32_t>(state.text_kv_base) ||
            static_cast<std::uint8_t>(chunk.control->modality) !=
                prompt.token_types[state.text_kv_base]) {
            throw std::logic_error("visual MTP bridge does not match Vision scatter metadata");
        }
        visual_embedding =
            chunk.embeddings.slice(1, static_cast<std::int32_t>(column - scatter.begin()), 1);
        composed_embedding = &visual_embedding;
    }

    mtp_bridge_and_propose(state, bridge_token, *bridge.previous_hidden,
                           bridge.peer_previous_hidden, bridge.position, bridge.rope_position,
                           false, composed_embedding);
}

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose, const Tensor* peer_hidden, bool reasoning_guard) {
    const TextConfig& config = state.execution.parameters.model.config().text;
    // Qwen4Exp retains the wide residual (T66b: a zero-suffix hit of this architecture).
    const std::int32_t rows =
        config.qwen4() ? dimension(config.residual_width()) : dimension(config.hidden_size);
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != rows || hidden.ne[1] != 1 ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("sample_from_hidden requires BF16 [hidden,1]");
    }
    state.execution.work.reset();
    Tensor logits = state.execution.io.logits.slice(1, 0, 1);
    if (config.qwen4()) {
        TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                         state.text_kv, state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk,
                         state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                         state.execution.tp, state.rope_delta);
        card.set_qwen4_runtime(state.execution.qwen4);
        card.qwen4_logits_from_hidden(hidden, peer_hidden, logits);
    } else if (state.execution.tp != nullptr) {
        project_split_output_head(state, hidden, logits);
    } else {
        project(hidden, state.execution.parameters.text.output_head, logits, state.execution.work,
                state.execution.device.stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(state.execution.io.pos.data, &absolute_position,
                               sizeof(absolute_position), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    ops::sample(logits, state.execution.io.token,
                dimension(state.execution.parameters.model.resources().public_token_count),
                state.sampling, state.execution.io.pos, purpose, state.execution.work,
                state.execution.device.stream);
    if (reasoning_guard) {
        guard_first_token(
            logits, state.execution.io.token,
            dimension(state.execution.parameters.model.resources().public_token_count),
            state.sampling, state.execution.io.pos, state.execution.work,
            state.execution.device.stream);
    }
    state.execution.work.reset();
    if (state.execution.tp != nullptr) { state.execution.tp->work->reset(); }
}

} // namespace ninfer::models::qwen3_5::execution

namespace ninfer::models::qwen3_5::detail {

namespace {

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token);

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token) {
    const std::size_t tokens = prompt.token_ids.size();
    if (token >= tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("MTP bridge position is outside prepared prompt metadata");
    }
    return {prompt.positions[token], prompt.positions[tokens + token],
            prompt.positions[2 * tokens + token]};
}

} // namespace

void ProgramImpl::start_sequence(std::uint32_t lane, SequenceState& sequence,
                                 MaterializationTransaction& transaction) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    RequestControl& request = requests[lane];
    if (!transaction.plan || transaction.plan->impl_ == nullptr || !transaction.prepared ||
        !request.prefill) {
        throw std::invalid_argument("materialization staging is incomplete");
    }
    AdmissionCandidateImpl& request_plan = *transaction.plan->impl_;
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("staged prefill requires a free request lane");
    }
    auto& staged                           = *request.prefill;
    const auto started                     = Clock::now();
    const std::uint32_t prompt_tokens      = staged.prompt_tokens;
    const std::uint32_t base               = staged.base;
    const std::uint32_t initial_mtp_extent = staged.initial_mtp_extent;
    request.lifecycle                      = Lifecycle::Empty;
    // The sequence's own output ceiling: the last frontier its Device KV lease may cover, so
    // on-demand growth never leases pages the request cannot reach.
    request.lease_ceiling =
        std::min(capacity, request_plan.summary.prompt_tokens +
                               (request_plan.summary.effective_output_tokens == 0
                                    ? 0U
                                    : request_plan.summary.effective_output_tokens - 1U));
    try {
        const std::uint32_t state_slots = request_plan.demand.active_entitlement.device.state_slots;
        const bool preserving_source =
            (transaction.has_source || transaction.has_shared_source) &&
            transaction.source_mode == runtime::PrivateSourceMode::Retain;
        const bool text_prefix_fork    = request_plan.text_prefix_fork_required;
        const bool backend_prefix_fork = request_plan.backend_prefix_fork_required;
        if (request_plan.reuse == ReusePath::Root) {
            if (transaction.reserved_state_count != state_slots || state_slots == 0 ||
                !transaction.root_text_address || !transaction.text_activation ||
                transaction.root_backend_address.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0) ||
                transaction.backend_activation.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0)) {
                throw std::logic_error("root materialization reservations are incomplete");
            }
            release_sequence_kv(sequence);
            release_sequence_state(sequence);
            sequence.state = ActiveStateBinding{.read  = transaction.reserved_states[0],
                                                .write = transaction.reserved_states[0]};
            transaction.reserved_states[0] = {};
            if (state_slots == 2) {
                sequence.reserved_state        = transaction.reserved_states[1];
                transaction.reserved_states[1] = {};
            }
            transaction.reserved_state_count = 0;

            SequenceKVBundle bundle{.text = *transaction.root_text_address};
            transaction.root_text_address.reset();
            if (transaction.root_backend_address) {
                bundle.backend = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
            }
            sequence.kv.emplace(bundle);
        } else if (preserving_source) {
            const bool private_source_ready = transaction.has_source &&
                                              transaction.source_index < continuation_capacity &&
                                              continuation_slots[transaction.source_index].role ==
                                                  ContinuationSlotRole::Catalogued;
            const bool shared_source_ready =
                transaction.has_shared_source &&
                transaction.shared_source_index < shared_prefix_capacity &&
                shared_prefix_slots[transaction.shared_source_index].role ==
                    SharedPrefixSlotRole::Catalogued;
            if (private_source_ready == shared_source_ready ||
                transaction.reserved_state_count != state_slots || state_slots == 0 ||
                !transaction.root_text_address || !transaction.text_prefix_fork ||
                !transaction.prefix_forks_ready ||
                transaction.root_backend_address.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0)) {
                throw std::logic_error("retained materialization is incomplete");
            }
            const StateImageHandle selected =
                private_source_ready
                    ? selected_state(continuation_states[transaction.source_index],
                                     request_plan.reuse, request_plan.selected_checkpoint)
                    : shared_prefix_states[transaction.shared_source_index].state;
            const StateImageHandle current = transaction.reserved_states[0];
            if (state_store->residency(selected) == StateReplicaResidency::HostOnly) {
                if (state_store->role(current) != StateImageRole::ActiveMutable) {
                    throw std::logic_error("Host retained Fork destination was not published");
                }
                sequence.state = ActiveStateBinding{.read = current, .write = current};
            } else if (transaction.split_state_identity) {
                if (!private_source_ready ||
                    state_store->residency(selected) != StateReplicaResidency::Both) {
                    throw std::logic_error("StateImage identity split source changed");
                }
                state_store->split_device_replica_identity(selected, current);
                sequence.state = ActiveStateBinding{.read = current, .write = current};
            } else {
                const StateImageSelectors selectors = state_store->begin_fork(selected, current);
                if (speculative_backend == SpeculativeBackend::DFlash2) {
                    state_images->copy_dflash_local(selectors.source, selectors.destination,
                                                    device.stream);
                }
                sequence.state = ActiveStateBinding{
                    .read           = selected,
                    .write          = current,
                    .fork_pending   = true,
                    .read_ownership = StateReadOwnership::ExternalOwner,
                };
            }
            transaction.reserved_states[0]   = {};
            transaction.split_state_identity = false;
            if (state_slots == 2) {
                sequence.reserved_state        = transaction.reserved_states[1];
                transaction.reserved_states[1] = {};
            }
            transaction.reserved_state_count = 0;
            sequence.rewrite_state.reset();
            sequence.rewrite_checkpoint = {};

            SequenceKVBundle bundle{.text = *transaction.root_text_address};
            transaction.root_text_address.reset();
            if (transaction.root_backend_address) {
                bundle.backend = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
            }
            sequence.kv.emplace(bundle);
        } else {
            if (request_plan.state_fork_required !=
                transaction.state_fork_destination.has_value()) {
                throw std::logic_error("private materialization StateImage Fork is incomplete");
            }
            if (transaction.reserved_state_count > 1 ||
                (transaction.reserved_state_count != 0 && sequence.reserved_state)) {
                throw std::logic_error("private materialization StateImage reservation is invalid");
            }
            if (transaction.reserved_state_count == 1) {
                sequence.reserved_state          = transaction.reserved_states[0];
                transaction.reserved_states[0]   = {};
                transaction.reserved_state_count = 0;
            }
        }

        if (!preserving_source) {
            std::array<HostKVPageReplicaRelease, 2> stale_tail_replicas{};
            std::size_t stale_tail_count           = 0;
            const auto preflight_inactive_truncate = [&](KVAddressSpaceStore& addresses,
                                                         LogicalKVPageStore& pages,
                                                         KVAddressSpaceHandle address,
                                                         std::optional<std::uint32_t> frontier) {
                if (!frontier ||
                    (addresses.committed_frontier(address) == *frontier &&
                     addresses.mapped_pages(address) == kv_pages_for_frontier(*frontier))) {
                    return;
                }
                bool releases_tail               = false;
                const std::uint32_t target_pages = kv_pages_for_frontier(*frontier);
                if (target_pages != 0) {
                    const LogicalKVPageHandle tail =
                        addresses.logical_page(address, target_pages - 1U);
                    const std::uint32_t columns =
                        *frontier -
                        (target_pages - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
                    if (columns != pages.committed_columns(tail) && pages.host_resident(tail)) {
                        if (host_kv_extents == nullptr ||
                            stale_tail_count == stale_tail_replicas.size()) {
                            throw std::logic_error("stale Host KV tail replica is not releasable");
                        }
                        stale_tail_replicas[stale_tail_count++] =
                            HostKVPageReplicaRelease{.pages = &pages, .page = tail};
                        releases_tail = true;
                    }
                }
                if (!addresses.can_destructive_truncate_inactive(address, *frontier,
                                                                 releases_tail)) {
                    throw std::logic_error(
                        "selected private KV frontier is not destructively materializable");
                }
            };
            if (!sequence.kv) {
                throw std::logic_error("materialization destination has no KV address space");
            }
            if (!text_prefix_fork) {
                preflight_inactive_truncate(*text_kv_addresses, *text_kv_pages, sequence.kv->text,
                                            transaction.text_activation_frontier);
            }
            if (sequence.kv->backend && !backend_prefix_fork) {
                preflight_inactive_truncate(*backend_kv_addresses, *backend_kv_pages,
                                            *sequence.kv->backend,
                                            transaction.backend_activation_frontier);
            }
            if (stale_tail_count != 0) {
                const std::span<const HostKVPageReplicaRelease> releases(stale_tail_replicas.data(),
                                                                         stale_tail_count);
                if (!host_kv_extents->release_page_replicas(releases)) {
                    throw std::logic_error(
                        "stale Host KV tail replicas cannot be released atomically");
                }
            }
            if (!text_prefix_fork && transaction.text_activation_frontier &&
                (text_kv_addresses->committed_frontier(sequence.kv->text) !=
                     *transaction.text_activation_frontier ||
                 text_kv_addresses->mapped_pages(sequence.kv->text) !=
                     kv_pages_for_frontier(*transaction.text_activation_frontier))) {
                text_kv_addresses->destructive_truncate_inactive(
                    sequence.kv->text, *transaction.text_activation_frontier);
            }
            if (!backend_prefix_fork && transaction.backend_activation_frontier &&
                sequence.kv->backend &&
                (backend_kv_addresses->committed_frontier(*sequence.kv->backend) !=
                     *transaction.backend_activation_frontier ||
                 backend_kv_addresses->mapped_pages(*sequence.kv->backend) !=
                     kv_pages_for_frontier(*transaction.backend_activation_frontier))) {
                backend_kv_addresses->destructive_truncate_inactive(
                    *sequence.kv->backend, *transaction.backend_activation_frontier);
            }
            if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
        }
        if ((text_prefix_fork || backend_prefix_fork) && !transaction.prefix_forks_ready) {
            throw std::logic_error("materialization prefix forks are incomplete");
        }
        if (text_prefix_fork) {
            text_kv_addresses->commit_prefix_fork(std::move(*transaction.text_prefix_fork),
                                                  device.stream);
            transaction.text_prefix_fork.reset();
            if (!preserving_source) {
                const KVAddressSpaceHandle source_address = sequence.kv->text;
                sequence.kv->text                         = *transaction.root_text_address;
                transaction.root_text_address.reset();
                if (!text_kv_addresses->release(source_address)) {
                    throw std::logic_error("consumed Text KV source remained pinned after COW");
                }
            }
        } else {
            text_kv_addresses->commit_activation(std::move(*transaction.text_activation),
                                                 device.stream);
            transaction.text_activation.reset();
        }
        if (backend_prefix_fork) {
            backend_kv_addresses->commit_prefix_fork(std::move(*transaction.backend_prefix_fork),
                                                     device.stream);
            transaction.backend_prefix_fork.reset();
            if (!preserving_source) {
                const KVAddressSpaceHandle source_address = *sequence.kv->backend;
                sequence.kv->backend                      = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
                if (!backend_kv_addresses->release(source_address)) {
                    throw std::logic_error("consumed Backend KV source remained pinned after COW");
                }
            }
        } else if (transaction.backend_activation) {
            backend_kv_addresses->commit_activation(std::move(*transaction.backend_activation),
                                                    device.stream);
            transaction.backend_activation.reset();
        }
        transaction.prefix_forks_ready = false;
        transaction.text_activation_frontier.reset();
        transaction.backend_activation_frontier.reset();
        transaction.prepared = false;

        const bool preserve_rewrite =
            request_plan.rewrite_disposition == RewriteCheckpointDisposition::RetainExisting;
        const auto activate_consumed_state = [&](StateImageHandle selected) {
            if (!request_plan.state_fork_required) {
                if (transaction.state_fork_destination ||
                    state_store->checkpoint_references(selected) != 0) {
                    throw std::logic_error("planned StateImage Move is no longer valid");
                }
                state_store->move_checkpoint_to_active(selected);
                sequence.state = ActiveStateBinding{.read = selected, .write = selected};
                return;
            }
            if (!transaction.state_fork_destination ||
                state_store->checkpoint_references(selected) == 0) {
                throw std::logic_error("planned StateImage Fork is no longer valid");
            }
            const StateImageHandle destination = *transaction.state_fork_destination;
            if (transaction.state_restored) {
                if (state_store->role(destination) != StateImageRole::ActiveMutable) {
                    throw std::logic_error("restored StateImage Fork destination is unavailable");
                }
                sequence.state = ActiveStateBinding{.read = destination, .write = destination};
            } else {
                const std::uint32_t references = state_store->checkpoint_references(selected);
                const std::uint32_t lineage_references =
                    owned_checkpoint_references(sequence, selected);
                if (lineage_references > references) {
                    throw std::logic_error("consumed StateImage Fork ownership is inconsistent");
                }
                const StateReadOwnership read_ownership =
                    lineage_references == references ? StateReadOwnership::LineageCheckpoint
                                                     : StateReadOwnership::ExternalOwner;
                const StateImageSelectors selectors =
                    state_store->begin_fork(selected, destination);
                if (speculative_backend == SpeculativeBackend::DFlash2) {
                    state_images->copy_dflash_local(selectors.source, selectors.destination,
                                                    device.stream);
                }
                sequence.state = ActiveStateBinding{
                    .read           = selected,
                    .write          = destination,
                    .fork_pending   = true,
                    .read_ownership = read_ownership,
                };
            }
            transaction.state_fork_destination.reset();
        };
        if (request_plan.reuse == ReusePath::Root) {
            sequence.rewrite_checkpoint = {};
            ordered_reset(sequence);
            sequence.ledger.clear();
            sequence.prefix_digests.clear();
            sequence.text_kv_valid = 0;
            sequence.mtp_kv_valid  = 0;
        } else if (preserving_source) {
            const SequenceState* private_source =
                transaction.has_source ? &continuation_states[transaction.source_index] : nullptr;
            SharedPrefixState* shared_source =
                transaction.has_shared_source
                    ? &shared_prefix_states[transaction.shared_source_index]
                    : nullptr;
            const std::uint32_t source_text_frontier =
                private_source != nullptr ? private_source->text_kv_valid : shared_source->frontier;
            if (!sequence.kv || source_text_frontier < base) {
                throw std::logic_error("retained prefix has incomplete Text KV");
            }
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base       = base == 0 ? 0 : base - 1U;
                const std::uint32_t source_backend = private_source != nullptr
                                                         ? private_source->mtp_kv_valid
                                                         : shared_source->backend_frontier;
                if (!request_plan.prepare_mtp || source_backend < mtp_base) {
                    throw std::logic_error("retained prefix has incomplete MTP KV");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash2) {
                const std::uint32_t source_backend = private_source != nullptr
                                                         ? private_source->dflash_context_frontier
                                                         : shared_source->frontier;
                if (source_backend < base) {
                    throw std::logic_error("retained prefix has incomplete DFlash KV");
                }
                sequence.dflash_context_frontier = base;
            }
            sequence.tail_hidden_valid =
                base == prompt_tokens &&
                (private_source != nullptr ? private_source->tail_hidden_valid
                                           : shared_source->tail_hidden_valid);
            if (shared_source != nullptr) {
                if (shared_source->active_references == std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error("shared-prefix active reference overflow");
                }
                ++shared_source->active_references;
                sequence.shared_prefix_references.push_back(transaction.shared_source_index);
            }
            refresh_state_views(sequence);
            bind_sequence_kv(sequence);
        } else if (request_plan.reuse == ReusePath::PrivateEndpoint) {
            if (!state_store->valid(sequence.state.read) ||
                sequence.state.read != sequence.state.write || sequence.state.fork_pending ||
                state_store->role(sequence.state.read) != StateImageRole::CheckpointImmutable) {
                throw std::logic_error("resident endpoint StateImage is not movable");
            }
            if (!preserve_rewrite && sequence.rewrite_state) {
                const StateImageHandle dropped = *sequence.rewrite_state;
                state_store->release_checkpoint_reference(dropped);
                sequence.rewrite_state.reset();
                sequence.rewrite_checkpoint = {};
                if (dropped != sequence.state.read &&
                    state_store->checkpoint_references(dropped) == 0 &&
                    !state_store->release(dropped)) {
                    throw std::logic_error("dropped rewrite StateImage could not be released");
                }
            }
            activate_consumed_state(sequence.state.read);
            if (!sequence.kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (sequence.text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the append frontier");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error("resident MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash2 &&
                       sequence.dflash_context_frontier != base) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
            bind_sequence_kv(sequence);
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.text_kv_valid = base;
            sequence.ledger.resize(base);
            sequence.prefix_digests.truncate(base);
            reserve_state_entitlement(sequence, state_slots);
            refresh_state_views(sequence);
        } else if (is_rewrite_checkpoint_restore(request_plan.reuse)) {
            if (!sequence.kv || sequence.text_kv_valid < base) {
                throw std::logic_error("resident rewrite checkpoint has no complete KV allocation");
            }
            if (!sequence.rewrite_state || !state_store->valid(*sequence.rewrite_state) ||
                state_store->role(*sequence.rewrite_state) != StateImageRole::CheckpointImmutable ||
                (sequence.endpoint_valid &&
                 (!state_store->valid(sequence.state.read) ||
                  sequence.state.read != sequence.state.write || sequence.state.fork_pending ||
                  state_store->role(sequence.state.read) != StateImageRole::CheckpointImmutable))) {
                throw std::logic_error("resident rewrite StateImage is not movable");
            }
            const StateImageHandle checkpoint = *sequence.rewrite_state;
            if (sequence.endpoint_valid && sequence.state.read == checkpoint) {
                throw std::logic_error("resident endpoint aliases its rewrite StateImage");
            }
            if (sequence.endpoint_valid && !state_store->release(sequence.state.read)) {
                throw std::logic_error("superseded endpoint StateImage could not be released");
            }
            if (!preserve_rewrite) {
                state_store->release_checkpoint_reference(checkpoint);
                sequence.rewrite_state.reset();
                sequence.rewrite_checkpoint = {};
            }
            activate_consumed_state(checkpoint);
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error(
                        "rewrite-checkpoint MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash2) {
                if (!dflash || sequence.dflash_context_frontier < base) {
                    throw std::logic_error("planned DFlash rewrite checkpoint is unavailable");
                }
                sequence.dflash_context_frontier = base;
            }
            bind_sequence_kv(sequence);
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.tail_hidden_valid = base == prompt_tokens;
            sequence.ledger.resize(base);
            sequence.prefix_digests.truncate(base);
            reserve_state_entitlement(sequence, state_slots);
            refresh_state_views(sequence);
        } else {
            throw std::logic_error("request plan has an invalid prefix reuse path");
        }

        sequence.endpoint_valid = false;
        if (!preserving_source) { trim_sequence_kv(sequence, base, backend_kv_valid(sequence)); }
        bind_sequence_kv(sequence);
        const std::uint32_t backend_materialized =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity,
                           prompt_tokens + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
                : 0U;
        // Active KV in RAM: give back the RAM of the Host KV pages above the live extent
        // (conversations that ended or left the pool) before the prompt maps its own pages.
        decoder->text_kv.page_pool().trim_host_backing();
        if (qwen3_5::PagedKVCache* mtp = decoder->mtp_cache()) {
            mtp->page_pool().trim_host_backing();
        }
        // T64: before the prompt maps its KV, size the KV backing for the prompt and the first
        // decode tokens (up to one grow step) and give the rest back to the expert cache.
        if (qwen4_elastic) {
            const std::uint32_t ahead = std::min<std::uint32_t>(
                request_plan.summary.effective_output_tokens + draft_window + 1U,
                qwen4_elastic->step_tokens());
            qwen4_elastic->rebalance(
                std::min(capacity, prompt_tokens + ahead),
                backend_materialized == 0 ? 0U : std::min(capacity, backend_materialized + ahead));
        }
        ensure_sequence_kv_mapped(sequence, prompt_tokens, backend_materialized);
        install_sampling(sequence, request, request_plan.sampling,
                         request_plan.accepted_length_steps,
                         request_plan.summary.effective_output_tokens);
        sequence.rope_delta = staged.prompt.rope_delta;
        set_device_i32(io.rope_delta, sequence.rope_delta);
        // Rank 1's copy of the per-sequence control, as the KV rows are published on every rank;
        // the MTP proposal steps offset rank 1's positions by it (TpExecution::rope_delta).
        if (peer) { set_peer_i32(peer->io.rope_delta, sequence.rope_delta); }

        request.timings              = {};
        request.pending              = {};
        request.publish_continuation = request_plan.summary.publish_continuation;
        sequence.mtp_draft_count     = 0;
        sequence.tail_hidden_valid   = base == prompt_tokens && sequence.tail_hidden_valid;
        sequence.ledger.swap(materialization_ledger_);
        sequence.prefix_identity.swap(materialization_identity_);
        sequence.prefix_digests.swap(materialization_prefix_digests_);
        sequence.rebuild_work       = request_plan.root_rebuild_work;
        sequence.rebuild_tail_begin = request_plan.root_rebuild_tail_begin;

        if (speculative_backend == SpeculativeBackend::DFlash2) {
            if (!dflash || !io.dflash_prefill || !dflash_prefill_host_ingress) {
                throw std::logic_error("DFlash prefill state is incomplete");
            }
        }

        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        request.lifecycle = Lifecycle::Prefilling;
    } catch (...) {
        try {
            synchronize_devices();
        } catch (...) {}
        clear_lane_best_effort(sequence, request);
        throw;
    }
}

runtime::PrefillStepResult
ProgramImpl::advance_prefill_raw(std::uint32_t lane, runtime::ExecutionTiming* failed_timing) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    return advance_prefill(active_sequence(lane), requests[lane], failed_timing);
}

runtime::ExecutionTiming ProgramImpl::resolve_prefill_raw(std::uint32_t lane, bool terminal,
                                                          runtime::ExecutionTiming* failed_timing) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (requests[lane].pending.kind != PendingKind::Begin) {
        throw std::logic_error("prefill resolution requires a pending prefill token");
    }
    return resolve_non_speculative_pending(active_sequence(lane), requests[lane], 1, terminal,
                                           std::nullopt, failed_timing);
}

runtime::ExecutionTiming ProgramImpl::resolve_pending_raw(
    std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
    std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled,
    std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (lanes.empty() || lanes.size() > max_concurrency || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || cancelled.size() != lanes.size() ||
        prefix_execution_splits.size() != lanes.size()) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }

    if (lanes.size() == 1 && lanes.front() < max_concurrency &&
        requests[lanes.front()].pending.kind == PendingKind::Begin) {
        const std::uint32_t lane = lanes.front();
        if (requests[lane].lifecycle != Lifecycle::Pending) {
            throw std::logic_error("prefill pending token no longer matches Program state");
        }
        if (cancelled.front()) {
            if (accepted_tokens.front() != 0 || !terminal.front()) {
                throw std::logic_error("cancelled prefill pending decision is invalid");
            }
            if (!clear_lane_strict(active_sequence(lane), requests[lane])) {
                throw std::logic_error("cancelled prefill lane is not strictly releasable");
            }
        } else {
            timing.pause();
            timing.include(resolve_non_speculative_pending(
                active_sequence(lane), requests[lane], accepted_tokens.front(),
                terminal.front() != 0, prefix_execution_splits.front(), failed_timing));
            timing.resume_post();
        }
        return timing.finish();
    }

    if (speculative_backend == SpeculativeBackend::None) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
                requests[lane].pending.kind != PendingKind::Ordinary) {
                throw std::logic_error("ordinary pending batch no longer matches Program state");
            }
            if (cancelled[row]) {
                if (!clear_lane_strict(active_sequence(lane), requests[lane])) {
                    throw std::logic_error("cancelled decode lane is not strictly releasable");
                }
            } else {
                timing.pause();
                timing.include(resolve_non_speculative_pending(
                    active_sequence(lane), requests[lane], accepted_tokens[row], terminal[row] != 0,
                    prefix_execution_splits[row], failed_timing));
                timing.resume_post();
            }
        }
        return timing.finish();
    }

    if (!replay_fold) {
        throw std::logic_error("speculative pending batch has no ReplaySSM records");
    }

    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
    std::array<std::int32_t, kMaximumConcurrency> hidden_selectors{};
    bool needs_hidden_correction = false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
            requests[lane].pending.kind != PendingKind::Speculative) {
            throw std::logic_error("speculative pending batch no longer matches Program state");
        }
        const PendingCandidate& pending = requests[lane].pending;
        const SequenceState& sequence   = active_sequence(lane);
        if (sequence.execution_frontier != pending.base_E ||
            sequence.ledger_frontier != pending.base_S ||
            sequence.ledger.size() != pending.base_S ||
            sequence.prefix_identity.size() != pending.base_S ||
            sequence.prefix_digests.size() != pending.base_S ||
            sequence.text_kv_valid != pending.base_E ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != pending.base_E) ||
            (speculative_backend == SpeculativeBackend::DFlash2 &&
             sequence.dflash_context_frontier != pending.base_E)) {
            throw std::logic_error("speculative pending row is not at its recorded base");
        }
        const std::uint32_t committed = cancelled[row] ? 0U : accepted_tokens[row];
        if ((cancelled[row] && accepted_tokens[row] != 0) ||
            (!cancelled[row] && (committed == 0 || committed > pending.produced ||
                                 (!terminal[row] && committed != pending.produced)))) {
            throw std::logic_error("speculative pending row has an invalid committed prefix");
        }
        const StateImageSelectors selectors = state_selectors(sequence);
        fold_rows[row] =
            ops::GdnReplayFoldRow{.source_state_slot      = selectors.source,
                                  .destination_state_slot = selectors.destination,
                                  .commit_columns         = static_cast<std::int32_t>(committed)};
        const bool partial_terminal =
            !cancelled[row] && terminal[row] && committed < pending.produced;
        hidden_selectors[row] =
            static_cast<std::int32_t>(partial_terminal ? committed - 1U : pending.produced - 1U);
        needs_hidden_correction = needs_hidden_correction || partial_terminal;
    }

    const auto tail_started = Clock::now();
    try {
        timing.resume_submit();
        replay_fold->execute(std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                             device.stream);
        // Qwen4Exp: the PLE convolution history advances by the same committed prefixes.
        fold_qwen4_ple(std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()));
        if (peer) {
            // Rank 1 folds its own records into its own GDN state with the same rows and commit
            // counts: the accepted prefix belongs to the round, and the two ranks' records differ
            // only in which heads and channels they cover.
            if (!peer->replay_fold) {
                throw std::logic_error("tensor-parallel speculative batch has no rank 1 records");
            }
            const ScopedCurrentDevice rank1(peer->device.device);
            peer->replay_fold->execute(
                std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                peer->device.stream);
        }

        // Sparse acceptance reads counts. Publish only the prefix licensed by the Frontend.
        if (speculative_backend == SpeculativeBackend::DFlash2) {
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (cancelled[row] || !requests[lanes[row]].sampling_host.token_counts) {
                    continue;
                }
                const auto count = static_cast<std::int32_t>(accepted_tokens[row]);
                Tensor ids =
                    io.dflash_decode->licensed_tokens.slice(1, static_cast<std::int32_t>(row), 1)
                        .slice(0, 0, count)
                        .view({count});
                Tensor counts =
                    token_counts.slice(1, static_cast<std::int32_t>(lanes[row]), 1)
                        .view({dimension(parameters.model.resources().public_token_count)});
                ops::increment_token_counts(ids, counts, device.stream);
            }
        }

        if (needs_hidden_correction) {
            const auto batch = static_cast<std::int32_t>(lanes.size());
            Tensor selector_tensor;
            Tensor hidden;
            Tensor selected;
            Tensor destinations;
            if (speculative_backend == SpeculativeBackend::Mtp && io.mtp_decode) {
                qwen3_5::MtpDecodeState& frame = *io.mtp_decode;
                selector_tensor                = frame.current_extents.slice(0, 0, batch);
                hidden                         = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.state_destination_slots.slice(0, 0, batch);
            } else if (speculative_backend == SpeculativeBackend::DFlash2 && io.dflash_decode) {
                qwen3_5::DFlashDecodeState& frame = *io.dflash_decode;
                selector_tensor                   = frame.proposal_extents.slice(0, 0, batch);
                hidden                            = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.state_destination_slots.slice(0, 0, batch);
            } else {
                throw std::logic_error("partial speculative commit has no target frame");
            }
            CUDA_CHECK(cudaMemcpyAsync(selector_tensor.data, hidden_selectors.data(),
                                       lanes.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device.stream));
            ops::speculative_select_accepted_hidden(hidden, selector_tensor, selected,
                                                    device.stream);
            ops::scatter(selected, destinations, state_images->continuation_hidden_store(),
                         device.stream);
            if (peer_retains_hidden()) {
                // Rank 1's round verified the same columns into its own frame; it retains the
                // same committed column of its own hidden in the same slots.
                if (!peer->io.mtp_decode) {
                    throw std::logic_error("tensor-parallel MTP commit has no rank 1 frame");
                }
                qwen3_5::MtpDecodeState& peer_frame = *peer->io.mtp_decode;
                Tensor peer_selectors               = peer_frame.current_extents.slice(0, 0, batch);
                Tensor peer_selected = peer_frame.target_continuation_hidden.slice(1, 0, batch);
                const ScopedCurrentDevice rank1(peer->device.device);
                CUDA_CHECK(cudaMemcpyAsync(peer_selectors.data, hidden_selectors.data(),
                                           lanes.size() * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, peer->device.stream));
                ops::speculative_select_accepted_hidden(peer_frame.target_hidden.slice(2, 0, batch),
                                                        peer_selectors, peer_selected,
                                                        peer->device.stream);
                ops::scatter(peer_selected, peer_frame.state_destination_slots.slice(0, 0, batch),
                             peer->state_images->continuation_hidden_store(), peer->device.stream);
            }
        }

        if (speculative_backend == SpeculativeBackend::DFlash2) {
            std::array<std::uint32_t, kMaximumConcurrency> append_lanes{};
            std::array<std::uint32_t, kMaximumConcurrency> append_starts{};
            std::array<std::uint32_t, kMaximumConcurrency> append_counts{};
            std::size_t append_size = 0;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (!cancelled[row] && terminal[row]) {
                    append_lanes[append_size]  = lanes[row];
                    append_starts[append_size] = requests[lanes[row]].pending.base_E;
                    append_counts[append_size] = accepted_tokens[row];
                    ++append_size;
                }
            }
            if (append_size != 0) {
                enqueue_dflash_context_append(
                    std::span<const std::uint32_t>(append_lanes.data(), append_size),
                    std::span<const std::uint32_t>(append_starts.data(), append_size),
                    std::span<const std::uint32_t>(append_counts.data(), append_size));
            }
        }

        // T49: a Qwen4Exp MTP commit that keeps every lane active does not wait for its fold:
        // the fold and the PLE advance are stream-ordered before the next round on each rank's
        // stream, read only kernel arguments and arena memory that later work reaches in stream
        // order, and the host prepares the next round (PLE rows, expert window) meanwhile. A
        // device fault in them surfaces at that round's synchronization. Terminal or cancelled
        // rows release resources below and keep the wait.
        bool defer_wait = speculative_backend == SpeculativeBackend::Mtp &&
                          parameters.model.config().text.qwen4() && !needs_hidden_correction;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            defer_wait = defer_wait && terminal[row] == 0 && cancelled[row] == 0;
        }
        if (defer_wait && peer) {
            // The next round is one graph launched on rank 0's stream: it must also follow rank
            // 1's fold, which ran on rank 1's stream.
            const ScopedCurrentDevice rank1(peer->device.device);
            if (!qwen4_peer_commit_event) { qwen4_peer_commit_event.emplace(peer->device); }
            qwen4_peer_commit_event->record(peer->device.stream);
            qwen4_peer_commit_event->wait(device.stream);
        }
        if (!defer_wait) {
            timing.begin_wait();
            synchronize_devices();
            timing.end_wait();
        }
        work.reset();
    } catch (...) {
        try {
            synchronize_devices();
        } catch (...) {}
        work.reset();
        clear_execution_failure_lanes(lanes);
        throw;
    }

    const double tail_seconds = std::chrono::duration<double>(Clock::now() - tail_started).count();
    const std::uint32_t width = draft_window + 1U;
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = active_sequence(lanes[row]);
            RequestControl& request = requests[lanes[row]];
            if (cancelled[row]) {
                if (!clear_lane_strict(sequence, request)) {
                    throw std::logic_error("cancelled speculative lane is not strictly releasable");
                }
                continue;
            }

            const PendingCandidate pending = request.pending;
            const std::uint32_t committed  = accepted_tokens[row];
            settle_state_fork(sequence);
            const TokenId* token_base =
                speculative_backend == SpeculativeBackend::Mtp
                    ? mtp_host_egress->licensed_tokens.data() + row * width
                    : dflash_host_egress->licensed_tokens.data() + row * width;
            sequence.ledger.insert(sequence.ledger.end(), token_base, token_base + committed);
            commit_generated_prefix_identity(sequence, pending.base_S,
                                             std::span<const TokenId>(token_base, committed),
                                             prefix_execution_splits[row]);
            advance_rebuild_work(sequence, pending.base_E + committed, prefill_chunk);
            sequence.execution_frontier = pending.base_E + committed;
            sequence.ledger_frontier    = pending.base_S + committed;
            sequence.text_kv_valid      = sequence.execution_frontier;
            sequence.tail_hidden_valid  = true;

            if (speculative_backend == SpeculativeBackend::Mtp) {
                sequence.mtp_kv_valid = sequence.execution_frontier;
                if (terminal[row]) {
                    sequence.mtp_draft_count = 0;
                } else {
                    const std::int32_t next = mtp_host_egress->next_extents[row];
                    if (qwen4_mtp_calib != nullptr) {
                        const auto extent = mtp_host_ingress->current_extents[row];
                        if (extent > 0) {
                            std::fprintf(qwen4_mtp_calib, "%d %u", extent, committed - 1U);
                            for (std::uint32_t k = 0; k < draft_window; ++k) {
                                std::fprintf(qwen4_mtp_calib, " %.5g",
                                             sequence.mtp_draft_logprobs[k]);
                            }
                            std::fputc('\n', qwen4_mtp_calib);
                        }
                        for (std::uint32_t k = 0; k < draft_window; ++k) {
                            sequence.mtp_draft_logprobs[k] =
                                mtp_host_egress->next_draft_logprobs[k * max_concurrency + row];
                        }
                    }
                    sequence.mtp_draft_count = static_cast<std::uint32_t>(next);
                    if (mtp_min_p > 0.0F) {
                        // T41 adaptive window: the next round verifies the proposals up to the
                        // first whose probability (chain product or its own) is below the floor.
                        const float floor  = std::log(mtp_min_p);
                        float chain        = 0.0F;
                        std::uint32_t keep = 0;
                        for (; keep < sequence.mtp_draft_count; ++keep) {
                            const float lp =
                                mtp_host_egress->next_draft_logprobs[keep * max_concurrency + row];
                            chain = mtp_window_rule == MtpWindowRule::Step ? lp : chain + lp;
                            if (!(chain >= floor)) { break; }
                        }
                        sequence.mtp_draft_count = keep;
                    }
                    for (std::uint32_t step = 0; step < sequence.mtp_draft_count; ++step) {
                        sequence.mtp_drafts[step] =
                            mtp_host_egress->next_drafts[step * max_concurrency + row];
                    }
                }
            } else {
                sequence.dflash_context_frontier =
                    terminal[row] ? sequence.execution_frontier : pending.base_E;
            }

            commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            if (terminal[row]) {
                request.lifecycle = Lifecycle::Finishable;
            } else {
                request.lifecycle = Lifecycle::Active;
            }
            request.pending = {};
            request.timings.decode_seconds += tail_seconds;
        }
    } catch (...) {
        clear_execution_failure_lanes(lanes);
        throw;
    }
    return timing.finish();
}

runtime::PrefillStepResult ProgramImpl::advance_prefill(SequenceState& sequence,
                                                        RequestControl& request,
                                                        runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *request.prefill;
    if (staged.pending_capture_offer != 0) {
        throw std::logic_error("prefill cannot advance while a capture offer is pending");
    }
    const runtime::BeginSummary summary{.prompt_tokens        = staged.prompt_tokens,
                                        .reused_prompt_tokens = staged.base,
                                        .prefix_reuse_path    = staged.reuse};
    std::uint32_t processed_prompt_tokens = 0;
    const auto started                    = Clock::now();
    try {
        // Disk tier (v0.7.2): a root prefill first loads the longest saved checkpoint its prompt
        // begins with; the loaded tokens count as processed (the admitted Begin stays Root).
        if (disk_kv && !staged.disk_attempted) {
            processed_prompt_tokens += disk_cache_resume(sequence, request);
        }
        if (staged.next_capture < staged.capture_groups.size() &&
            staged.capture_groups[staged.next_capture].frontier == staged.cursor) {
            if (staged.cursor != staged.base ||
                !staged.capture_groups[staged.next_capture].shared ||
                staged.capture_groups[staged.next_capture].rewrite ||
                staged.capture_groups[staged.next_capture].long_anchor) {
                throw std::logic_error("zero-prefill capture is not a shared base promotion");
            }
            // An exact-hit restore that had to Fork (its StateImage is still referenced elsewhere,
            // e.g. a turn anchor after a reasoning-close continuation) has written nothing yet;
            // the first prefill chunk would close it, but this promotion publishes the base before
            // any chunk runs and a capture cannot be reserved over an unsettled Fork. Close it here
            // the way a terminal Begin does (decode.cpp).
            if (sequence.state.fork_pending) {
                const StateImageSelectors selectors = state_selectors(sequence);
                state_images->copy_slot(selectors.source, selectors.destination, device.stream);
                device.synchronize();
                settle_state_fork(sequence);
            }
            if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
            staged.pending_capture_offer = next_capture_offer_id_;
            return runtime::PrefillStepResult{
                .summary = summary,
                .timing  = timing.finish(),
            };
        }
        StateImageSelectors selectors = state_selectors(sequence);
        Tensor rewrite_capture_hidden;
        Tensor peer_rewrite_capture_hidden;
        Tensor* rewrite_capture_hidden_ptr      = nullptr;
        Tensor* peer_rewrite_capture_hidden_ptr = nullptr;
        // Each chunk that may reach a capture frontier retains the frontier's hidden in the
        // destination StateImage; under MTP at tp 2 rank 1 retains its copy in the same slot.
        const auto bind_rewrite_capture_hidden = [&] {
            rewrite_capture_hidden_ptr      = nullptr;
            peer_rewrite_capture_hidden_ptr = nullptr;
            if (staged.next_capture >= staged.capture_groups.size()) { return; }
            rewrite_capture_hidden = state_images->continuation_hidden_slot(selectors.destination);
            rewrite_capture_hidden_ptr = &rewrite_capture_hidden;
            if (peer_retains_hidden()) {
                peer_rewrite_capture_hidden =
                    peer->state_images->continuation_hidden_slot(selectors.destination);
                peer_rewrite_capture_hidden_ptr = &peer_rewrite_capture_hidden;
            }
        };
        bind_rewrite_capture_hidden();
        const std::optional<execution::TpExecution> tp = prefill_tp_binding(sequence);
        execution::PrefillContext schedule_state{
            {device, parameters, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, tp ? &*tp : nullptr, graph_peer_bridge(), qwen4_runtime()},
            text_kv_view(sequence),
            mtp_kv_view(sequence),
            decoder->text_kv,
            decoder->mtp_cache(),
            dflash ? &*dflash : nullptr,
            staged.cursor,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            rewrite_capture_hidden_ptr,
            selectors.source,
            selectors.destination,
            staged.initial_mtp_extent,
            dflash_prefill_host_ingress,
            peer_rewrite_capture_hidden_ptr};
        schedule_state.rope_delta = sequence.rope_delta;
        // Forced tokens of another lane may have repointed the prefill KV row scalars since this
        // lane's bind; every prefill step publishes its own rows on every rank.
        publish_kv_rows(sequence);

        if (staged.mtp_bridge == MtpBridgeMode::BeforeSuffix) {
            // After a disk resume the bridge runs at the loaded frontier.
            const std::uint32_t bridge_base =
                staged.disk_base != 0 ? staged.disk_base : staged.base;
            if (staged.cursor != bridge_base || bridge_base == 0 ||
                staged.cursor >= staged.prompt_tokens) {
                throw std::logic_error("staged MTP bridge is outside the reusable suffix");
            }
            mark_workspace_usage(workspace_plan.mtp_prefill);
            const Tensor& previous_hidden = sequence.tail_hidden;
            const Tensor peer_previous_hidden =
                peer_retains_hidden() ? peer_tail_hidden(sequence) : Tensor{};
            const execution::MtpBridgeInput bridge{
                .previous_hidden      = &previous_hidden,
                .peer_previous_hidden = peer_retains_hidden() ? &peer_previous_hidden : nullptr,
                .position             = checked_i32(bridge_base - 1, "MTP bridge position"),
                .rope_position        = prompt_rope_position(staged.prompt, bridge_base - 1),
            };
            if (staged.vision) {
                in_qwen4_window([&] {
                    execution::mtp_bridge_multimodal(schedule_state, staged.prompt, *staged.vision,
                                                     bridge);
                });
            } else {
                Tensor bridge_token = io.mtp->target_input_ids.slice(0, 0, 1);
                const TokenId token = staged.prompt.token_ids[bridge_base];
                CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                           cudaMemcpyHostToDevice, device.stream));
                in_qwen4_window([&] {
                    execution::mtp_bridge_and_propose(schedule_state, bridge_token, previous_hidden,
                                                      peer_retains_hidden() ? &peer_previous_hidden
                                                                            : nullptr,
                                                      bridge.position, bridge.rope_position, false);
                });
            }
            sequence.mtp_kv_valid = bridge_base;
            commit_sequence_kv(sequence, sequence.text_kv_valid, sequence.mtp_kv_valid);
            staged.mtp_bridge = MtpBridgeMode::None;
        }

        // Structured output: the call that samples the first token binds the lane's mask into the
        // lane's sampling config (the matcher does not move during the prompt) and keeps the
        // reasoning EOS guard off, so nothing redraws outside the mask.
        if (row_constrained(0)) {
            schedule_state.reasoning_guard = false;
            if (staged.cursor + std::min(prefill_chunk, staged.prompt_tokens - staged.cursor) >=
                staged.prompt_tokens) {
                bind_prefill_grammar_mask(sequence, request);
            }
        }

        if (staged.cursor < staged.prompt_tokens) {
            const std::uint32_t nominal =
                std::min(prefill_chunk, staged.prompt_tokens - staged.cursor);
            mark_workspace_usage(staged.prepare_mtp ? workspace_plan.mtp_prefill
                                                    : workspace_plan.text_prefill);
            if (speculative_backend == SpeculativeBackend::DFlash2) {
                mark_workspace_usage(workspace_plan.dflash_context);
            }
            std::uint32_t remaining          = nominal;
            std::uint32_t final_chunk_tokens = 0;
            bool finalized                   = false;
            while (remaining != 0) {
                schedule_state.text_kv_base           = staged.cursor;
                selectors                             = state_selectors(sequence);
                schedule_state.state_source_slot      = selectors.source;
                schedule_state.state_destination_slot = selectors.destination;
                bind_rewrite_capture_hidden();
                schedule_state.rewrite_checkpoint_hidden      = rewrite_capture_hidden_ptr;
                schedule_state.peer_rewrite_checkpoint_hidden = peer_rewrite_capture_hidden_ptr;

                const bool final_candidate = staged.cursor + remaining == staged.prompt_tokens;
                const std::optional<std::uint32_t> capture_frontier =
                    staged.next_capture < staged.capture_groups.size()
                        ? std::optional<std::uint32_t>(
                              staged.capture_groups[staged.next_capture].frontier)
                        : std::nullopt;
                std::optional<std::uint32_t> split_frontier = capture_frontier;
                const auto rewrite_split                    = std::upper_bound(
                    staged.prompt.identity.rewrite_execution_frontiers.begin(),
                    staged.prompt.identity.rewrite_execution_frontiers.end(), staged.cursor);
                // Routed experts cost each prefill call an expert pass (streamed over PCIe,
                // or the CPU path below 2 048 columns): splitting at every rewrite frontier (one
                // per assistant turn) made a 10-turn 1.2k-token prompt 2.4x slower in
                // ninfer-serve. Rewrite frontiers only keep resumed and root GDN decompositions
                // bit-identical (they are not captures), so a routed-expert model splits at
                // capture frontiers only; NINFER_QWEN4_REWRITE_SPLITS=1 restores them.
                if (rewrite_split != staged.prompt.identity.rewrite_execution_frontiers.end() &&
                    (!split_frontier || *rewrite_split < *split_frontier) &&
                    (!parameters.model.config().text.qwen4_routed_experts() ||
                     qwen4_rewrite_splits())) {
                    split_frontier = *rewrite_split;
                }
                execution::PrefillChunkResult result;
                timing.pause();
                if (staged.vision) {
                    const VisionWorkspacePlan* vision_plan =
                        workspace_plan.rank_vision(0, vision_rank);
                    if (vision_plan == nullptr) {
                        throw std::logic_error("active Vision prefill lost its workspace plan");
                    }
                    mark_workspace_usage(vision_plan->capacity_bytes);
                    result = execution::prefill_multimodal_chunk(schedule_state, staged.prompt,
                                                                 *staged.vision, remaining,
                                                                 split_frontier, final_candidate);
                } else {
                    result = execution::prefill_text_chunk(
                        schedule_state, std::span<const TokenId>(staged.prompt.token_ids),
                        remaining, split_frontier, final_candidate);
                }
                timing.include(result.timing);
                timing.resume_post();
                if (result.processed_tokens == 0 || result.processed_tokens > remaining) {
                    throw std::logic_error("ordinary prefill chunk made invalid progress");
                }
                if (staged.vision) { staged.vision->release_encoded_media_payloads(); }
                staged.cursor += result.processed_tokens;
                processed_prompt_tokens += result.processed_tokens;
                remaining -= result.processed_tokens;
                final_chunk_tokens     = result.processed_tokens;
                sequence.text_kv_valid = staged.cursor;
                if (staged.prepare_mtp) { sequence.mtp_kv_valid = staged.cursor; }
                if (speculative_backend == SpeculativeBackend::DFlash2) {
                    sequence.dflash_context_frontier = staged.cursor;
                }
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));

                // Prompt transitions are canonical immediately. If this was the first write after
                // an immutable source, close the Fork before potentially freezing a new rewrite.
                settle_state_fork(sequence);
                const bool reached_capture = capture_frontier && staged.cursor == *capture_frontier;
                if (reached_capture) {
                    if (result.finalized) {
                        // The prompt-frontier state becomes publishable only after the generated
                        // Begin token is committed. commit() emits the offer for this group.
                    } else {
                        staged.elapsed_seconds +=
                            std::chrono::duration<double>(Clock::now() - started).count();
                        if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
                        staged.pending_capture_offer = next_capture_offer_id_;
                        return runtime::PrefillStepResult{
                            .summary                 = summary,
                            .processed_prompt_tokens = processed_prompt_tokens,
                            .timing                  = timing.finish(),
                        };
                    }
                }

                finalized = result.finalized;
                if (finalized || remaining == 0) { break; }
            }

            if (!finalized) {
                if (staged.cursor == staged.prompt_tokens) {
                    throw std::logic_error("staged prefill reached the prompt without sampling");
                }
                staged.elapsed_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                return runtime::PrefillStepResult{
                    .summary                 = summary,
                    .processed_prompt_tokens = processed_prompt_tokens,
                    .timing                  = timing.finish(),
                };
            }
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("staged prefill sampled before the prompt frontier");
            }
            timing.resume_submit();
            copy_tail(sequence, static_cast<std::int32_t>(final_chunk_tokens) - 1);
        } else {
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!sequence.tail_hidden_valid) {
                throw std::logic_error("zero-suffix reuse has no target tail hidden");
            }
            const Tensor peer_sample_hidden =
                peer_retains_hidden() ? peer_tail_hidden(sequence) : Tensor{};
            execution::sample_from_hidden(
                schedule_state, sequence.tail_hidden,
                checked_i32(staged.prompt_tokens, "sample position"), ops::kSamplePurposePrefill,
                peer_retains_hidden() ? &peer_sample_hidden : nullptr,
                schedule_state.reasoning_guard &&
                    execution::reasoning_eos_guard_applies(staged.prompt.token_ids));
            set_device_i32(io.rope_pos, checked_i32(staged.prompt_tokens, "rope position") +
                                            sequence.rope_delta);
            if (staged.prepare_mtp) {
                if (staged.mtp_bridge != MtpBridgeMode::AfterExactHit) {
                    throw std::logic_error("zero-suffix MTP reuse has no exact-hit bridge");
                }
                mark_workspace_usage(workspace_plan.mtp_prefill);
                const auto bridge_rope =
                    prompt_rope_position(staged.prompt, staged.prompt_tokens - 1);
                const Tensor peer_previous_hidden =
                    peer_retains_hidden() ? peer_tail_hidden(sequence) : Tensor{};
                in_qwen4_window([&] {
                    execution::mtp_bridge_and_propose(
                        schedule_state, io.token, sequence.tail_hidden,
                        peer_retains_hidden() ? &peer_previous_hidden : nullptr,
                        checked_i32(staged.prompt_tokens - 1, "MTP full-prefix bridge position"),
                        bridge_rope, staged.initial_mtp_extent != 0);
                });
                sequence.mtp_kv_valid = staged.prompt_tokens;
                commit_sequence_kv(sequence, sequence.text_kv_valid, sequence.mtp_kv_valid);
                staged.mtp_bridge = MtpBridgeMode::None;
            }
        }

        copy_round_token();
        std::array<TokenId, qwen3_5::kMtpDecodeMaximumDrafts> initial_drafts{};
        if (staged.prepare_mtp && staged.initial_mtp_extent != 0) {
            CUDA_CHECK(cudaMemcpyAsync(initial_drafts.data(), io.mtp->draft_tokens.data,
                                       staged.initial_mtp_extent * sizeof(TokenId),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        timing.begin_wait();
        synchronize_devices();
        timing.end_wait();
        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        const double vision_seconds       = staged.vision ? staged.vision->elapsed_seconds() : 0.0;
        const std::uint32_t prompt_tokens = staged.prompt_tokens;

        if (qwen4_tfdump) {
            // Validation: teacher-forced first token (NINFER_QWEN4_TFDUMP_FORCE).
            if (const std::int32_t forced = qwen4_tfdump->forced_token(prompt_tokens);
                forced >= 0) {
                host_tokens[0] = forced;
            }
        }
        if (qwen4_replay) {
            // T59 replay: the reference's first token after the prompt.
            if (const execution::Qwen4ReplaySequence* ref = qwen4_replay->match(sequence.ledger);
                ref != nullptr && sequence.ledger.size() == prompt_tokens) {
                host_tokens[0] = ref->ids[prompt_tokens];
                ++qwen4_replay->forced_tokens;
            }
        }
        if (qwen4_replay_record != nullptr) {
            std::fprintf(qwen4_replay_record, "req %u\n", static_cast<unsigned>(prompt_tokens));
        }
        qwen4_round_follows = false; // T59 round log: the next round starts a request
        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (sequence.ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        sequence.ledger.push_back(host_tokens[0]);
        sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        sequence.prefix_digests.append_generated(std::span<const TokenId>(host_tokens, 1),
                                                 sequence.rope_delta);
        sequence.text_kv_valid = prompt_tokens;
        if (staged.prepare_mtp) {
            if (sequence.mtp_kv_valid != prompt_tokens) {
                throw std::logic_error("staged MTP prefill did not reach the prompt frontier");
            }
            sequence.mtp_draft_count = staged.initial_mtp_extent;
            // The prompt's proposals carry no recorded probability (T41 calibration skips NaN).
            sequence.mtp_draft_logprobs.fill(std::numeric_limits<float>::quiet_NaN());
            std::copy_n(initial_drafts.begin(), staged.initial_mtp_extent,
                        sequence.mtp_drafts.begin());
        } else if (speculative_backend == SpeculativeBackend::DFlash2 &&
                   sequence.dflash_context_frontier != prompt_tokens) {
            throw std::logic_error("staged DFlash prefill did not reach the prompt frontier");
        }
        sequence.tail_hidden_valid      = true;
        request.timings.vision_seconds  = vision_seconds;
        request.timings.prefill_seconds = std::max(0.0, staged.elapsed_seconds - vision_seconds);
        staged.prompt.release_all_media_payloads();
        if (staged.vision) { staged.vision->retire_handoff(); }

        const bool prompt_frontier_capture =
            staged.next_capture < staged.capture_groups.size() &&
            staged.capture_groups[staged.next_capture].frontier == prompt_tokens;
        if (!prompt_frontier_capture) { request.prefill.reset(); }
        request.pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                             .base_E        = 0,
                                             .base_S        = 0,
                                             .prompt_tokens = prompt_tokens,
                                             .produced      = 1};
        request.lifecycle = Lifecycle::Pending;
        return runtime::PrefillStepResult{
            .summary = summary,
            .round   = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
            .processed_prompt_tokens = processed_prompt_tokens,
            .complete                = true,
            .timing                  = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            synchronize_devices();
        } catch (...) {}
        timing.end_wait();
        const std::uint32_t lane = sequence.lane;
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        throw;
    }
}


} // namespace ninfer::models::qwen3_5::detail

#include "models/qwen3_5/program/tp2/prefill_tp2.inc"
