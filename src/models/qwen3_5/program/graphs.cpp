#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "core/nvtx.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace {

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label);

std::size_t free_device_bytes(int device) {
    const ScopedCurrentDevice scope(device);
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

template <class Prepare, class Synchronize>
void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device,
                              Prepare&& prepare, Synchronize&& synchronize);

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label) {
    if (profiles.empty() || profiles.front().min != 0 || profiles.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        if (profiles[i].min > profiles[i].max ||
            (i != 0 && profiles[i].min != profiles[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage has a gap");
        }
    }
}

// `synchronize` waits for every device a graph has nodes on: a tensor-parallel executable is
// uploaded and launched from rank 0's stream as one unit, but its rank 1 nodes retire on rank 1.
template <class Prepare, class Synchronize>
void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device,
                              Prepare&& prepare, Synchronize&& synchronize) {
    if (family.profiles.empty()) {
        throw std::logic_error(std::string(label) + " CUDA Graph family has no profiles");
    }

    for (std::size_t i = 0; i < family.profiles.size(); ++i) {
        DecodeGraphProfile& profile = family.profiles[i];
        if (!profile.definition.ready()) {
            throw std::logic_error(std::string(label) + " CUDA Graph definition is empty");
        }
        const auto existing =
            std::find_if(family.topologies.begin(), family.topologies.end(),
                         [&](const DecodeGraphTopology& topology) {
                             return topology.topology_class == profile.topology_class;
                         });
        if (existing != family.topologies.end()) { continue; }

        family.topologies.emplace_back();
        DecodeGraphTopology& topology = family.topologies.back();
        topology.topology_class       = profile.topology_class;
        topology.executable.instantiate(profile.definition);
        topology.installed_profile = i;
    }

    const auto install_and_upload = [&](DecodeGraphTopology& topology, std::size_t profile_index) {
        if (topology.installed_profile != profile_index) {
            topology.executable.update(family.profiles[profile_index].definition);
            topology.installed_profile = profile_index;
        }
        topology.executable.upload(device.stream);
        synchronize();
    };

    for (DecodeGraphTopology& topology : family.topologies) {
        std::optional<std::size_t> first_profile;
        for (std::size_t i = 0; i < family.profiles.size(); ++i) {
            if (family.profiles[i].topology_class == topology.topology_class) {
                if (!first_profile) {
                    first_profile = i;
                    install_and_upload(topology, i);

                    DecodeGraphProfile& profile = family.profiles[i];
                    prepare(profile.min_execution_frontier, profile.batch_size);
                    synchronize();
                    topology.executable.launch(device.stream);
                    synchronize();
                    continue;
                }
                install_and_upload(topology, i);
            }
        }
        if (!first_profile) {
            throw std::logic_error(std::string(label) + " CUDA Graph topology has no definitions");
        }
        if (topology.installed_profile != *first_profile) {
            install_and_upload(topology, *first_profile);
        }
    }
}

} // namespace

void ProgramImpl::prepare_graphs() {
    if (!use_cuda_graph) { return; }
    nvtx::ScopedRange prepare_range(nvtx::Name::CudaGraphPrepare, nvtx::Category::Graph);

    // The planned graph allowance covers the installed executables and the driver and module
    // state the eager warmups and captures materialize, so the observation spans all of it, on
    // every rank. It is reported, not enforced.
    const std::size_t ranks = peer ? 2U : 1U;
    const auto rank_device  = [&](std::size_t rank) {
        return rank == 0 ? device.device : peer->device.device;
    };
    synchronize_devices();
    std::array<std::size_t, 2> free_before{};
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        free_before[rank] = free_device_bytes(rank_device(rank));
    }

    std::array<StateImageHandle, kMaximumConcurrency> capture_states{};
    for (std::uint32_t row = 0; row < max_concurrency; ++row) {
        std::optional<StateImageHandle> state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::bad_alloc(); }
        capture_states[row] = *state;
    }
    const auto capture_state_slot = [&](std::uint32_t row) {
        return state_store->physical_slot(capture_states.at(row));
    };

    std::vector<KVAddressSpaceHandle> text_capture_allocations;
    std::vector<KVAddressSpaceHandle> mtp_capture_allocations;
    const auto reserve_capture_rows = [&](qwen3_5::PagedKVCache& cache,
                                          KVAddressSpaceStore& addresses,
                                          std::vector<KVAddressSpaceHandle>& allocations,
                                          const char* label) {
        DeviceKVPagePool& pool       = cache.page_pool();
        KVExecutionTablePool& tables = cache.execution_tables();
        if (pool.capacity_pages() < max_concurrency) {
            throw std::invalid_argument(std::string(label) +
                                        " cannot provide one Paged KV page per concurrent request");
        }
        allocations.reserve(max_concurrency);
        for (std::uint32_t row = 0; row < max_concurrency; ++row) {
            std::optional<KVAddressSpaceHandle> allocation =
                addresses.create_active(1, static_cast<std::int32_t>(row), device.stream);
            if (!allocation) { throw std::bad_alloc(); }
            allocations.push_back(*allocation);
            addresses.ensure_mapped_to_tokens(*allocation, 1, device.stream);

            // Capture profiles exercise arbitrary context envelopes. Repeating each row's private
            // page across its temporary table keeps every dummy read/write address valid without
            // reserving C full contexts solely for graph construction.
            tables.publish_repeated(addresses.execution_row(*allocation).handle(),
                                    addresses.physical_page(*allocation, 0),
                                    tables.logical_page_capacity(), device.stream);
        }
    };
    reserve_capture_rows(decoder->text_kv, *text_kv_addresses, text_capture_allocations,
                         "target KV cache");
    if (speculative_backend == SpeculativeBackend::Mtp) {
        reserve_capture_rows(*decoder->mtp_cache(), *backend_kv_addresses, mtp_capture_allocations,
                             "MTP KV cache");
    }
    // Rank 1's capture rows, pages and StateImages follow through the mirrors attached at
    // construction.
    synchronize_devices();
    const auto synchronize_all = [&] { synchronize_devices(); };

    // The round controls every rank's captures read. The DFlash prefill count is added for rank 0
    // alone, whose drafter produces it.
    const auto stable_controls = [](const qwen3_5::RoundState& state) {
        std::vector<Tensor> controls{
            state.token,
            state.pos,
            state.rope_pos,
            state.rope_delta,
        };
        if (state.mtp) {
            controls.push_back(state.mtp->position);
            controls.push_back(state.mtp->draft_tokens);
            controls.push_back(state.mtp->target_input_ids);
            controls.push_back(state.mtp->target_positions);
        }
        return controls;
    };
    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls = stable_controls(io);
        if (io.dflash_prefill) { controls.push_back(io.dflash_prefill->local_append_count); }
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
        if (peer) {
            const std::vector<Tensor> peer_controls = stable_controls(peer->io);
            const ScopedCurrentDevice rank1(peer->device.device);
            for (const Tensor& tensor : peer_controls) {
                CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), peer->device.stream));
            }
        }
    };
    const auto zero_capture_pages =
        [&](qwen3_5::PagedKVCache& cache, const KVAddressSpaceStore& addresses,
            const std::vector<KVAddressSpaceHandle>& allocations, std::uint32_t batch_size) {
            std::vector<DeviceKVPageHandle> pages;
            pages.reserve(batch_size);
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                pages.push_back(addresses.physical_page(allocations[row], 0));
            }
            cache.page_pool().zero_pages(pages, device.stream);
        };
    const auto prepare_representative = [&](std::uint32_t frontier, std::uint32_t batch_size) {
        if (batch_size == 0 || batch_size > max_concurrency) {
            throw std::logic_error("CUDA Graph representative batch is invalid");
        }
        work.reset();
        if (peer) { peer->work.reset(); }
        clear_stable_controls();
        zero_capture_pages(decoder->text_kv, *text_kv_addresses, text_capture_allocations,
                           batch_size);
        if (decoder->mtp_cache() != nullptr) {
            zero_capture_pages(*decoder->mtp_cache(), *backend_kv_addresses,
                               mtp_capture_allocations, batch_size);
        }
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            state_images->zero_slot(capture_state_slot(row), device.stream);
            if (dflash) {
                const Tensor pending =
                    dflash->pending_features.slice(2, static_cast<std::int32_t>(row), 1);
                CUDA_CHECK(cudaMemsetAsync(pending.data, 0, pending.bytes(), device.stream));
            }
            if (peer && peer->dflash) {
                const ScopedCurrentDevice rank1(peer->device.device);
                const Tensor pending =
                    peer->dflash->pending_features.slice(2, static_cast<std::int32_t>(row), 1);
                CUDA_CHECK(cudaMemsetAsync(pending.data, 0, pending.bytes(), peer->device.stream));
            }
        }
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        if (io.mtp) {
            set_device_i32(io.mtp->position,
                           checked_i32(frontier, "graph representative MTP position"));
        }
        if (io.dflash_decode) {
            *dflash_host_ingress       = {};
            *dflash_host_egress        = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            const std::uint32_t width  = draft_window + 1U;
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                dflash_host_ingress->anchors[row] = 0;
                dflash_host_ingress->execution_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash frontier");
                dflash_host_ingress->context_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash context frontier");
                dflash_host_ingress->proposal_valid_columns[row] = static_cast<std::int32_t>(width);
                dflash_host_ingress->proposal_extents[row] = static_cast<std::int32_t>(extent);
                dflash_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t column = 0; column < width; ++column) {
                    dflash_host_ingress->target_rope_positions[row * width + column] =
                        checked_i32(frontier + std::min(column, extent),
                                    "graph representative DFlash target RoPE position");
                }
                dflash_host_ingress->text_kv_table_rows[row]      = static_cast<std::int32_t>(row);
                dflash_host_ingress->active_lanes[row]            = static_cast<std::int32_t>(row);
                dflash_host_ingress->state_source_slots[row]      = capture_state_slot(row);
                dflash_host_ingress->state_destination_slots[row] = capture_state_slot(row);
                dflash_host_ingress->sampling[row]                = {};
            }
        }
        if (io.mtp_decode) {
            *mtp_host_ingress          = {};
            *mtp_host_egress           = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            const std::uint32_t width  = draft_window + 1U;
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                mtp_host_ingress->anchors[row] = 0;
                mtp_host_ingress->base_frontiers[row] =
                    checked_i32(frontier, "graph representative MTP frontier");
                mtp_host_ingress->remaining_budgets[row] =
                    checked_i32(capacity, "graph representative MTP budget");
                mtp_host_ingress->current_extents[row] = static_cast<std::int32_t>(extent);
                mtp_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t step = 0; step < draft_window; ++step) {
                    mtp_host_ingress->current_drafts[row * draft_window + step] = 0;
                }
                for (std::uint32_t column = 0; column < width; ++column) {
                    mtp_host_ingress->target_rope_positions[row * width + column] =
                        checked_i32(frontier + std::min(column, extent),
                                    "graph representative MTP RoPE position");
                }
                mtp_host_ingress->text_kv_table_rows[row]      = static_cast<std::int32_t>(row);
                mtp_host_ingress->mtp_kv_table_rows[row]       = static_cast<std::int32_t>(row);
                mtp_host_ingress->state_source_slots[row]      = capture_state_slot(row);
                mtp_host_ingress->state_destination_slots[row] = capture_state_slot(row);
                mtp_host_ingress->rope_deltas[row]             = 0;
                mtp_host_ingress->sampling[row]                = {};
            }
        }
        if (io.ordinary) {
            *ordinary_host_ingress = {};
            *ordinary_host_egress  = {};
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                ordinary_host_ingress->tokens[row] = 0;
                ordinary_host_ingress->cache_positions[row] =
                    checked_i32(frontier, "graph representative ordinary position");
                ordinary_host_ingress->rope_positions[row] =
                    checked_i32(frontier, "graph representative ordinary RoPE position");
                ordinary_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                ordinary_host_ingress->state_source_slots[row] = capture_state_slot(row);
                ordinary_host_ingress->state_destination_slots[row] = capture_state_slot(row);
                ordinary_host_ingress->sampling[row]                = {};
            }
        }
        // The one execution that follows (eager warm-up or graph launch) does not stage n-gram
        // rows: release its PLE layer at once (Qwen4Runtime::decode_ple_staged).
        raise_qwen4_ple_staged();
    };
    const auto execution_core = [&] {
        return execution::ExecutionCore{device,
                                        parameters,
                                        work,
                                        state_images->linear(),
                                        replay_records ? &*replay_records : nullptr,
                                        io,
                                        prefill_hidden,
                                        prefill_chunk,
                                        proposal_head,
                                        tp_binding(),
                                        graph_peer_bridge(),
                                        qwen4_runtime()};
    };

    // Under WSL2 a captured exchange can hang in a graph's first launch although the startup
    // probe passed (issue #1: every MTP round, not the ordinary ones). degrade_peer_mailbox() then
    // steps the transport down and everything is captured again, instead of failing startup.
    // NINFER_TP_MAILBOX_FAULT=draft reports such a hang while the MTP draft phase is on the
    // mailbox, =any while any mailbox is attached: test aids for hardware where none happens.
    const char* fault_env         = std::getenv("NINFER_TP_MAILBOX_FAULT");
    const std::string_view fault  = fault_env != nullptr ? fault_env : "";
    const auto synchronize_graphs = [&] {
        if (peer_mailbox && (fault == "any" ||
                             (fault == "draft" && speculative_backend == SpeculativeBackend::Mtp &&
                              !tp_execution->staged_draft_collectives))) {
            peer_mailbox->report_hang();
        }
        synchronize_devices();
    };
    const auto capture_and_instantiate = [&] {
        if (speculative_backend == SpeculativeBackend::None) {
            const auto ordinary_profiles = ordinary_graph_profiles(capacity);
            validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
            const std::uint32_t ordinary_batch_limit = max_concurrency;
            execution::OrdinaryBatchContext ordinary_state{
                execution_core(),
                decoder->text_kv,
                *io.ordinary,
                *ordinary_host_ingress,
                *ordinary_host_egress,
                state_images->continuation_hidden_store(),
                peer ? &*peer->io.ordinary : nullptr};
            const GraphExecutionProfile code_warm = ordinary_profiles.front();
            // A module first touched inside a capture region cannot be loaded there, and batch
            // shape selects kernels. One eager batch-1 round warms the single-device code; at tp 2
            // every batch size is warmed eagerly on both devices before any capture.
            const std::uint32_t warm_batches = tensor_parallel() ? ordinary_batch_limit : 1U;
            for (std::uint32_t batch_size = 1; batch_size <= warm_batches; ++batch_size) {
                prepare_representative(code_warm.min, batch_size);
                synchronize_all();
                execution::ordinary_decode_batch(ordinary_state,
                                                 static_cast<std::int32_t>(batch_size),
                                                 {code_warm.min + 1, code_warm.max + 1}, nullptr);
                synchronize_all();
            }

            ordinary_graphs.profiles.reserve(ordinary_profiles.size() * ordinary_batch_limit);
            for (std::uint32_t batch_size = 1; batch_size <= ordinary_batch_limit; ++batch_size) {
                for (const GraphExecutionProfile planned : ordinary_profiles) {
                    ordinary_graphs.profiles.emplace_back();
                    DecodeGraphProfile& profile    = ordinary_graphs.profiles.back();
                    profile.batch_size             = batch_size;
                    profile.min_execution_frontier = planned.min;
                    profile.max_execution_frontier = planned.max;
                    profile.topology_class =
                        planned.topology_class * ordinary_batch_limit + (batch_size - 1U);
                    const ops::CausalAttentionExecutionEnvelope envelope{planned.min + 1,
                                                                         planned.max + 1};
                    execution::capture_ordinary_decode_batch(ordinary_state,
                                                             static_cast<std::int32_t>(batch_size),
                                                             envelope, profile.definition);
                }
            }
        }

        if (speculative_backend == SpeculativeBackend::Mtp) {
            const auto planned_profiles = mtp_graph_profiles(capacity, draft_window);
            validate_graph_profiles(planned_profiles, capacity - 1, "MTP");
            execution::MtpBatchContext mtp_state{
                execution_core(),
                decoder->text_kv,
                *decoder->mtp_cache(),
                *io.mtp_decode,
                *mtp_host_ingress,
                *mtp_host_egress,
                state_images->continuation_hidden_store(),
                peer ? &*peer->io.mtp_decode : nullptr,
                peer ? &peer->state_images->continuation_hidden_store() : nullptr,
                mtp_min_p > 0.0F || qwen4_mtp_calib != nullptr};
            const GraphExecutionProfile code_warm = planned_profiles.front();
            // As for ordinary rounds: every batch size is warmed eagerly on both devices at tp 2
            // before any capture, since batch shape selects kernels.
            const std::uint32_t warm_batches = tensor_parallel() ? max_concurrency : 1U;
            for (std::uint32_t batch_size = 1; batch_size <= warm_batches; ++batch_size) {
                prepare_representative(code_warm.min, batch_size);
                synchronize_all();
                execution::mtp_decode_batch(
                    mtp_state, static_cast<std::int32_t>(batch_size), draft_window,
                    mtp_causal_attention_envelopes(code_warm.max, draft_window, capacity), nullptr);
                synchronize_all();
            }

            mtp_graphs.profiles.reserve(planned_profiles.size() * max_concurrency);
            for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
                for (const GraphExecutionProfile planned : planned_profiles) {
                    mtp_graphs.profiles.emplace_back();
                    DecodeGraphProfile& profile    = mtp_graphs.profiles.back();
                    profile.batch_size             = batch_size;
                    profile.min_execution_frontier = planned.min;
                    profile.max_execution_frontier = planned.max;
                    profile.topology_class =
                        planned.topology_class * max_concurrency + (batch_size - 1U);
                    execution::capture_mtp_decode_batch(
                        mtp_state, static_cast<std::int32_t>(batch_size), draft_window,
                        mtp_causal_attention_envelopes(planned.max, draft_window, capacity),
                        profile.definition);
                }
            }
        }
        if (speculative_backend == SpeculativeBackend::DFlash2) {
            const auto planned_profiles = dflash_graph_profiles(capacity, draft_window);
            validate_graph_profiles(planned_profiles, capacity - 1, "DFlash");
            execution::DFlashBatchContext dflash_state{
                execution_core(),
                decoder->text_kv,
                *dflash,
                *io.dflash_decode,
                *dflash_host_ingress,
                *dflash_host_egress,
                state_images->continuation_hidden_store(),
                peer ? &*peer->io.dflash_decode : nullptr,
                peer ? &peer->state_images->continuation_hidden_store() : nullptr};
            const GraphExecutionProfile code_warm = planned_profiles.front();
            const ops::CausalAttentionExecutionEnvelope code_warm_target{
                1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                       capacity, static_cast<std::uint64_t>(code_warm.max) + draft_window + 1ULL))};
            // As for ordinary and MTP rounds: every batch size is warmed eagerly on both devices at
            // tp 2 before any capture, since batch shape selects kernels.
            const std::uint32_t warm_batches = tensor_parallel() ? max_concurrency : 1U;
            for (std::uint32_t batch_size = 1; batch_size <= warm_batches; ++batch_size) {
                prepare_representative(code_warm.min, batch_size);
                synchronize_all();
                execution::dflash_decode_batch(
                    dflash_state, static_cast<std::int32_t>(batch_size), draft_window,
                    dflash_envelopes(code_warm.min, code_warm.max, draft_window), code_warm_target,
                    nullptr);
                synchronize_all();
            }

            dflash_graphs.profiles.reserve(planned_profiles.size() * max_concurrency);
            for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
                for (const GraphExecutionProfile planned : planned_profiles) {
                    dflash_graphs.profiles.emplace_back();
                    DecodeGraphProfile& profile    = dflash_graphs.profiles.back();
                    profile.batch_size             = batch_size;
                    profile.min_execution_frontier = planned.min;
                    profile.max_execution_frontier = planned.max;
                    profile.topology_class =
                        planned.topology_class * max_concurrency + (batch_size - 1U);
                    const ops::CausalAttentionExecutionEnvelope target_envelope{
                        1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                               capacity,
                               static_cast<std::uint64_t>(planned.max) + draft_window + 1ULL))};

                    execution::capture_dflash_decode_batch(
                        dflash_state, static_cast<std::int32_t>(batch_size), draft_window,
                        dflash_envelopes(planned.min, planned.max, draft_window), target_envelope,
                        profile.definition);
                }
            }
        }

        if (!ordinary_graphs.profiles.empty()) {
            instantiate_graph_family(ordinary_graphs, "ordinary", device, prepare_representative,
                                     synchronize_graphs);
        }
        if (speculative_backend == SpeculativeBackend::Mtp) {
            instantiate_graph_family(mtp_graphs, "MTP", device, prepare_representative,
                                     synchronize_graphs);
        }
        if (speculative_backend == SpeculativeBackend::DFlash2) {
            instantiate_graph_family(dflash_graphs, "DFlash", device, prepare_representative,
                                     synchronize_graphs);
        }
    };
    for (;;) {
        try {
            capture_and_instantiate();
            break;
        } catch (const std::runtime_error&) {
            // synchronize_devices() found the hang word set: both devices are idle, the executables
            // are discarded by degrade_peer_mailbox() before the mailbox they address.
            if (!peer_mailbox || !peer_mailbox->hang_reported() ||
                !degrade_peer_mailbox("in the first launch of a CUDA graph")) {
                throw;
            }
        }
    }

    clear_stable_controls();
    state_images->zero_all(device.stream);
    if (dflash) {
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_features.data, 0,
                                   dflash->prefill_features.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_positions.data, 0,
                                   dflash->prefill_positions.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->pending_features.data, 0,
                                   dflash->pending_features.bytes(), device.stream));
    }
    if (peer && peer->dflash) {
        // The split drafter's rank 1 state, as rank 0's above (its rings: state_images' mirror).
        const ScopedCurrentDevice rank1(peer->device.device);
        for (const Tensor* t : {&peer->dflash->prefill_features, &peer->dflash->prefill_positions,
                                &peer->dflash->pending_features}) {
            CUDA_CHECK(cudaMemsetAsync(t->data, 0, t->bytes(), peer->device.stream));
        }
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    synchronize_devices();
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        const std::size_t free_after = free_device_bytes(rank_device(rank));
        graph_observed_bytes[rank] =
            free_before[rank] > free_after ? free_before[rank] - free_after : 0;
    }
    for (std::uint32_t row = 0; row < max_concurrency; ++row) {
        if (!state_store->release(capture_states[row])) {
            throw std::logic_error("CUDA Graph capture StateImage could not be released");
        }
    }

    const auto release_capture_rows = [](KVAddressSpaceStore& addresses,
                                         std::vector<KVAddressSpaceHandle>& allocations) {
        for (const KVAddressSpaceHandle allocation : allocations) {
            addresses.deactivate(allocation);
            if (!addresses.release(allocation)) {
                throw std::logic_error("CUDA Graph capture KV address space could not be released");
            }
        }
        allocations.clear();
    };
    if (!mtp_capture_allocations.empty()) {
        release_capture_rows(*backend_kv_addresses, mtp_capture_allocations);
    }
    release_capture_rows(*text_kv_addresses, text_capture_allocations);
}


} // namespace ninfer::models::qwen3_5::detail
