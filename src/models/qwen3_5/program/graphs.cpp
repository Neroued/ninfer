#include "models/qwen3_5/program/program_impl.h"
#include "models/common/context/work.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "core/nvtx.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace {

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label);

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

void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device) {
    if (family.profiles.empty()) {
        throw std::logic_error(std::string(label) + " CUDA Graph family has no profiles");
    }

    for (std::size_t i = 0; i < family.profiles.size(); ++i) {
        DecodeGraphProfile& profile = family.profiles[i];
        if (!profile.text && !profile.definition.ready()) {
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
        if (profile.text) {
            const execution::TextGraphExecutable* shared = nullptr;
            for (const auto& other : family.topologies) {
                if (other.text &&
                    family.profiles[*other.installed_profile].batch_size == profile.batch_size) {
                    shared = other.text.get();
                    break;
                }
            }
            topology.text = std::make_unique<execution::TextGraphExecutable>(*profile.text, shared);
        } else {
            topology.executable.instantiate(profile.definition);
        }
        topology.installed_profile = i;
    }

    const auto install_and_upload = [&](DecodeGraphTopology& topology, std::size_t profile_index) {
        DecodeGraphProfile& profile = family.profiles[profile_index];
        if (topology.installed_profile != profile_index) {
            if (profile.text)
                topology.text->update(*profile.text);
            else
                topology.executable.update(profile.definition);
            topology.installed_profile = profile_index;
        }
        if (topology.text) {
            topology.text->activate(*profile.text);
            topology.text->upload(device.stream());
        } else {
            topology.executable.upload(device.stream());
        }
        device.synchronize();
    };

    for (DecodeGraphTopology& topology : family.topologies) {
        std::optional<std::size_t> first_profile;
        for (std::size_t i = 0; i < family.profiles.size(); ++i) {
            if (family.profiles[i].topology_class == topology.topology_class) {
                if (!first_profile) {
                    first_profile = i;
                    install_and_upload(topology, i);

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

DecodeGraphProfile& DecodeGraphFamily::select(std::uint32_t batch_size, std::uint32_t frontier) {
    const auto it =
        std::find_if(profiles.begin(), profiles.end(), [&](const DecodeGraphProfile& p) {
            return p.batch_size == batch_size && p.min_execution_frontier <= frontier &&
                   frontier <= p.max_execution_frontier;
        });
    if (it == profiles.end()) {
        throw std::logic_error("CUDA Graph profile coverage is incomplete");
    }
    return *it;
}

DecodeGraphTopology& DecodeGraphFamily::install(DecodeGraphProfile& profile) {
    const auto it =
        std::find_if(topologies.begin(), topologies.end(), [&](const DecodeGraphTopology& t) {
            return t.topology_class == profile.topology_class;
        });
    if (it == topologies.end()) { throw std::logic_error("CUDA Graph topology is unavailable"); }
    const auto index = static_cast<std::size_t>(&profile - profiles.data());
    if (it->installed_profile != index) {
        if (profile.text)
            it->text->update(*profile.text);
        else
            it->executable.update(profile.definition);
        it->installed_profile = index;
    }
    if (it->text) it->text->activate(*profile.text);
    return *it;
}

void ProgramImpl::prepare_graphs() {
    if (!use_cuda_graph) { return; }
    std::optional<moe::Schedule::Preparation> preparation;
    if (moe) preparation.emplace(*moe);
    nvtx::ScopedRange prepare_range(nvtx::Name::CudaGraphPrepare, nvtx::Category::Graph);

    std::array<StateImageHandle, kMaximumConcurrency> capture_states{};
    for (std::uint32_t row = 0; row < max_concurrency; ++row) {
        std::optional<StateImageHandle> state = state_store->reserve_destination();
        if (!state) { throw std::bad_alloc(); }
        state_images->storage().clear(state_store->physical_slot(*state), device.execution_view());
        state_store->complete_initialization(*state);
        capture_states[row] = *state;
    }
    const auto capture_state_slot = [&](std::uint32_t row) {
        return state_store->physical_slot(capture_states.at(row));
    };

    std::vector<context::KVHistory> text_capture_allocations;
    std::vector<context::KVHistory> mtp_capture_allocations;
    std::vector<context::KVHistory> dflash_capture_allocations;
    const auto reserve_capture_rows = [&](qwen3_5::PagedKVCache& cache,
                                          context::KVHistoryStore& histories,
                                          std::vector<context::KVHistory>& allocations,
                                          const char* label) {
        DeviceKVPagePool& pool       = cache.page_pool();
        KVExecutionTablePool& tables = cache.execution_tables();
        if (pool.capacity_pages() < max_concurrency) {
            throw std::invalid_argument(std::string(label) +
                                        " cannot provide one Paged KV page per concurrent request");
        }
        allocations.reserve(max_concurrency);
        for (std::uint32_t row = 0; row < max_concurrency; ++row) {
            auto binding = histories.prepare_binding(nullptr, 0, 1, row);
            binding.publish_restores();
            binding.prepare_tail_copy();
            binding.enqueue_mapping(device.stream());
            allocations.push_back(binding.commit());
            auto& allocation = allocations.back();
            histories.prepare_write(allocation, 0, 1, device.stream());

            // Capture profiles exercise arbitrary context envelopes. Repeating each row's private
            // page across its temporary table keeps every dummy read/write address valid without
            // reserving C full contexts solely for graph construction.
            tables.publish_repeated(histories.execution_row(allocation.handle()).handle(),
                                    histories.physical_page(allocation.handle(), 0),
                                    tables.logical_page_capacity(), device.stream());
        }
    };
    reserve_capture_rows(decoder->text_kv, *text_kv_histories, text_capture_allocations,
                         "target KV cache");
    if (speculative_backend == SpeculativeBackend::Mtp) {
        reserve_capture_rows(*decoder->mtp_cache(), *backend_kv_histories, mtp_capture_allocations,
                             "MTP KV cache");
    } else if (dflash && dflash->full) {
        reserve_capture_rows(*dflash->full, *backend_kv_histories, dflash_capture_allocations,
                             "DFlash Full KV cache");
    }
    device.synchronize();

    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls{
            io.token,
            io.pos,
            io.rope_pos,
            io.rope_delta,
        };
        if (io.mtp) {
            controls.push_back(io.mtp->position);
            controls.push_back(io.mtp->draft_tokens);
            controls.push_back(io.mtp->target_input_ids);
            controls.push_back(io.mtp->target_positions);
        }
        if (io.dflash_prefill) { controls.push_back(io.dflash_prefill->local_append_count); }
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream()));
        }
    };
    const auto zero_capture_pages =
        [&](qwen3_5::PagedKVCache& cache, const context::KVHistoryStore& histories,
            const std::vector<context::KVHistory>& allocations, std::uint32_t batch_size) {
            std::vector<DeviceKVPageHandle> pages;
            pages.reserve(batch_size);
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                pages.push_back(histories.physical_page(allocations[row].handle(), 0));
            }
            cache.page_pool().zero_pages(pages, device.stream());
        };
    const auto prepare_representative = [&](std::uint32_t frontier, std::uint32_t batch_size) {
        if (batch_size == 0 || batch_size > max_concurrency) {
            throw std::logic_error("CUDA Graph representative batch is invalid");
        }
        work.reset();
        clear_stable_controls();
        zero_capture_pages(decoder->text_kv, *text_kv_histories, text_capture_allocations,
                           batch_size);
        if (decoder->mtp_cache() != nullptr) {
            zero_capture_pages(*decoder->mtp_cache(), *backend_kv_histories,
                               mtp_capture_allocations, batch_size);
        }
        if (dflash && dflash->full) {
            zero_capture_pages(*dflash->full, *backend_kv_histories, dflash_capture_allocations,
                               batch_size);
        }
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            state_images->storage().clear(capture_state_slot(row), device.execution_view());
            if (dflash) {
                const Tensor pending =
                    dflash->pending_features.slice(2, static_cast<std::int32_t>(row), 1);
                CUDA_CHECK(cudaMemsetAsync(pending.data, 0, pending.bytes(), device.stream()));
            }
        }
        set_device_i32(io.pos, context::checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos,
                       context::checked_i32(frontier, "graph representative rope position"));
        if (io.mtp) {
            set_device_i32(io.mtp->position,
                           context::checked_i32(frontier, "graph representative MTP position"));
        }
        if (io.dflash_decode) {
            *dflash_host_ingress       = {};
            *dflash_host_egress        = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            const std::uint32_t width  = draft_window + 1U;
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                dflash_host_ingress->anchors[row] = 0;
                dflash_host_ingress->execution_frontiers[row] =
                    context::checked_i32(frontier, "graph representative DFlash frontier");
                dflash_host_ingress->context_frontiers[row] =
                    context::checked_i32(frontier, "graph representative DFlash context frontier");
                dflash_host_ingress->proposal_valid_columns[row] = static_cast<std::int32_t>(width);
                dflash_host_ingress->proposal_extents[row] = static_cast<std::int32_t>(extent);
                dflash_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t column = 0; column < width; ++column) {
                    dflash_host_ingress->target_rope_positions[row * width + column] =
                        context::checked_i32(frontier + std::min(column, extent),
                                             "graph representative DFlash target RoPE position");
                }
                dflash_host_ingress->text_kv_table_rows[row]      = static_cast<std::int32_t>(row);
                dflash_host_ingress->dflash_kv_table_rows[row]    = static_cast<std::int32_t>(row);
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
                    context::checked_i32(frontier, "graph representative MTP frontier");
                mtp_host_ingress->remaining_budgets[row] =
                    context::checked_i32(capacity, "graph representative MTP budget");
                mtp_host_ingress->current_extents[row] = static_cast<std::int32_t>(extent);
                mtp_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t step = 0; step < draft_window; ++step) {
                    mtp_host_ingress->current_drafts[row * draft_window + step] = 0;
                }
                for (std::uint32_t column = 0; column < width; ++column) {
                    mtp_host_ingress->target_rope_positions[row * width + column] =
                        context::checked_i32(frontier + std::min(column, extent),
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
                    context::checked_i32(frontier, "graph representative ordinary position");
                ordinary_host_ingress->rope_positions[row] =
                    context::checked_i32(frontier, "graph representative ordinary RoPE position");
                ordinary_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                ordinary_host_ingress->state_source_slots[row] = capture_state_slot(row);
                ordinary_host_ingress->state_destination_slots[row] = capture_state_slot(row);
                ordinary_host_ingress->sampling[row]                = {};
            }
        }
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
                                        moe.get()};
    };

    const auto text_definition = [&](const DecodeGraphFamily& family, std::uint32_t batch) {
        const execution::TextGraphDefinition* shared = nullptr;
        for (const auto& profile : family.profiles) {
            if (profile.batch_size == batch && profile.text) {
                shared = profile.text.get();
                break;
            }
        }
        return std::make_unique<execution::TextGraphDefinition>(parameters.text.layers.size(),
                                                                shared);
    };

    if (speculative_backend == SpeculativeBackend::None) {
        const auto ordinary_profiles = ordinary_graph_profiles(capacity);
        validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
        const std::uint32_t ordinary_batch_limit = max_concurrency;
        execution::OrdinaryBatchContext ordinary_state{
            execution_core(),      decoder->text_kv,
            *io.ordinary,          *ordinary_host_ingress,
            *ordinary_host_egress, state_images->continuation_hidden_store()};
        const GraphExecutionProfile code_warm = ordinary_profiles.front();
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        execution::ordinary_decode_batch(ordinary_state, 1, {code_warm.min + 1, code_warm.max + 1},
                                         nullptr);
        device.synchronize();

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
                std::optional<execution::TextGraphRun> recording;
                if (moe) {
                    profile.text = text_definition(ordinary_graphs, batch_size);
                    recording.emplace(*profile.text, device.execution_view());
                    ordinary_state.execution.text_graph = &*recording;
                }
                execution::capture_ordinary_decode_batch(ordinary_state,
                                                         static_cast<std::int32_t>(batch_size),
                                                         envelope, profile.definition);
                ordinary_state.execution.text_graph = nullptr;
            }
        }
    }

    if (!ordinary_graphs.profiles.empty()) {
        instantiate_graph_family(ordinary_graphs, "ordinary", device);
        for (auto& topology : ordinary_graphs.topologies) {
            const auto& profile = ordinary_graphs.profiles[*topology.installed_profile];
            prepare_representative(profile.min_execution_frontier, profile.batch_size);
            device.synchronize();
            if (topology.text) {
                execution::TextGraphRun replay(*topology.text, device.execution_view());
                auto core       = execution_core();
                core.text_graph = &replay;
                execution::OrdinaryBatchContext state{core,
                                                      decoder->text_kv,
                                                      *io.ordinary,
                                                      *ordinary_host_ingress,
                                                      *ordinary_host_egress,
                                                      state_images->continuation_hidden_store()};
                execution::ordinary_decode_batch(
                    state, profile.batch_size,
                    {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1},
                    nullptr);
            } else {
                topology.executable.launch(device.stream());
            }
            device.synchronize();
        }
    }
    if (speculative_backend != SpeculativeBackend::None) {
        using execution::SpeculativePhase;
        const bool mtp = speculative_backend == SpeculativeBackend::Mtp;
        const auto forward_profiles =
            mtp ? mtp_graph_profiles(capacity, draft_window)
                : dflash_graph_profiles(speculative_backend, capacity, draft_window);
        // DFlash acceptance has no context-dependent nodes or resource envelope.
        const auto finish_profiles =
            mtp ? forward_profiles : std::vector<GraphExecutionProfile>{{0, capacity - 1}};
        validate_graph_profiles(forward_profiles, capacity - 1, "speculative forward");
        validate_graph_profiles(finish_profiles, capacity - 1, "speculative finish");
        const auto mtp_context = [&] {
            return execution::MtpBatchContext{execution_core(),
                                              decoder->text_kv,
                                              *decoder->mtp_cache(),
                                              *io.mtp_decode,
                                              *mtp_host_ingress,
                                              *mtp_host_egress,
                                              state_images->continuation_hidden_store()};
        };
        const auto dflash_context = [&] {
            return execution::DFlashBatchContext{execution_core(),
                                                 decoder->text_kv,
                                                 *dflash,
                                                 *io.dflash_decode,
                                                 *dflash_host_ingress,
                                                 *dflash_host_egress,
                                                 state_images->continuation_hidden_store(),
                                                 dflash_draft_handoff->tokens(),
                                                 dflash_draft_handoff->ready};
        };
        const auto target_envelope = [&](std::uint32_t frontier) {
            return ops::CausalAttentionExecutionEnvelope{
                1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                       capacity, static_cast<std::uint64_t>(frontier) + draft_window + 1ULL))};
        };
        const auto run = [&](SpeculativePhase phase, std::int32_t batch, std::uint32_t frontier,
                             DecodeGraphExecutable* executable,
                             execution::TextGraphExecutable* text = nullptr) {
            std::optional<execution::TextGraphRun> replay;
            if (text) replay.emplace(*text, device.execution_view());
            if (mtp) {
                auto state = mtp_context();
                if (replay) state.execution.text_graph = &*replay;
                execution::mtp_decode_batch(
                    state, batch, draft_window,
                    mtp_causal_attention_envelopes(frontier, draft_window, capacity), executable,
                    phase);
            } else {
                auto state = dflash_context();
                if (replay) state.execution.text_graph = &*replay;
                execution::dflash_decode_batch(state, batch, draft_window,
                                               dflash_envelopes(frontier, draft_window),
                                               target_envelope(frontier), executable, phase);
            }
        };
        const auto capture = [&](SpeculativePhase phase, DecodeGraphProfile& profile) {
            const auto batch    = static_cast<std::int32_t>(profile.batch_size);
            const auto frontier = profile.max_execution_frontier;
            std::optional<execution::TextGraphRun> recording;
            if (moe && phase == SpeculativePhase::Forward) {
                profile.text = text_definition(speculative_forward_graphs, profile.batch_size);
                recording.emplace(*profile.text, device.execution_view());
            }
            if (mtp) {
                auto state = mtp_context();
                if (recording) state.execution.text_graph = &*recording;
                execution::capture_mtp_decode_batch(
                    state, batch, draft_window,
                    mtp_causal_attention_envelopes(frontier, draft_window, capacity),
                    profile.definition, phase);
            } else {
                auto state = dflash_context();
                if (recording) state.execution.text_graph = &*recording;
                execution::capture_dflash_decode_batch(
                    state, batch, draft_window, dflash_envelopes(frontier, draft_window),
                    target_envelope(frontier), profile.definition, phase);
            }
        };

        const auto code_warm = forward_profiles.front();
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        run(SpeculativePhase::Forward, 1, code_warm.max, nullptr);
        run(SpeculativePhase::Finish, 1, code_warm.max, nullptr);
        device.synchronize();

        for (const auto phase : {SpeculativePhase::Forward, SpeculativePhase::Finish}) {
            const bool forward = phase == SpeculativePhase::Forward;
            auto& family       = forward ? speculative_forward_graphs : speculative_finish_graphs;
            const auto& planned_profiles = forward ? forward_profiles : finish_profiles;
            family.profiles.reserve(planned_profiles.size() * max_concurrency);
            for (std::uint32_t batch = 1; batch <= max_concurrency; ++batch) {
                for (const auto planned : planned_profiles) {
                    family.profiles.emplace_back();
                    auto& profile                  = family.profiles.back();
                    profile.batch_size             = batch;
                    profile.min_execution_frontier = planned.min;
                    profile.max_execution_frontier = planned.max;
                    profile.topology_class = planned.topology_class * max_concurrency + batch - 1U;
                    capture(phase, profile);
                }
            }
            instantiate_graph_family(family, forward ? "speculative forward" : "speculative finish",
                                     device);
        }

        // Warm complete transactions: finish consumes persistent outputs of its paired forward.
        for (auto& topology : speculative_forward_graphs.topologies) {
            auto& forward = speculative_forward_graphs.profiles[*topology.installed_profile];
            speculative_forward_graphs.install(forward);
            auto& finish      = speculative_finish_graphs.select(forward.batch_size,
                                                                 forward.min_execution_frontier);
            auto& finish_exec = speculative_finish_graphs.install(finish);
            prepare_representative(forward.min_execution_frontier, forward.batch_size);
            device.synchronize();
            if (topology.text)
                run(SpeculativePhase::Forward, forward.batch_size, forward.max_execution_frontier,
                    nullptr, topology.text.get());
            else
                topology.executable.launch(device.stream());
            finish_exec.executable.launch(device.stream());
            device.synchronize();
        }
    }

    if (moe) moe->restore_cache_after_preparation();
    clear_stable_controls();
    state_images->storage().clear_all(device.execution_view());
    if (dflash) {
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_features.data, 0,
                                   dflash->prefill_features.bytes(), device.stream()));
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_positions.data, 0,
                                   dflash->prefill_positions.bytes(), device.stream()));
        CUDA_CHECK(cudaMemsetAsync(dflash->pending_features.data, 0,
                                   dflash->pending_features.bytes(), device.stream()));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream()));
    device.synchronize();
    for (std::uint32_t row = 0; row < max_concurrency; ++row) {
        if (!state_store->release(capture_states[row])) {
            throw std::logic_error("CUDA Graph capture StateImage could not be released");
        }
    }

    dflash_capture_allocations.clear();
    mtp_capture_allocations.clear();
    text_capture_allocations.clear();
}

} // namespace ninfer::models::qwen3_5::detail
