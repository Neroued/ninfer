#include "models/qwen3_5/program/program_impl.h"
#include "models/common/context/work.h"
#include "models/qwen3_5/program/execution_context.h"
#include "core/buffer.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/startup.h"
#include "core/device.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

static_assert(std::is_nothrow_move_assignable_v<SpeculativeStats>);

ProgramImpl::ProgramImpl(const execution::Parameters& parameters_in, const SequencePlanImpl& plan,
                         DeviceContext& device_in, const StartupObserver& startup_observer)
    : parameters(parameters_in), device(device_in), capacity(plan.capacity),
      kv_capacity(plan.kv_capacity), max_concurrency(plan.max_concurrency),
      context_cache(plan.context_cache), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), speculative_backend(plan.speculative_backend),
      kv_storage(plan.kv_storage), proposal_head(plan.proposal_head),
      vision_enabled(plan.features.vision), use_cuda_graph(plan.use_cuda_graph),
      causal_scoring(plan.causal_scoring), decision_scoring(plan.decision_scoring),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      workspace_plan(plan.workspace), persistent(device.device(), plan.persistent.bytes),
      workspace_storage(device.device(), plan.workspace.capacity),
      work(workspace_storage.span().subspan(0, plan.workspace.general_capacity)),
      round_host(plan.causal_scoring
                     ? std::nullopt
                     : std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::PrefillRoundHost))),
      score_logprobs_host(plan.causal_scoring ? std::make_optional<PinnedHostBuffer>(
                                                    kCausalScoreTile * sizeof(float))
                                              : std::nullopt),
      ordinary_host(
          !plan.causal_scoring && plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_5::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::MtpDecodeIngress) +
                                                          sizeof(qwen3_5::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(is_masked_draft_backend(plan.speculative_backend)
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::DFlashDecodeIngress) +
                                                             sizeof(qwen3_5::DFlashDecodeEgress) +
                                                             sizeof(qwen3_5::DFlashPrefillIngress))
                      : std::nullopt),
      prefill_gpu_timer_(device_in) {
    if (parameters.profile != plan.profile || parameters.model.options() != plan.features) {
        throw std::invalid_argument("Program parameters do not match the frozen sequence plan");
    }
    if (workspace_plan.general_capacity == 0 ||
        workspace_plan.vision.has_value() != vision_enabled ||
        causal_scoring != plan.persistent.score_hidden.has_value() ||
        causal_scoring != (workspace_plan.causal_score != 0) ||
        (!decision_scoring && workspace_plan.branches.has_value()) ||
        (workspace_plan.vision &&
         workspace_plan.vision->general_capacity_bytes != workspace_plan.general_capacity)) {
        throw std::invalid_argument("Qwen3.5 workspace plan does not match startup features");
    }
    const MemorySpan backing = persistent.span();
    decoder      = std::make_unique<qwen3_5::DecoderState>(backing, plan.persistent.decoder);
    state_images = std::make_unique<StateImageViews>(backing, plan.persistent.state_images);
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear().all_layers_view());
    }
    if (plan.persistent.dflash) {
        auto* local = state_images->dflash_local();
        if (!local) { throw std::logic_error("DFlash StateImage has no local state"); }
        dflash.emplace(backing, *plan.persistent.dflash, *local);
    }
    const auto host_bytes = plan.context_cache.host_capacity_bytes.value();
    std::vector<HostKVPageLayout> layouts{
        plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry())};
    text_host_kv_page_stride = layouts.front().page_stride;
    if (const auto* backend = backend_kv_cache()) {
        auto layout                 = plan_host_kv_page_layout(backend->page_pool().geometry());
        backend_host_kv_page_stride = layout.page_stride;
        if (layout != layouts.front()) { layouts.push_back(std::move(layout)); }
    }
    std::size_t minimum_stride    = state_images->storage().image_bytes();
    std::size_t minimum_kv_stride = layouts.front().page_stride;
    for (const auto& layout : layouts) {
        minimum_stride    = std::min(minimum_stride, layout.page_stride);
        minimum_kv_stride = std::min(minimum_kv_stride, layout.page_stride);
    }
    const auto checked_count = [](std::uint64_t value) {
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("context resource descriptor capacity exceeds uint32");
        }
        return static_cast<std::uint32_t>(value);
    };
    const auto logical_states = checked_count(state_images->storage().slot_count() +
                                              host_bytes / state_images->storage().image_bytes());
    // One immutable image can serve private replay, an explicit anchor, a public view and an
    // exact-hit endpoint. Alias descriptors are bounded separately from physical state slots.
    checkpoints.resize(checked_count(4ULL * logical_states + 2ULL * max_concurrency));
    const auto history_capacity = checked_count(checkpoints.size() + max_concurrency + 2U);
    if (host_bytes) {
        StartupPhaseScope phase(startup_observer, StartupPhase::HostContextPin,
                                StartupProgressUnit::Bytes, host_bytes);
        host_context_arena = std::make_unique<HostContextArena>(host_bytes, minimum_stride);
        host_kv_arena      = std::make_unique<HostKVArena>(*host_context_arena, layouts);
        const auto extents = checked_count(host_bytes / minimum_kv_stride);
        if (extents) {
            host_kv_extents = std::make_unique<context::HostKVExtentStore>(*host_kv_arena, extents);
        }
        phase.complete(host_bytes, host_bytes);
    }
    state_store              = std::make_unique<StateImageStore>(state_images->storage(),
                                                                 host_context_arena.get(), logical_states);
    const auto logical_pages = [&](const DeviceKVPagePool& pool) {
        const auto stride = plan_host_kv_page_layout(pool.geometry()).page_stride;
        return checked_count(pool.capacity_pages() + host_bytes / stride);
    };
    text_kv_pages = std::make_unique<context::LogicalKVPageStore>(
        decoder->text_kv.page_pool(), logical_pages(decoder->text_kv.page_pool()));
    text_kv_histories = std::make_unique<context::KVHistoryStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), history_capacity,
        decoder->text_kv.execution_tables().logical_page_capacity(), host_kv_extents.get());
    if (auto* backend = backend_kv_cache()) {
        backend_kv_pages = std::make_unique<context::LogicalKVPageStore>(
            backend->page_pool(), logical_pages(backend->page_pool()));
        backend_kv_histories = std::make_unique<context::KVHistoryStore>(
            *backend_kv_pages, backend->execution_tables(), history_capacity,
            backend->execution_tables().logical_page_capacity(), host_kv_extents.get());
    }

    const auto register_pool = [&](runtime::KvPoolId id, std::string_view label,
                                   const DeviceKVPagePool& pool, std::size_t host_stride) {
        kv_pools.push_back({id, label, &pool.geometry(), pool.capacity_pages(),
                            pool.device_copy_work(1).payload_bytes, host_stride});
    };
    register_pool(kTextKvPool, "text", text_kv_pages->physical_pool(), text_host_kv_page_stride);
    if (backend_kv_pages) {
        register_pool(kBackendKvPool,
                      speculative_backend == SpeculativeBackend::Mtp ? "mtp" : "draft_full",
                      backend_kv_pages->physical_pool(), backend_host_kv_page_stride);
    }

    migration_context_.emplace(device, kv_pools.size());

    io = qwen3_5::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() !=
        (!causal_scoring && speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    prefill_hidden = plan.persistent.prefill_hidden.bind(backing);
    if (plan.persistent.score_hidden) {
        score_hidden = plan.persistent.score_hidden->bind(backing);
    }
    if (plan.persistent.token_counts) {
        token_counts = plan.persistent.token_counts->bind(backing);
    }
    if (plan.persistent.sampling_config) {
        sampling_config = plan.persistent.sampling_config->bind(backing);
    }
    if (plan.persistent.grammar_masks) {
        grammar_masks_device = plan.persistent.grammar_masks->bind(backing);
        grammar_masks_host.emplace(grammar_masks_device.bytes());
    }
    if (is_masked_draft_backend(speculative_backend)) {
        dflash_draft_handoff.emplace(device, draft_window * max_concurrency);
    }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        lane_epochs[lane]    = 1;
        sequences[lane].lane = lane;
    }
    host_tokens = round_host
                      ? &static_cast<qwen3_5::PrefillRoundHost*>(round_host->data())->sampled_token
                      : nullptr;
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_5::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_5::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_5::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_5::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_5::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_5::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_5::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_5::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress));
        *dflash_host_ingress        = {};
        *dflash_host_egress         = {};
        dflash_prefill_host_ingress = reinterpret_cast<qwen3_5::DFlashPrefillIngress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress) + sizeof(qwen3_5::DFlashDecodeEgress));
        *dflash_prefill_host_ingress = {};
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream()));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream()));
    }
    if (!causal_scoring) {
        CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream()));
        CUDA_CHECK(
            cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream()));
    }
    if (plan.moe) {
        std::vector<const moe::Parameters*> layers;
        for (const auto& block : parameters.text.layers) {
            if (const auto* layer = std::get_if<moe::Parameters>(&block.ffn))
                layers.push_back(layer);
        }
        moe = std::make_unique<moe::Resources>(
            plan.moe, layers, parameters.model.expert_banks(), device,
            plan.graph_probe ? moe::ResourcePurpose::GraphCalibration
                             : moe::ResourcePurpose::Execution);
    }
    device.synchronize();

    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

void ProgramImpl::initialize_graphs() {
    if (!use_cuda_graph) return;
    device.synchronize();
    const auto before = device_memory_snapshot(device.device());
    prepare_graphs();
    device.synchronize();
    graph_memory = GraphMemorySummary{.preparation_increment_bytes = device_memory_growth(
                                          before, device_memory_snapshot(device.device()))};
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

void ProgramImpl::release_graphs() {
    if (moe) moe->synchronize();
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
    device.synchronize();
    ordinary_graphs            = {};
    speculative_forward_graphs = {};
    speculative_finish_graphs  = {};
}

ProgramImpl::~ProgramImpl() noexcept {
    if (moe) {
        try {
            moe->synchronize();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "MoE teardown synchronization: %s\n", error.what());
        }
    }
    if (device.transfer_stream() != nullptr) (void)cudaStreamSynchronize(device.transfer_stream());
    if (device.stream() != nullptr) (void)cudaStreamSynchronize(device.stream());
    ordinary_graphs            = {};
    speculative_forward_graphs = {};
    speculative_finish_graphs  = {};
    moe.reset();
}

std::vector<float> ProgramImpl::causal_score(qwen::PreparedPromptData&& prompt,
                                             std::uint32_t first_target) {
    if (!causal_scoring || !score_hidden || !score_logprobs_host ||
        workspace_plan.causal_score == 0) {
        throw std::logic_error("Program was not constructed for causal scoring");
    }
    if (speculative_backend != SpeculativeBackend::None || vision_enabled || use_cuda_graph ||
        context_cache.enabled) {
        throw std::logic_error("causal scoring Program has generation-only startup features");
    }
    const std::size_t token_count_size = prompt.token_ids.size();
    if (token_count_size < 2 || token_count_size > capacity) {
        throw std::invalid_argument("causal score token count must be in [2,capacity]");
    }
    if (first_target == 0 || first_target >= token_count_size) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    if (prompt.has_media()) {
        throw std::invalid_argument("causal scoring accepts text tokens only");
    }

    const auto token_count                     = static_cast<std::uint32_t>(token_count_size);
    const std::uint32_t predictor_count        = token_count - 1U;
    const std::uint32_t scored_predictor_begin = first_target - 1U;
    const std::uint32_t required_pages         = context::kv_pages_for_frontier(predictor_count);
    if (required_pages == 0) { throw std::logic_error("causal score requires KV pages"); }

    std::optional<StateImageHandle> state;
    std::optional<context::KVHistory> address;
    const auto cleanup = [&] {
        bool released = true;
        address.reset();
        if (state) {
            released = state_store->release(*state) && released;
            state.reset();
        }
        if (!released) { throw std::logic_error("causal score resources could not be released"); }
    };

    std::vector<float> output;
    output.reserve(token_count_size - first_target);
    std::vector<TokenId> staged_targets;
    staged_targets.reserve(kCausalScoreTile);
    std::uint32_t staged_columns = 0;

    try {
        state = state_store->reserve_destination();
        if (!state) { throw std::bad_alloc(); }
        state_images->storage().clear(state_store->physical_slot(*state), device.execution_view());
        state_store->complete_initialization(*state);
        auto binding = text_kv_histories->prepare_binding(nullptr, 0, predictor_count, 0);
        binding.publish_restores();
        binding.prepare_tail_copy();
        binding.enqueue_mapping(device.stream());
        address.emplace(binding.commit());
        text_kv_histories->prepare_write(*address, 0, predictor_count, device.stream());

        const std::int32_t state_slot = state_store->physical_slot(*state);

        const auto flush = [&] {
            if (staged_columns == 0) { return; }
            if (staged_columns != staged_targets.size() || staged_columns > kCausalScoreTile) {
                throw std::logic_error("causal score staging has an invalid shape");
            }
            work.reset();
            mark_workspace_usage(workspace_plan.causal_score);
            const auto columns = static_cast<std::int32_t>(staged_columns);
            Tensor logits      = work.alloc(
                DType::BF16, {dimension(parameters.model.config().text.vocab_size), columns});
            Tensor target_ids        = work.alloc(DType::I32, {columns});
            Tensor logprobs          = work.alloc(DType::FP32, {columns});
            Tensor hidden            = score_hidden->slice(1, 0, columns);
            const auto public_tokens = dimension(parameters.model.resources().public_token_count);
            execution::project(hidden, parameters.text.output_head, logits, work,
                               device.execution_view());
            CUDA_CHECK(cudaMemcpyAsync(target_ids.data, staged_targets.data(), target_ids.bytes(),
                                       cudaMemcpyHostToDevice, device.stream()));
            ops::target_logprobs(logits, target_ids, public_tokens, logprobs,
                                 device.execution_view());
            CUDA_CHECK(cudaMemcpyAsync(score_logprobs_host->data(), logprobs.data, logprobs.bytes(),
                                       cudaMemcpyDeviceToHost, device.stream()));
            device.synchronize();
            const auto* host = static_cast<const float*>(score_logprobs_host->data());
            output.insert(output.end(), host, host + staged_columns);
            staged_targets.clear();
            staged_columns = 0;
            work.reset();
        };

        std::uint32_t cursor = 0;
        while (cursor < predictor_count) {
            const std::uint32_t nominal = std::min(prefill_chunk, predictor_count - cursor);
            execution::PrefillContext schedule_state{
                {device, parameters, work, state_images->linear(), nullptr, io, prefill_hidden,
                 prefill_chunk, proposal_head, moe.get()},
                {},
                decoder->text_kv,
                nullptr,
                nullptr,
                cursor,
                nullptr,
                nullptr,
                state_slot,
                state_slot,
                0,
                0,
                nullptr};
            mark_workspace_usage(workspace_plan.text_prefill);
            const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                schedule_state, std::span<const TokenId>(prompt.token_ids), nominal, std::nullopt,
                false);
            if (result.finalized || result.processed_tokens == 0 ||
                result.processed_tokens > nominal) {
                throw std::logic_error("causal score Prefill made invalid progress");
            }
            const std::uint32_t chunk_begin = cursor;
            cursor += result.processed_tokens;
            text_kv_histories->commit_frontier(address->handle(), cursor);

            std::uint32_t selected = std::max(chunk_begin, scored_predictor_begin);
            while (selected < cursor) {
                const std::uint32_t available = cursor - selected;
                const std::uint32_t room      = kCausalScoreTile - staged_columns;
                const std::uint32_t count     = std::min(available, room);
                Tensor source =
                    prefill_hidden.slice(1, static_cast<std::int32_t>(selected - chunk_begin),
                                         static_cast<std::int32_t>(count));
                Tensor destination = score_hidden->slice(
                    1, static_cast<std::int32_t>(staged_columns), static_cast<std::int32_t>(count));
                CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                           cudaMemcpyDeviceToDevice, device.stream()));
                for (std::uint32_t column = 0; column < count; ++column) {
                    staged_targets.push_back(prompt.token_ids[selected + column + 1U]);
                }
                selected += count;
                staged_columns += count;
                if (staged_columns == kCausalScoreTile) { flush(); }
            }
        }
        flush();
        if (output.size() != token_count_size - first_target) {
            throw std::logic_error("causal score produced the wrong number of logprobs");
        }
        cleanup();
        return output;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        try {
            cleanup();
        } catch (...) {}
        throw;
    }
}

MemorySummary ProgramImpl::memory_summary() const {
    const auto& weights = parameters.model.storage_stats();
    MemorySummary out;
    out.device        = device.device().index();
    out.max_context   = capacity;
    out.prefill_chunk = prefill_chunk;
    if (moe) {
        const auto& plan = moe->plan();
        MoeMemorySummary summary;
        summary.host_weights_bytes    = weights.host_capacity_bytes;
        summary.registered_host_bytes = weights.registered_host_bytes;
        summary.cache_bytes           = plan.cache.config.capacity_bytes;
        summary.fixed_bytes           = plan.cache.config.fixed_bytes;
        summary.window_bytes          = summary.cache_bytes - summary.fixed_bytes;
        summary.window_count          = plan.window_count;
        const auto usage              = moe->cache_usage();
        summary.used_bytes            = usage.used_bytes;
        summary.target_bytes          = plan.target_bytes;
        summary.ready_bytes           = usage.ready_cache_bytes;
        summary.cpu_workers           = static_cast<std::uint32_t>(plan.team.worker_cpus.size());
        for (const auto& layer : plan.layers) {
            summary.fixed_experts_per_layer.push_back(layer.fixed_experts);
            summary.target_experts_per_layer.push_back(layer.cache_experts);
            summary.ready_experts_per_layer.push_back(
                usage.ready_experts_per_bank.at(layer.spec.bank.index));
        }
        summary.bridge_device_bytes = plan.bridge.device_frame_bytes;
        summary.bridge_pinned_bytes = plan.bridge.pinned_frame_bytes;
        summary.cpu_workspace_bytes = plan.bridge.cpu_workspace_bytes;
        summary.uploaded_bytes      = moe->uploaded_bytes();
        summary.prefetched_bytes    = usage.prefetched_bytes;
        out.moe                     = std::move(summary);
    }
    out.kv_capacity = kv_capacity;
    out.kv_cache    = kv_storage;
    out.weights  = ArenaMemorySummary{weights.device_capacity_bytes, weights.device_capacity_bytes,
                                     weights.device_capacity_bytes};
    out.sequence = ArenaMemorySummary{persistent.size(), persistent.size(), persistent.size()};
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
        if (request.replay && request.replay->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.replay->vision->active_handoff_bytes());
        }
    }
    std::size_t active_workspace_bytes = work.used();
    if (branches) {
        active_workspace_bytes = std::max(active_workspace_bytes, workspace_plan.branches->bytes);
    }
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        active_workspace_bytes =
            std::max(active_workspace_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
    out.workspace = ArenaMemorySummary{workspace_storage.size(), active_workspace_bytes,
                                       std::max(work.peak_used(), workspace_logical_peak_bytes)};
    if (workspace_plan.vision) {
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(capacity, qwen::kMaximumPromptVisionTokens)),
            .max_item_tokens        = workspace_plan.vision->max_merged_tokens,
            .general_capacity_bytes = workspace_plan.vision->general_capacity_bytes,
            .encode_peak_bytes      = workspace_plan.vision->encode_peak_bytes,
            .handoff_offset_bytes   = workspace_plan.vision->handoff_offset_bytes,
            .handoff_capacity_bytes = workspace_plan.vision->handoff_capacity_bytes,
            .handoff_active_bytes   = active_handoff_bytes,
            .handoff_peak_bytes     = vision_handoff_peak_bytes,
        };
    }
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph                   = graph_memory;
    out.kv_payload_bytes             = kv_payload_bytes;
    out.host_state_occupied_slots    = state_store->host_occupied();
    if (host_kv_arena) { out.host_kv_occupied_bytes = host_kv_arena->occupied_bytes(); }
    if (host_context_arena) {
        out.host_context_capacity_bytes = host_context_arena->capacity_bytes();
        out.host_context_occupied_bytes = host_context_arena->occupied_bytes();
        out.host_context_reserved_bytes = host_context_arena->reserved_bytes();
    }
    return out;
}

void ProgramImpl::reset_memory_peaks() noexcept {
    work.reset_peak();
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
        if (request.replay && request.replay->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.replay->vision->active_handoff_bytes());
        }
    }
    vision_handoff_peak_bytes    = active_handoff_bytes;
    workspace_logical_peak_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        workspace_logical_peak_bytes =
            std::max(workspace_logical_peak_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
}

} // namespace ninfer::models::qwen3_5::detail
