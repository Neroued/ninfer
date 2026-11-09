#include "models/qwen3_5/program/program_impl.h"
#include "ninfer/ops/ple.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/startup.h"
#include "core/device.h"
#include "core/device_scope.h"
#include "core/expert_cache.h"
#include "ops/sparse_moe/host/host_moe_executor.h"
#include "core/nvtx.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <cstdlib>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

static_assert(std::is_nothrow_move_assignable_v<SpeculativeStats>);

namespace {

// Guardrail (capy-flash): the Host tiers are pinned at startup and the kernel cannot page them
// out. Pinning more than the machine can spare does not fail here but later, as an OOM kill of
// the server or of something else (Qwen3.8-Flash-Next already pins its routed experts). Refuse
// to start when the tiers would leave less than max(4 GiB, 5 % of RAM) available, and say how
// much fits. NINFER_HOST_TIER_RAM_GUARD=0 disables the check.
std::uint64_t meminfo_bytes(const char* key) {
    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (file == nullptr) { return 0; }
    char line[256];
    std::uint64_t kib            = 0;
    const std::size_t key_length = std::strlen(key);
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        if (std::strncmp(line, key, key_length) == 0 && line[key_length] == ':') {
            kib = std::strtoull(line + key_length + 1, nullptr, 10);
            break;
        }
    }
    std::fclose(file);
    return kib * 1024ULL;
}

void require_host_tier_fits_ram(std::uint64_t state_bytes, std::uint64_t kv_bytes) {
    const std::uint64_t pinned = state_bytes + kv_bytes;
    if (pinned == 0) { return; }
    if (const char* guard = std::getenv("NINFER_HOST_TIER_RAM_GUARD");
        guard != nullptr && std::string_view(guard) == "0") {
        return;
    }
    // #40: free CMA counts in MemAvailable but cannot be page-locked.
    const std::uint64_t free_ram  = meminfo_bytes("MemAvailable");
    const std::uint64_t cma_free  = meminfo_bytes("CmaFree");
    const std::uint64_t available = free_ram > cma_free ? free_ram - cma_free : 0;
    const std::uint64_t total     = meminfo_bytes("MemTotal");
    if (free_ram == 0 || total == 0) { return; }
    const std::uint64_t reserve = std::max<std::uint64_t>(4ULL << 30, total / 20U);
    if (available >= pinned + reserve) { return; }
    const auto gib = [](std::uint64_t bytes) {
        char text[32];
        std::snprintf(text, sizeof(text), "%.1f", static_cast<double>(bytes) / (1ULL << 30));
        return std::string(text);
    };
    const std::uint64_t fits =
        available > reserve + state_bytes ? available - reserve - state_bytes : 0;
    throw std::invalid_argument(
        "the Host context tiers would pin " + gib(pinned) + " GiB of RAM (state " +
        gib(state_bytes) + " GiB, KV " + gib(kv_bytes) + " GiB) but only " + gib(available) +
        " GiB are available outside CMA and " + gib(reserve) +
        " GiB must stay free: lower --host-kv-mib (at most " + std::to_string(fits >> 20) +
        " MiB with these state slots) or --host-state-slots, or set "
        "NINFER_HOST_TIER_RAM_GUARD=0 to skip this check");
}

std::uint32_t normalized_private_capacity(const ContextCacheOptions& options);

std::uint32_t normalized_private_capacity(const ContextCacheOptions& options) {
    if (!options.max_private_continuations || *options.max_private_continuations == 0) {
        throw std::logic_error("Qwen3.5 context cache private capacity is not normalized");
    }
    return *options.max_private_continuations;
}

// Both ranks of a tensor-parallel Program start from zeroed persistent and workspace arenas, so
// the two ranks' KV pages, StateImages and scratch hold identical bytes before their first
// write. A single-device Program's arenas start undefined, as every owner fills what it reads.
ZeroFill tensor_parallel_zero_fill(const SequencePlanImpl& plan) noexcept {
    return plan.peer_persistent ? ZeroFill::Yes : ZeroFill::No;
}

// T64: the persistent arena of one rank as an ElasticArena (null without elastic KV).
std::unique_ptr<ElasticArena> make_elastic_arena(const SequencePlanImpl& plan,
                                                 const PersistentLayout& layout, int device) {
    if (!plan.elastic_kv.enabled) { return nullptr; }
    const std::uint32_t step = (plan.elastic_kv.grow_tokens + kPagedKVPageSize - 1U) /
                               static_cast<std::uint32_t>(kPagedKVPageSize);
    const auto ranges = qwen4_elastic_ranges(layout, step);
    return std::make_unique<ElasticArena>(device, layout.elastic_bytes, ranges);
}

} // namespace

ProgramImpl::ProgramImpl(const execution::Parameters& parameters_in, const SequencePlanImpl& plan,
                         DeviceContext& device_in, const StartupObserver& startup_observer,
                         ExecutionContext* execution_in,
                         const execution::Parameters* peer_parameters_in)
    : parameters(parameters_in), device(device_in), capacity(plan.capacity),
      kv_capacity(plan.kv_capacity), max_concurrency(plan.max_concurrency),
      context_cache(plan.context_cache),
      continuation_capacity(normalized_private_capacity(plan.context_cache)),
      shared_prefix_capacity(plan.context_cache.max_shared_prefixes.value_or(0)),
      prefill_chunk(plan.prefill_chunk), qwen4_prompt_base_chunk(plan.qwen4_base_chunk),
      kv_lease_window_tokens(plan.kv_lease_window_tokens), draft_window(plan.draft_window),
      mtp_min_p(plan.mtp_min_p), mtp_window_rule(plan.mtp_window_rule),
      speculative_backend(plan.speculative_backend), kv_storage(plan.kv_storage),
      proposal_head(plan.proposal_head), vision_enabled(plan.features.vision),
      vision_rank(plan.features.vision_rank), use_cuda_graph(plan.use_cuda_graph),
      causal_scoring(plan.causal_scoring), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      elastic_arena(make_elastic_arena(plan, plan.persistent, device_in.device)),
      persistent(plan.persistent.bytes, tensor_parallel_zero_fill(plan)),
      workspace_storage(plan.workspace.rank_capacity(0, plan.features.vision_rank),
                        tensor_parallel_zero_fill(plan)),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      continuation_states(continuation_capacity), continuation_slots(continuation_capacity),
      shared_prefix_states(shared_prefix_capacity), shared_prefix_slots(shared_prefix_capacity),
      round_host(plan.causal_scoring ? std::nullopt
                                     : std::make_optional<PinnedHostBuffer>(sizeof(TokenId))),
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
      dflash_host(plan.speculative_backend == SpeculativeBackend::DFlash2
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::DFlashDecodeIngress) +
                                                             sizeof(qwen3_5::DFlashDecodeEgress) +
                                                             sizeof(qwen3_5::DFlashPrefillIngress))
                      : std::nullopt),
      context_source_ready_(device_in), context_completion_(device_in),
      context_transfer_timers_{CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream)} {
    if (&parameters != plan.parameters || parameters.model.options() != plan.features) {
        throw std::invalid_argument("Program parameters do not match the frozen sequence plan");
    }
    if (workspace_plan.general_capacity == 0 ||
        workspace_plan.vision.has_value() != vision_enabled ||
        causal_scoring != plan.persistent.score_hidden.has_value() ||
        causal_scoring != (workspace_plan.causal_score != 0) ||
        (workspace_plan.vision &&
         workspace_plan.vision->general_capacity_bytes != workspace_plan.general_capacity)) {
        throw std::invalid_argument("Qwen3.5 workspace plan does not match startup features");
    }
    construct_tensor_parallel(plan, execution_in, peer_parameters_in);
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    if (!plan.context_cache.max_private_continuations || !plan.context_cache.max_shared_prefixes) {
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    }
    const std::uint64_t address_capacity64 =
        static_cast<std::uint64_t>(*plan.context_cache.max_private_continuations) +
        *plan.context_cache.max_shared_prefixes;
    if (address_capacity64 == 0 || address_capacity64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 KV address-space capacity exceeds uint32");
    }
    // One unpublished descriptor is reserved for the single in-flight active-capture snapshot.
    // Published private/shared address spaces remain bounded by P + S; the transaction slot lets a
    // full shared catalog replace one entry without releasing the old checkpoint before the new
    // snapshot has been prepared.
    if (address_capacity64 == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 KV transaction address capacity exceeds uint32");
    }
    const auto address_capacity      = static_cast<std::uint32_t>(address_capacity64 + 1U);
    const auto logical_page_capacity = [&](const DeviceKVPagePool& pool) {
        const HostKVPageLayout host_layout =
            plan_host_kv_page_layout(pool.geometry(), pool.host_replicas());
        const std::uint64_t host_pages =
            plan.context_cache.host_kv_capacity_bytes / host_layout.page_stride;
        const std::uint64_t total = static_cast<std::uint64_t>(pool.capacity_pages()) + host_pages;
        if (total > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.5 logical KV page capacity exceeds uint32");
        }
        return static_cast<std::uint32_t>(total);
    };

    decoder = std::make_unique<qwen3_5::DecoderState>(
        device.device, backing, plan.persistent.decoder,
        elastic_arena ? elastic_arena->span() : DeviceSpan{});
    state_images =
        std::make_unique<qwen3_5::StateImageDevicePool>(backing, plan.persistent.state_images);
    // Mirrors first: a mirrored pool's Host records hold both ranks' halves, so every Host layout
    // and logical capacity below depends on them.
    if (peer) { attach_tensor_parallel_mirrors(); }
    text_host_kv_page_stride =
        plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry(),
                                 decoder->text_kv.page_pool().host_replicas())
            .page_stride;
    text_kv_pages = std::make_unique<LogicalKVPageStore>(
        decoder->text_kv.page_pool(), logical_page_capacity(decoder->text_kv.page_pool()));
    text_kv_addresses = std::make_unique<KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), address_capacity,
        decoder->text_kv.execution_tables().logical_page_capacity());
    require_host_tier_fits_ram(static_cast<std::uint64_t>(state_images->host_layout().image_bytes) *
                                   plan.context_cache.host_state_slots,
                               plan.context_cache.host_kv_capacity_bytes);
    if (plan.context_cache.host_state_slots != 0) {
        const std::uint64_t host_state_bytes =
            static_cast<std::uint64_t>(state_images->host_layout().image_bytes) *
            plan.context_cache.host_state_slots;
        StartupPhaseScope host_state_phase(startup_observer, StartupPhase::HostStatePin,
                                           StartupProgressUnit::Bytes, host_state_bytes);
        host_state_images = std::make_unique<qwen3_5::HostStatePool>(
            state_images->host_layout(), plan.context_cache.host_state_slots);
        host_state_phase.complete(host_state_bytes, host_state_bytes);
    }
    const std::uint64_t logical_state_capacity =
        static_cast<std::uint64_t>(state_images->slot_count()) +
        plan.context_cache.host_state_slots;
    if (logical_state_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 logical StateImage capacity exceeds uint32");
    }
    state_store = std::make_unique<StateImageStore>(
        *state_images, host_state_images.get(), static_cast<std::uint32_t>(logical_state_capacity));
    pressure_private_owner_scratch_.resize(continuation_capacity);
    pressure_shared_owner_scratch_.resize(shared_prefix_capacity);
    pressure_private_drop_scratch_.resize(continuation_capacity);
    const std::size_t pressure_checkpoint_capacity =
        2U + context_cache.max_long_anchors_per_continuation.value_or(0U);
    for (auto& dropped : pressure_private_drop_scratch_) {
        dropped.reserve(pressure_checkpoint_capacity);
    }
    pressure_state_scratch_.reserve(static_cast<std::size_t>(logical_state_capacity));
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear().all_layers_view());
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None) ||
        replay_fold.has_value() != replay_records.has_value()) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (plan.persistent.dflash) {
        CyclicKVCache* local = state_images->dflash_local();
        if (local == nullptr) {
            throw std::logic_error("DFlash StateImage has no local fixed state");
        }
        dflash.emplace(backing, *plan.persistent.dflash, *local);
    }
    if (dflash.has_value() != plan.features.dflash2()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }
    if (qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
        backend_host_kv_page_stride = plan_host_kv_page_layout(backend->page_pool().geometry(),
                                                               backend->page_pool().host_replicas())
                                          .page_stride;
        backend_kv_pages = std::make_unique<LogicalKVPageStore>(
            backend->page_pool(), logical_page_capacity(backend->page_pool()));
        backend_kv_addresses = std::make_unique<KVAddressSpaceStore>(
            *backend_kv_pages, backend->execution_tables(), address_capacity,
            backend->execution_tables().logical_page_capacity());
    }
    pressure_text_page_scratch_.resize(text_kv_pages->capacity());
    pressure_text_selected_pages_.reserve(text_kv_pages->capacity());
    if (backend_kv_pages) {
        pressure_backend_page_scratch_.resize(backend_kv_pages->capacity());
        pressure_backend_selected_pages_.reserve(backend_kv_pages->capacity());
    }
    if (plan.context_cache.host_kv_capacity_bytes != 0) {
        std::vector<HostKVPageLayout> layouts;
        layouts.push_back(plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry(),
                                                   decoder->text_kv.page_pool().host_replicas()));
        if (const qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
            HostKVPageLayout backend_layout = plan_host_kv_page_layout(
                backend->page_pool().geometry(), backend->page_pool().host_replicas());
            if (backend_layout != layouts.front()) { layouts.push_back(std::move(backend_layout)); }
        }
        StartupPhaseScope host_kv_phase(
            startup_observer, StartupPhase::HostKvPin, StartupProgressUnit::Bytes,
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes));
        host_kv_arena = std::make_unique<HostKVArena>(
            plan.context_cache.host_kv_capacity_bytes,
            std::span<const HostKVPageLayout>(layouts.data(), layouts.size()));
        host_kv_phase.complete(
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes),
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes));
        std::size_t minimum_stride = layouts.front().page_stride;
        for (const HostKVPageLayout& layout : layouts) {
            minimum_stride = std::min(minimum_stride, layout.page_stride);
        }
        const std::size_t extent_capacity =
            plan.context_cache.host_kv_capacity_bytes / minimum_stride;
        if (extent_capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.5 Host KV extent capacity exceeds uint32");
        }
        if (extent_capacity != 0) {
            host_kv_extents = std::make_unique<HostKVExtentStore>(
                *host_kv_arena, static_cast<std::uint32_t>(extent_capacity));
        }
    }

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
    if (io.dflash_prefill.has_value() != (speculative_backend == SpeculativeBackend::DFlash2)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != (speculative_backend == SpeculativeBackend::DFlash2)) {
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
        // Structured output: the pinned twin of the lane slots, then one staged prefill config
        // per lane (program/grammar_masks.cpp).
        grammar_masks_device = plan.persistent.grammar_masks->bind(backing);
        grammar_masks_host.emplace(grammar_staged_config_offset(grammar_masks_device.bytes()) +
                                   kMaximumConcurrency * sizeof(ops::SamplingConfig));
    }
    active_continuations.fill(continuation_capacity);
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) { lane_epochs[lane] = 1; }
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        SequenceState& sequence = continuation_states[index];
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_digests.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.long_anchors.reserve(context_cache.max_long_anchors_per_continuation.value_or(0));
        // One retained shared resume source can coexist with every fixed per-request candidate.
        sequence.shared_prefix_references.reserve(8U);
    }
    materialization_ledger_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_identity_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_prefix_digests_.reserve(static_cast<std::size_t>(capacity) + 1ULL);

    set_device_i32(io.text_kv_table_row, 0);
    if (!causal_scoring) { set_device_i32(io.backend_kv_table_row, 0); }
    if (peer) { bind_tensor_parallel_execution(); }

    host_tokens = round_host ? static_cast<TokenId*>(round_host->data()) : nullptr;
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
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    if (!causal_scoring) {
        CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
        CUDA_CHECK(
            cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    }
    if (plan.persistent.qwen4) { construct_qwen4(plan, backing); }
    if (elastic_arena) {
        auto* offload = dynamic_cast<execution::Qwen4OffloadExperts*>(qwen4_experts.get());
        if (offload == nullptr || (peer && !peer->elastic_arena)) {
            throw std::logic_error("elastic KV needs the offload expert backend on every rank");
        }
        std::vector<Qwen4ElasticVram::Rank> ranks;
        ranks.push_back(Qwen4ElasticVram::Rank{
            elastic_arena.get(), device.stream, &plan.persistent, &decoder->text_kv.page_pool(),
            decoder->mtp_cache() != nullptr ? &decoder->mtp_cache()->page_pool() : nullptr});
        if (peer) {
            ranks.push_back(Qwen4ElasticVram::Rank{
                peer->elastic_arena.get(), peer->device.stream, &*plan.peer_persistent,
                &peer->decoder->text_kv.page_pool(),
                peer->decoder->mtp_cache() != nullptr ? &peer->decoder->mtp_cache()->page_pool()
                                                      : nullptr});
        }
        const std::uint32_t step = (plan.elastic_kv.grow_tokens + kPagedKVPageSize - 1U) /
                                   static_cast<std::uint32_t>(kPagedKVPageSize);
        qwen4_elastic = std::make_unique<Qwen4ElasticVram>(std::move(ranks), *offload, step,
                                                           plan.elastic_kv.kv_budget_bytes);
    }
    synchronize_devices();
    if (use_cuda_graph) {
        StartupPhaseScope graph_phase(startup_observer, StartupPhase::CudaGraphPrepare);
        prepare_graphs();
        graph_phase.complete();
    }
    work.reset();
    work.reset_peak();
    if (peer) { peer->work.reset(); }
    workspace_logical_peak_bytes = 0;
    initialize_disk_cache();
}

namespace {

// NINFER_QWEN4_MTP_PROBE=<file>: appends every MTP-layer routed-expert call's inputs (eager only):
// per call I32 T, then x BF16 [2560,T], ids I32 [10,T], weights FP32 [10,T] (T24 calibration).
class Qwen4MtpProbeFile final : public execution::Qwen4MoeProbe {
public:
    Qwen4MtpProbeFile(const char* path, std::int32_t layer) : layer_(layer) {
        file_ = std::fopen(path, "ab");
        if (file_ == nullptr) { throw std::runtime_error("cannot open NINFER_QWEN4_MTP_PROBE"); }
    }

    ~Qwen4MtpProbeFile() override { std::fclose(file_); }

    [[nodiscard]] bool wants(std::int32_t layer) const override { return layer == layer_; }

    void capture(std::int32_t, const Tensor& x, const Tensor& ids, const Tensor& weights,
                 cudaStream_t stream) override {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::int32_t T = x.ne[1];
        std::fwrite(&T, sizeof(T), 1, file_);
        for (const Tensor* t : {&x, &ids, &weights}) {
            std::vector<std::byte> host(t->bytes());
            CUDA_CHECK(cudaMemcpy(host.data(), t->data, host.size(), cudaMemcpyDeviceToHost));
            std::fwrite(host.data(), 1, host.size(), file_);
        }
        std::fflush(file_);
    }

private:
    std::FILE* file_ = nullptr;
    std::int32_t layer_;
};

} // namespace

void ProgramImpl::construct_qwen4(const SequencePlanImpl& plan, DeviceSpan backing) {
    const auto& config = parameters.model.config().text;
    const auto& q      = *plan.persistent.qwen4;
    if (!parameters.qwen4 ||
        (peer && (!peer_parameters || !peer_parameters->qwen4 || !tp_execution))) {
        throw std::logic_error("Text forward Program state requires its blocks on every rank");
    }
    execution::Qwen4Runtime runtime;
    execution::Qwen4Runtime peer_runtime;
    // Routed experts (MoE); a dense FFN binds no expert backend (runtime.experts stays null).
    if (config.qwen4_routed_experts()) {
        construct_qwen4_experts(plan, backing, runtime, peer_runtime);
    }
    // PLE: the n-gram table gather (mapped, with prefetch), its staging and the convolution state.
    // At tp 2 rank 0 gathers once and both ranks upload the same staging (replicated PLE); each
    // rank keeps its own convolution states (its StateImage, mirrored slot operations).
    if (config.ple) {
        const auto& block = parameters.qwen4->layers.at(config.ple->layer);
        if (!block.ple) { throw std::logic_error("Qwen4Exp PLE layer has no PLE parameters"); }
        qwen4_ple_gather = std::make_unique<execution::PleGather>(
            block.ple->table, block.ple->hash, execution::PleResidency::Mapped, true);
        if (qwen4_ple_gather->width() != config.ple->embed_dim) {
            throw std::logic_error("Qwen4Exp PLE gather width does not match the configuration");
        }
        qwen4_ple_host.emplace(static_cast<std::size_t>(config.ple->embed_dim) * prefill_chunk *
                               sizeof(std::uint16_t));
        runtime.ple_gather = qwen4_ple_gather.get();
        runtime.ple_host   = static_cast<std::uint16_t*>(qwen4_ple_host->data());
        runtime.ple_states = state_images->ple_states(0);
        if (q.ple_record) { runtime.ple_record = q.ple_record->bind(backing); }
        if (q.decode_ple_rows) {
            runtime.decode_ple_rows = q.decode_ple_rows->bind(backing);
            qwen4_decode_ple_buffer.emplace(runtime.decode_ple_rows.bytes());
            qwen4_decode_ple_host = static_cast<std::uint16_t*>(qwen4_decode_ple_buffer->data());
            std::fill_n(qwen4_decode_ple_host, runtime.decode_ple_rows.numel(), std::uint16_t{0});
            runtime.decode_ple_host = qwen4_decode_ple_host;
        }
        if (peer) {
            peer_runtime.ple_states = peer->state_images->ple_states(0);
            if (q.decode_ple_rows) {
                if (!peer->qwen4_decode_ple_rows.data) {
                    throw std::logic_error("Qwen4Exp rank 1 has no decode PLE rows");
                }
                peer_runtime.decode_ple_rows = peer->qwen4_decode_ple_rows;
                peer_runtime.decode_ple_host = qwen4_decode_ple_host;
                peer_runtime.ple_record      = peer->qwen4_ple_record;
            }
        }
        if (q.decode_ple_rows) {
            // Deferred decode rows: one mapped word per rank (own cache line), consumed by the
            // rank's PLE layer; the rounds raise them after gathering the rows post-launch.
            // Allocated before any capture (the graphs bake in the addresses).
            constexpr std::size_t kLine = 64;
            qwen4_ple_staged_words.emplace(2 * kLine, HostMapping::Yes);
            auto* base = static_cast<std::byte*>(qwen4_ple_staged_words->data());
            if (qwen4_ple_staged_words->device_data() != qwen4_ple_staged_words->data()) {
                throw std::logic_error("Qwen4Exp deferred PLE rows need unified addressing");
            }
            std::fill_n(base, 2 * kLine, std::byte{0});
            qwen4_ple_staged[0]       = reinterpret_cast<std::uint32_t*>(base);
            qwen4_ple_staged[1]       = reinterpret_cast<std::uint32_t*>(base + kLine);
            runtime.decode_ple_staged = qwen4_ple_staged[0];
            if (peer) { peer_runtime.decode_ple_staged = qwen4_ple_staged[1]; }
            qwen4_ple_stager =
                std::make_unique<Qwen4PleStager>([this] { stage_qwen4_ple_round(); });
        }
    }
    // Validation token sequences: one whole sequence (prompt + continuation) per line.
    const auto read_sequences = [](const char* path, const char* variable) {
        std::vector<std::vector<TokenId>> sequences;
        std::FILE* file = std::fopen(path, "r");
        if (file == nullptr) { throw std::runtime_error(std::string("cannot open ") + variable); }
        std::vector<TokenId> line;
        long long value = 0;
        int c           = 0;
        bool in_number  = false;
        while ((c = std::fgetc(file)) != EOF) {
            if (c >= '0' && c <= '9') {
                value     = (in_number ? value * 10 : 0) + (c - '0');
                in_number = true;
                continue;
            }
            if (in_number) { line.push_back(static_cast<TokenId>(value)); }
            in_number = false;
            if (c == '\n' && !line.empty()) {
                sequences.push_back(std::move(line));
                line = {};
            }
        }
        if (in_number) { line.push_back(static_cast<TokenId>(value)); }
        if (!line.empty()) { sequences.push_back(std::move(line)); }
        std::fclose(file);
        return sequences;
    };
    execution::require_qwen4_validation_opt_in();
    if (const char* oracle = std::getenv("NINFER_QWEN4_MTP_ORACLE");
        oracle != nullptr && *oracle != '\0' && speculative_backend == SpeculativeBackend::Mtp) {
        qwen4_mtp_oracle = read_sequences(oracle, "NINFER_QWEN4_MTP_ORACLE");
    }
    if (const char* forced = std::getenv("NINFER_QWEN4_FORCE_TOKENS");
        forced != nullptr && *forced != '\0') {
        qwen4_forced_tokens = read_sequences(forced, "NINFER_QWEN4_FORCE_TOKENS");
    }
    if (const char* none = std::getenv("NINFER_QWEN4_MTP_NO_DRAFTS")) {
        qwen4_mtp_no_drafts = std::string_view(none) == "1";
    }
    if (const char* cap = std::getenv("NINFER_QWEN4_MTP_EXTENT_CAP"); cap != nullptr && *cap != 0) {
        qwen4_mtp_extent_cap = static_cast<std::uint32_t>(std::stoul(cap));
    }
    if (const char* calib = std::getenv("NINFER_QWEN4_MTP_CALIB");
        calib != nullptr && *calib != '\0' && speculative_backend == SpeculativeBackend::Mtp) {
        qwen4_mtp_calib = std::fopen(calib, "a");
        if (qwen4_mtp_calib == nullptr) {
            throw std::runtime_error(std::string("NINFER_QWEN4_MTP_CALIB: cannot open ") + calib);
        }
    }
    qwen4_replay = execution::Qwen4Replay::from_environment();
    (void)execution::tp_rank_probe_enabled(); // T123: allocate the probe staging before any capture
    if (const char* rounds = std::getenv("NINFER_QWEN4_ROUND_LOG");
        rounds != nullptr && *rounds != '\0') {
        qwen4_round_log = std::fopen(rounds, "a");
        if (qwen4_round_log == nullptr) {
            throw std::runtime_error(std::string("NINFER_QWEN4_ROUND_LOG: cannot open ") + rounds);
        }
    }
    if (const char* record = std::getenv("NINFER_QWEN4_REPLAY_RECORD");
        record != nullptr && *record != '\0') {
        qwen4_replay_record = std::fopen(record, "a");
        if (qwen4_replay_record == nullptr) {
            throw std::runtime_error(std::string("NINFER_QWEN4_REPLAY_RECORD: cannot open ") +
                                     record);
        }
    }
    if (const char* top2 = std::getenv("NINFER_QWEN4_TOP2"); top2 != nullptr && *top2 != '\0') {
        qwen4_top2_file = std::fopen(top2, "a");
        if (qwen4_top2_file == nullptr) {
            throw std::runtime_error("cannot open NINFER_QWEN4_TOP2");
        }
    }
    // Validation dumps (NINFER_QWEN4_TFDUMP): hidden taps only where execution is eager.
    qwen4_tfdump = execution::Qwen4TfDump::from_environment(
        static_cast<std::int32_t>(config.num_hidden_layers));
    if (qwen4_tfdump) {
        qwen4_tfdump->write_meta(peer ? "capy-flash engine, tp 2" : "capy-flash engine, tp 1",
                                 parameters.model.info().name);
        if (!use_cuda_graph) { runtime.taps = qwen4_tfdump.get(); }
    }
    // T43: the draft shortlist head. The prefix is a view of rank 0's output head; the extra rows
    // are copied byte-exact from the output head (at tp 2 from either rank's vocabulary half,
    // rank 1's over UVA) into rank 0's planes, in ascending token ID order.
    if (q.draft_ids && plan.draft_split) {
        // Split over the ranks: rank 0 scores shortlist rows [0, R) through a view of its half,
        // rank 1 rows [R, 2R), gathered (ascending token IDs) from rank 0's half over UVA and
        // from its own half.
        const auto& head          = parameters.text.output_head;
        const std::int32_t prefix = q.draft_prefix;
        const auto rows           = prefix + static_cast<std::int32_t>(plan.draft_extras.size());
        const std::int32_t half   = rows / 2;
        if (!peer || !peer_parameters || !plan.peer_persistent || !plan.peer_persistent->qwen4 ||
            plan.peer_persistent->qwen4->draft_extra != half || half > prefix ||
            peer->qwen4_draft_planes[0] == nullptr) {
            throw std::logic_error("Qwen4Exp split draft shortlist is not planned on both ranks");
        }
        std::vector<std::int32_t> ids(static_cast<std::size_t>(rows));
        for (std::int32_t i = 0; i < prefix; ++i) { ids[static_cast<std::size_t>(i)] = i; }
        std::copy(plan.draft_extras.begin(), plan.draft_extras.end(), ids.begin() + prefix);
        Tensor device_ids = q.draft_ids->bind(backing);
        CUDA_CHECK(cudaMemcpyAsync(device_ids.data, ids.data(), device_ids.bytes(),
                                   cudaMemcpyHostToDevice, device.stream));
        const auto& peer_head    = peer_parameters->text.output_head;
        const std::int32_t rows0 = head.weight.n;
        const std::span<const std::int32_t> theirs(ids.data() + half,
                                                   static_cast<std::size_t>(half));
        const auto cut = std::lower_bound(theirs.begin(), theirs.end(), rows0);
        const std::vector<std::int32_t> from0(theirs.begin(), cut);
        std::vector<std::int32_t> from1(cut, theirs.end());
        for (auto& row : from1) { row -= rows0; }
        execution::gather_draft_rows(head.weight, from0, peer->qwen4_draft_planes, 0,
                                     device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        {
            const ScopedCurrentDevice rank1(peer->device.device);
            execution::gather_draft_rows(peer_head.weight, from1, peer->qwen4_draft_planes,
                                         static_cast<std::int32_t>(from0.size()),
                                         peer->device.stream);
            CUDA_CHECK(cudaStreamSynchronize(peer->device.stream));
        }
        execution::Qwen4DraftHead draft;
        draft.split       = true;
        draft.prefix_rows = half;
        draft.prefix      = execution::draft_row_view(head, 0, half);
        draft.peer_rows   = half;
        draft.peer        = execution::draft_rows_weight(peer_head, peer->qwen4_draft_planes, half);
        draft.token_ids   = static_cast<const std::int32_t*>(device_ids.data);
        qwen4_draft_head  = draft;
        runtime.draft_head = &*qwen4_draft_head;
    } else if (q.draft_ids) {
        const auto& head          = parameters.text.output_head;
        const std::int32_t prefix = q.draft_prefix;
        const std::int32_t extra  = q.draft_extra;
        if (static_cast<std::size_t>(extra) != plan.draft_extras.size() || prefix > head.weight.n) {
            throw std::logic_error("Qwen4Exp draft shortlist plan does not match its layout");
        }
        execution::Qwen4DraftHead draft;
        draft.prefix_rows = prefix;
        draft.extra_rows  = extra;
        if (prefix > 0) { draft.prefix = execution::draft_row_view(head, 0, prefix); }
        Tensor ids = q.draft_ids->bind(backing);
        std::vector<std::int32_t> host(static_cast<std::size_t>(prefix + extra));
        for (std::int32_t i = 0; i < prefix; ++i) { host[static_cast<std::size_t>(i)] = i; }
        std::copy(plan.draft_extras.begin(), plan.draft_extras.end(), host.begin() + prefix);
        CUDA_CHECK(cudaMemcpyAsync(ids.data, host.data(), ids.bytes(), cudaMemcpyHostToDevice,
                                   device.stream));
        draft.token_ids = static_cast<const std::int32_t*>(ids.data);
        if (extra > 0) {
            std::array<std::byte*, 3> planes{};
            for (std::size_t p = 0; p < planes.size(); ++p) {
                planes[p] = static_cast<std::byte*>(q.draft_planes[p].bind(backing).data);
            }
            draft.extra = execution::draft_rows_weight(head, planes, extra);
            // Rank 0 owns head rows [0, rows0): all of them at one device.
            const std::int32_t rows0 = head.weight.n;
            const auto split =
                std::lower_bound(plan.draft_extras.begin(), plan.draft_extras.end(), rows0);
            const std::vector<std::int32_t> own(plan.draft_extras.begin(), split);
            execution::gather_draft_rows(head.weight, own, planes, 0, device.stream);
            if (split != plan.draft_extras.end()) {
                if (!peer || !peer_parameters) {
                    throw std::logic_error("Qwen4Exp draft shortlist rows outside the output head");
                }
                std::vector<std::int32_t> theirs(split, plan.draft_extras.end());
                for (auto& row : theirs) { row -= rows0; }
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                const ScopedCurrentDevice rank1(peer->device.device);
                execution::gather_draft_rows(peer_parameters->text.output_head.weight, theirs,
                                             planes, static_cast<std::int32_t>(own.size()),
                                             peer->device.stream);
                CUDA_CHECK(cudaStreamSynchronize(peer->device.stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        qwen4_draft_head   = draft;
        runtime.draft_head = &*qwen4_draft_head;
    }
    qwen4_runtime_record = runtime;
    if (peer) {
        qwen4_peer_runtime_record = peer_runtime;
        tp_execution->qwen4       = &*qwen4_peer_runtime_record;
    }
}

// The routed-expert backends of a Qwen4Exp MoE on every rank: the in-place expert tables, the MTP
// layer's bank and its divisor planes, the offload or mapped backend, bound into each rank's
// runtime.
void ProgramImpl::construct_qwen4_experts(const SequencePlanImpl& plan, DeviceSpan backing,
                                          execution::Qwen4Runtime& runtime,
                                          execution::Qwen4Runtime& peer_runtime) {
    const auto& config = parameters.model.config().text;
    const auto& q      = *plan.persistent.qwen4;
    if (!q.expert_table || (peer && !peer->qwen4_expert_table.data)) {
        throw std::logic_error("Qwen4Exp routed experts have no planned expert table");
    }
    // The in-place expert tables: rank r reads expert e from the host-mapped bank when it is in
    // its partition (all experts at one device, e % 2 == r at tp 2: T14's ID-parity partition,
    // the routing being identical on both ranks), -1 otherwise.
    const auto upload_table = [&](Tensor table, int rank, int ranks, cudaStream_t stream) {
        std::vector<std::int32_t> slots(static_cast<std::size_t>(table.numel()));
        for (std::size_t e = 0; e < slots.size(); ++e) {
            slots[e] = static_cast<int>(e % static_cast<std::size_t>(ranks)) == rank
                           ? static_cast<std::int32_t>(e)
                           : -1;
        }
        CUDA_CHECK(cudaMemcpyAsync(table.data, slots.data(), table.bytes(), cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    const int ranks = peer ? execution::kTensorParallelWidth : 1;
    // MTP layer (T24): its bank, with a bank-wide weight divisor (ninfer-ext's artifact)
    // expanded into the per-expert planes moe_slots indexes (device copy for the zero-copy
    // backend, host copy for the offload backend's CPU path and its device divisor table).
    const bool mtp = speculative_backend == SpeculativeBackend::Mtp;
    std::optional<execution::Qwen4ExpertBank> mtp_host_bank;
    if (mtp) {
        if (!parameters.qwen4_mtp) { throw std::logic_error("Qwen4Exp MTP has no MTP layer"); }
        qwen4_mtp_bank = parameters.qwen4_mtp->layer.moe().experts;
        mtp_host_bank  = qwen4_mtp_bank;
        if (parameters.qwen4_mtp->uniform_gate_up_divisor > 0.0F) {
            if (!q.mtp_gate_up_divisors || !q.mtp_down_divisors) {
                throw std::logic_error("Qwen4Exp MTP divisor planes are not planned");
            }
            Tensor gate_up = q.mtp_gate_up_divisors->bind(backing);
            Tensor down    = q.mtp_down_divisors->bind(backing);
            qwen4_mtp_host_divisors.assign(static_cast<std::size_t>(gate_up.numel()),
                                           parameters.qwen4_mtp->uniform_gate_up_divisor);
            qwen4_mtp_host_divisors.resize(static_cast<std::size_t>(gate_up.numel() + down.numel()),
                                           parameters.qwen4_mtp->uniform_down_divisor);
            CUDA_CHECK(cudaMemcpy(gate_up.data, qwen4_mtp_host_divisors.data(), gate_up.bytes(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(down.data, qwen4_mtp_host_divisors.data() + gate_up.numel(),
                                  down.bytes(), cudaMemcpyHostToDevice));
            qwen4_mtp_bank->gate_up_divisors = static_cast<const float*>(gate_up.data);
            qwen4_mtp_bank->down_divisors    = static_cast<const float*>(down.data);
            mtp_host_bank->gate_up_divisors  = qwen4_mtp_host_divisors.data();
            mtp_host_bank->down_divisors =
                qwen4_mtp_host_divisors.data() + static_cast<std::size_t>(gate_up.numel());
        }
    }
    // Routed experts. At one device the table is the identity (the offload backend's device
    // reads of the host bank use it too).
    Tensor table = q.expert_table->bind(backing);
    upload_table(table, 0, ranks, device.stream);
    std::optional<execution::Qwen4OffloadExperts::Peer> offload_peer;
    if (plan.qwen4_experts.backend == RoutedExpertBackend::Offload) {
        if (peer) {
            // tp 2: one cache over both GPUs (partition by ID parity), the CPU path on rank 0.
            if (!q.cache_mirror || peer->qwen4_cache_table == nullptr ||
                peer->qwen4_cache_divisors == nullptr) {
                throw std::logic_error("Qwen4Exp tp 2 offloaded experts are not planned on rank 1");
            }
            offload_peer.emplace();
            offload_peer->device        = peer->device.device;
            offload_peer->stream        = peer->device.stream;
            offload_peer->pool          = peer->qwen4_cache_pool;
            offload_peer->table         = peer->qwen4_cache_table;
            offload_peer->divisors      = peer->qwen4_cache_divisors;
            offload_peer->stream_tables = peer->qwen4_stream_tables;
            offload_peer->mirror        = peer->qwen4_cache_mirror;
            offload_peer->reads         = peer->qwen4_cache_reads;
            offload_peer->stage_table   = peer->qwen4_stage_table;
            offload_peer->picks         = peer->qwen4_stage_picks;
            offload_peer->staging       = peer->qwen4_staging;
            offload_peer->slot_offsets  = peer->qwen4_slot_offsets;
        }
        if (q.expert_slots <= 0 ||
            q.cache_layers !=
                static_cast<std::int32_t>(parameters.qwen4->layers.size()) + (mtp ? 1 : 0)) {
            throw std::logic_error("Qwen4Exp offloaded experts need a resolved slot count");
        }
        // Cache layers: the Text layers, then the MTP layer (layer num_hidden_layers).
        std::vector<execution::Qwen4ExpertBank> banks;
        banks.reserve(static_cast<std::size_t>(q.cache_layers));
        for (const auto& layer : parameters.qwen4->layers) { banks.push_back(layer.moe().experts); }
        if (mtp) { banks.push_back(*mtp_host_bank); }
        execution::Qwen4OffloadExperts::DeviceMemory memory;
        memory.pool     = q.bind_pool(elastic_arena ? elastic_arena->span() : backing);
        memory.table    = static_cast<std::int32_t*>(q.cache_table->bind(backing).data);
        memory.divisors = static_cast<float*>(q.cache_divisors->bind(backing).data);
        memory.identity = static_cast<std::int32_t*>(table.data);
        memory.reads    = static_cast<std::int32_t*>(q.cache_reads->bind(backing).data);
        if (q.cache_slot_offsets) {
            memory.slot_offsets =
                static_cast<std::int64_t*>(q.cache_slot_offsets->bind(backing).data);
        }
        if (q.cache_mirror) {
            memory.mirror = static_cast<std::int32_t*>(q.cache_mirror->bind(backing).data);
        }
        if (q.staging_slots > 0) {
            if (!q.prefetch_tags || !q.prefetch_epoch || !q.prefetch_rows) {
                throw std::logic_error("Qwen4Exp prefetch memory is not planned");
            }
            memory.prefetch.tags = static_cast<std::uint32_t*>(q.prefetch_tags->bind(backing).data);
            memory.prefetch.epoch =
                static_cast<std::uint32_t*>(q.prefetch_epoch->bind(backing).data);
            memory.prefetch.rows = static_cast<std::int32_t*>(q.prefetch_rows->bind(backing).data);
            memory.prefetch.predict_workspace = q.prefetch_workspace.bind(backing).data;
        }
        memory.stream_tables =
            static_cast<std::int32_t*>(q.cache_stream_tables->bind(backing).data);
        if (q.stage_table) {
            memory.stage_table = static_cast<std::int32_t*>(q.stage_table->bind(backing).data);
            memory.picks       = static_cast<std::int32_t*>(q.stage_picks->bind(backing).data);
            for (std::size_t plane = 0; plane < memory.staging.size(); ++plane) {
                memory.staging[plane] =
                    static_cast<std::byte*>(q.stage_pool[plane].bind(backing).data);
            }
        }
        execution::Qwen4OffloadOptions options;
        options.slots_per_layer = q.expert_slots;
        options.elastic_columns = q.elastic_columns;
        // 0 (auto) is resolved by the Engine at load; callers that plan without it get the same
        // rule.
        options.cpu_threads          = plan.qwen4_experts.cpu_threads != 0
                                           ? static_cast<std::int32_t>(plan.qwen4_experts.cpu_threads)
                                           : ops::host::auto_expert_cpu_threads(
                                        ops::host::host_cpu_topology(),
                                        parameters.qwen4->layers.front().moe().experts.gguf);
        options.insert_cap           = static_cast<std::int32_t>(plan.qwen4_experts.insert_cap);
        options.admit_threshold      = plan.qwen4_experts.admit_threshold;
        options.half_life_tokens     = plan.qwen4_experts.half_life_tokens;
        options.evict_tau_windows    = plan.qwen4_experts.evict_tau_windows;
        options.prompt_tokens        = static_cast<std::int32_t>(prefill_chunk);
        const MissSplitOptions split = execution::effective_miss_split(plan.qwen4_experts);
        options.split                = execution::host_miss_split(split);
        options.split_staged         = split.mode != MissSplitMode::Off && split.staged;
        options.split_on_peer        = offload_peer.has_value() && split.peer_share;
        options.split_calibrate      = split.mode == MissSplitMode::Auto && split.calibrate;
        options.stream_min_tokens = static_cast<std::int32_t>(plan.qwen4_experts.stream_min_tokens);
        options.gguf_stream_min_tokens =
            static_cast<std::int32_t>(plan.qwen4_experts.gguf_stream_min_tokens);
        options.stream_ring_slots = static_cast<std::int32_t>(plan.qwen4_experts.stream_ring_slots);
        options.stream_cpu_tail   = plan.qwen4_experts.stream_cpu_tail;
        options.stream_cpu_max_tokens =
            static_cast<std::int32_t>(plan.qwen4_experts.stream_cpu_max_tokens);
        options.stream_cpu_expert_us = plan.qwen4_experts.stream_cpu_expert_us;
        options.stream_cpu_token_us  = plan.qwen4_experts.stream_cpu_token_us;
        options.stream_link_us       = plan.qwen4_experts.stream_link_us;
        // T42: chunks wider than the base chunk run on workspace lent from the slot pool.
        options.base_chunk             = static_cast<std::int32_t>(plan.qwen4_base_chunk);
        options.prompt_workspace_bytes = static_cast<std::int64_t>(plan.workspace.qwen4_prompt);
        for (const auto& [width, bytes] : plan.workspace.qwen4_prompt_steps) {
            options.prompt_workspace_steps.emplace_back(static_cast<std::int32_t>(width),
                                                        static_cast<std::int64_t>(bytes));
        }
        options.text_layers = static_cast<std::int32_t>(parameters.qwen4->layers.size());
        options.prefetch.slots_per_layer = q.staging_slots / 2;
        options.prefetch.min_score       = plan.qwen4_experts.prefetch_min_score;
        options.prefetch.first_layer =
            static_cast<std::int32_t>(plan.qwen4_experts.prefetch_first_layer);
        options.prefetch.link_gbps = plan.qwen4_experts.prefetch_link_gbps;
        options.warm_start         = plan.qwen4_experts.warm_start;
        // Validation / measurement aid (off by default): per-window JSON lines.
        if (const char* trace = std::getenv("NINFER_QWEN4_EXPERT_TRACE")) {
            options.trace_path = trace;
        }
        if (const char* dump = std::getenv("NINFER_QWEN4_ROUTING_DUMP")) {
            options.routing_path = dump;
        }
        if (const char* freeze = std::getenv("NINFER_QWEN4_EXPERT_FREEZE")) {
            options.freeze_after_prompt = std::string_view(freeze) == "1";
            // T59 fast bench: nothing is ever inserted (residency = initialization / snapshot).
            options.freeze_all = std::string_view(freeze) == "all";
        }
        if (const char* sync = std::getenv("NINFER_QWEN4_EXPERT_SYNC")) {
            options.sync_fills = std::string_view(sync) == "1";
        }
        if (const char* snap = std::getenv("NINFER_QWEN4_EXPERT_SNAPSHOT");
            snap != nullptr && *snap != 0) {
            options.snapshot_load = snap;
        }
        if (const char* snap = std::getenv("NINFER_QWEN4_EXPERT_SNAPSHOT_SAVE");
            snap != nullptr && *snap != 0) {
            options.snapshot_save = snap;
        }
        qwen4_experts = std::make_unique<execution::Qwen4OffloadExperts>(
            banks, options, memory, device.device, device.stream,
            offload_peer ? &*offload_peer : nullptr);
        qwen4_expert_slots      = static_cast<std::uint32_t>(q.expert_slots);
        qwen4_expert_pool_bytes = q.cache_pool_bytes;
    } else {
        // The mapped backend has every expert kernel read its bank in place: v0.7's pageable
        // fallback banks (--host-pinned-max-gib, or a refused page-locked allocation) cannot.
        const auto pinned = [](const execution::Qwen4ExpertBank& bank) {
            const ops::MoeSlotPool& b = bank.bank;
            if (bank.gguf) { // rows outside the NVFP4 planes (issue #22)
                return expert_cache::pinned_host_memory(bank.gguf_gate_up_rows) &&
                       expert_cache::pinned_host_memory(bank.gguf_down_rows);
            }
            return expert_cache::pinned_host_memory(b.gate_up_codes.base) &&
                   expert_cache::pinned_host_memory(b.gate_up_scales.base) &&
                   expert_cache::pinned_host_memory(b.down_codes.base) &&
                   expert_cache::pinned_host_memory(b.down_scales.base);
        };
        bool all_pinned = !mtp || pinned(*qwen4_mtp_bank);
        for (const auto& layer : parameters.qwen4->layers) {
            all_pinned = all_pinned && pinned(layer.moe().experts);
        }
        if (!all_pinned) {
            throw std::invalid_argument(
                "the mapped expert backend reads every expert bank in place from page-locked "
                "memory, and part of the banks is in pageable memory (page-locked limit); use "
                "the offload backend");
        }
        qwen4_experts = std::make_unique<execution::Qwen4MappedExperts>(table);
    }
    if (use_cuda_graph && !qwen4_experts->graph_capturable()) {
        throw std::logic_error("Qwen4Exp expert backend cannot run inside a CUDA Graph");
    }
    runtime.experts = qwen4_experts.get();
    if (mtp) {
        // The offload backend serves the MTP layer itself (resident slots + CPU, its own cache
        // layer); the zero-copy one reads the MTP bank in place through the same backend.
        runtime.mtp_experts     = qwen4_experts.get();
        runtime.mtp_expert_bank = &*qwen4_mtp_bank;
        if (const char* probe = std::getenv("NINFER_QWEN4_MTP_PROBE");
            probe != nullptr && *probe != '\0') {
            if (use_cuda_graph) {
                throw std::invalid_argument("NINFER_QWEN4_MTP_PROBE requires eager execution");
            }
            qwen4_mtp_probe = std::make_unique<Qwen4MtpProbeFile>(
                probe, static_cast<std::int32_t>(config.num_hidden_layers));
            runtime.moe_probe = qwen4_mtp_probe.get();
        }
    }
    if (peer) {
        const ScopedCurrentDevice rank1(peer->device.device);
        upload_table(peer->qwen4_expert_table, 1, ranks, peer->device.stream);
        if (offload_peer) {
            qwen4_peer_experts = std::make_unique<execution::Qwen4OffloadPeerExperts>(
                q.cache_layers, q.expert_slots, *offload_peer,
                static_cast<execution::Qwen4OffloadExperts*>(qwen4_experts.get()),
                q.elastic_columns);
        } else {
            qwen4_peer_experts =
                std::make_unique<execution::Qwen4MappedExperts>(peer->qwen4_expert_table);
        }
        peer_runtime.experts = qwen4_peer_experts.get();
        if (mtp) {
            // MTP at tp 2 (T33): rank 1 runs the MTP layer's experts of its partition through its
            // offload view (layer num_hidden_layers of the shared cache); the bank only supplies
            // the activation divisors there.
            if (!offload_peer || !peer_parameters->qwen4_mtp) {
                throw std::invalid_argument(
                    "Qwen4Exp MTP at tp 2 requires the offload expert backend and both ranks' "
                    "MTP layers");
            }
            qwen4_peer_mtp_bank          = peer_parameters->qwen4_mtp->layer.moe().experts;
            peer_runtime.mtp_experts     = qwen4_peer_experts.get();
            peer_runtime.mtp_expert_bank = &*qwen4_peer_mtp_bank;
        }
    }
}

void ProgramImpl::qwen4_begin_window(execution::Qwen4Window kind) {
    nvtx::ScopedRange range(nvtx::Name::Qwen4BeginWindow, nvtx::Category::Moe);
    if (qwen4_experts) { qwen4_experts->begin_window(device.stream, kind); }
    if (qwen4_peer_experts) {
        try {
            const ScopedCurrentDevice rank1(peer->device.device);
            qwen4_peer_experts->begin_window(peer->device.stream, kind);
        } catch (...) {
            if (qwen4_experts) { qwen4_experts->abandon_window(device.stream); }
            throw;
        }
    }
}

void ProgramImpl::qwen4_observe_window() {
    if (qwen4_experts) { qwen4_experts->observe_window(device.stream); }
    if (qwen4_peer_experts) {
        const ScopedCurrentDevice rank1(peer->device.device);
        qwen4_peer_experts->observe_window(peer->device.stream);
    }
}

void ProgramImpl::qwen4_end_window() {
    nvtx::ScopedRange range(nvtx::Name::Qwen4EndWindow, nvtx::Category::Moe);
    if (qwen4_experts) { qwen4_experts->end_window(device.stream); }
    if (qwen4_peer_experts) {
        const ScopedCurrentDevice rank1(peer->device.device);
        qwen4_peer_experts->end_window(peer->device.stream);
    }
}

void ProgramImpl::log_qwen4_round(char kind, std::uint32_t frontier, std::uint32_t extent,
                                  std::int32_t accepted, double round_seconds) {
    if (qwen4_round_log == nullptr) { return; }
    const auto now = std::chrono::steady_clock::now();
    const double gap =
        qwen4_round_follows
            ? std::chrono::duration<double>(now - qwen4_round_end).count() - round_seconds
            : -1e-6;
    long long missing = 0, cpu = 0, gpu = 0;
    double wall = 0, fill_wait = 0;
    if (const auto* offload =
            dynamic_cast<const execution::Qwen4OffloadExperts*>(qwen4_experts.get())) {
        const execution::Qwen4OffloadWindowStats& w = offload->last_window();
        missing                                     = static_cast<long long>(w.missing);
        cpu                                         = static_cast<long long>(w.host_experts);
        gpu                                         = static_cast<long long>(w.gpu_read_experts);
        wall                                        = w.wall_us;
        fill_wait                                   = w.fill_wait_us;
    }
    const int columns = qwen4_elastic ? qwen4_elastic->stats().columns : -1;
    std::fprintf(qwen4_round_log, "%c %u %u %d %.1f %.1f %lld %lld %lld %.1f %.1f %d\n", kind,
                 frontier, extent, accepted, 1e6 * round_seconds, 1e6 * gap, missing, cpu, gpu,
                 wall, fill_wait, columns);
    qwen4_round_end     = std::chrono::steady_clock::now();
    qwen4_round_follows = true;
}

void ProgramImpl::qwen4_abandon_window() noexcept {
    if (qwen4_experts) { qwen4_experts->abandon_window(device.stream); }
    if (qwen4_peer_experts) {
        try {
            const ScopedCurrentDevice rank1(peer->device.device);
            qwen4_peer_experts->abandon_window(peer->device.stream);
        } catch (...) {}
    }
}

void ProgramImpl::prefetch_qwen4_ple_next_round(std::span<const std::uint32_t> lanes, bool mtp) {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_T53_PLE_PREFETCH");
        return v == nullptr || *v == 0 || std::atoi(v) != 0;
    }();
    if (!enabled || !qwen4_ple_gather || (mtp ? !mtp_host_egress : !ordinary_host_egress)) {
        return;
    }
    const std::size_t context = parameters.model.config().text.ple->ngram_size - 1U;
    if (!mtp) {
        // The next round's column is the sampled token; its context the ledger's last tokens.
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const SequenceState& sequence = active_sequence(lanes[row]);
            const std::size_t ledger      = sequence.ledger.size();
            const std::size_t before      = std::min(context, ledger);
            std::array<std::int32_t, 8> window{};
            if (before + 1U > window.size()) { return; }
            std::size_t n = 0;
            for (std::size_t i = ledger - before; i < ledger; ++i) {
                window[n++] = sequence.ledger[i];
            }
            window[n++] = ordinary_host_egress->sampled_tokens[row];
            qwen4_ple_gather->prefetch(std::span<const std::int32_t>(window.data(), n), before, 1);
        }
        return;
    }
    const std::uint32_t width = draft_window + 1U;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const SequenceState& sequence = active_sequence(lanes[row]);
        const std::int32_t count      = mtp_host_egress->licensed_counts[row];
        if (count <= 0 || static_cast<std::uint32_t>(count) > width) { continue; }
        // After the commit the ledger is the current one + the licensed tokens; the next round
        // reads the `context` tokens before its anchor (the last licensed token), the anchor,
        // then the drafts (padding columns repeat the anchor), as stage_qwen4_ple_rows does.
        const TokenId* licensed  = mtp_host_egress->licensed_tokens.data() + row * width;
        const std::size_t ledger = sequence.ledger.size();
        const std::size_t total =
            ledger + static_cast<std::size_t>(count) - 1U; // before the anchor
        const std::size_t before = std::min(context, total);
        std::array<std::int32_t, 32> window{};
        if (before + width > window.size()) { return; }
        std::size_t n = 0;
        for (std::size_t i = total - before; i < total; ++i) {
            window[n++] = i < ledger ? sequence.ledger[i] : licensed[i - ledger];
        }
        const TokenId anchor    = licensed[count - 1];
        window[n++]             = anchor;
        const std::int32_t next = std::clamp(mtp_host_egress->next_extents[row], 0,
                                             static_cast<std::int32_t>(draft_window));
        for (std::uint32_t j = 0; j < draft_window; ++j) {
            window[n++] = static_cast<std::int32_t>(j) < next
                              ? mtp_host_egress->next_drafts[j * max_concurrency + row]
                              : anchor;
        }
        qwen4_ple_gather->prefetch(std::span<const std::int32_t>(window.data(), n), before, width);
    }
}

void ProgramImpl::raise_qwen4_ple_staged() noexcept {
    for (std::uint32_t* word : qwen4_ple_staged) {
        if (word != nullptr) { ops::host::raise_host_flag(word); }
    }
}

void ProgramImpl::queue_qwen4_ple_rows(std::uint32_t lane, std::uint32_t frontier,
                                       std::span<const TokenId> tokens, std::size_t first_column) {
    if (tokens.size() > Qwen4PleStage{}.tokens.size()) {
        throw std::logic_error("Qwen4Exp PLE round window exceeds its staging");
    }
    Qwen4PleStage stage;
    stage.lane         = lane;
    stage.frontier     = frontier;
    stage.count        = static_cast<std::uint32_t>(tokens.size());
    stage.first_column = first_column;
    std::copy(tokens.begin(), tokens.end(), stage.tokens.begin());
    qwen4_ple_queue.push_back(stage);
}

void ProgramImpl::begin_qwen4_ple_round() {
    qwen4_ple_queue.clear();
    // No stream waits on the words now (the previous round has synchronized): a raise that no
    // round consumed (a failed round) must not release this one before its rows are written.
    for (std::uint32_t* word : qwen4_ple_staged) {
        if (word != nullptr) { ops::host::lower_host_flag(word); }
    }
}

void ProgramImpl::stage_qwen4_ple_round() {
    // Raised whatever happens: a launched round waits for it at its PLE layer (a failed gather
    // fails the round; the device must still be able to finish it).
    struct Raise {
        ProgramImpl* program;

        ~Raise() { program->raise_qwen4_ple_staged(); }
    } raise{this};

    for (const Qwen4PleStage& stage : qwen4_ple_queue) {
        stage_qwen4_ple_rows(active_sequence(stage.lane).ledger, stage.frontier,
                             std::span<const TokenId>(stage.tokens.data(), stage.count),
                             stage.first_column);
    }
    qwen4_ple_queue.clear();
}

void ProgramImpl::stage_qwen4_ple_rows(std::span<const TokenId> ledger, std::uint32_t frontier,
                                       std::span<const TokenId> tokens, std::size_t first_column) {
    nvtx::ScopedRange range(nvtx::Name::Qwen4PleGather, nvtx::Category::Control,
                            static_cast<std::uint64_t>(tokens.size()));
    if (!qwen4_ple_gather || qwen4_decode_ple_host == nullptr || tokens.empty() ||
        ledger.size() < frontier || !qwen4_runtime_record ||
        first_column + tokens.size() >
            static_cast<std::size_t>(qwen4_runtime_record->decode_ple_rows.ne[1])) {
        throw std::logic_error("Qwen4Exp PLE round staging is invalid");
    }
    // The n-gram rows of position p read the ngram_size-1 tokens before p; the window's own
    // tokens come after the committed ones. A shorter history is the sequence start.
    const std::size_t context = parameters.model.config().text.ple->ngram_size - 1U;
    const std::size_t before  = std::min<std::size_t>(context, frontier);
    std::array<std::int32_t, 32> window{};
    if (before + tokens.size() > window.size()) {
        throw std::logic_error("Qwen4Exp PLE round window exceeds its staging");
    }
    std::size_t count = 0;
    for (std::size_t i = frontier - before; i < frontier; ++i) { window[count++] = ledger[i]; }
    for (const TokenId token : tokens) { window[count++] = token; }
    const std::size_t width = qwen4_ple_gather->width();
    qwen4_ple_gather->gather(std::span<const std::int32_t>(window.data(), count), before,
                             tokens.size(),
                             std::span<std::uint16_t>(qwen4_decode_ple_host + first_column * width,
                                                      tokens.size() * width));
}

void ProgramImpl::fold_qwen4_ple(std::span<const ops::GdnReplayFoldRow> rows) {
    fold_qwen4_ple_rank(rows, 0);
    if (peer && qwen4_peer_runtime_record) {
        const ScopedCurrentDevice rank1(peer->device.device);
        fold_qwen4_ple_rank(rows, 1);
    }
}

void ProgramImpl::fold_qwen4_ple_rank(std::span<const ops::GdnReplayFoldRow> rows, int rank) {
    const std::optional<execution::Qwen4Runtime>& runtime =
        rank == 0 ? qwen4_runtime_record : qwen4_peer_runtime_record;
    if (!runtime || runtime->ple_record.data == nullptr) { return; }
    if (rows.empty() || rows.size() > kMaximumConcurrency) {
        throw std::logic_error("Qwen4Exp PLE fold rows are invalid");
    }
    WorkspaceArena& arena = rank == 0 ? work : peer->work;
    cudaStream_t stream   = rank == 0 ? device.stream : peer->device.stream;
    const auto count      = static_cast<std::int32_t>(rows.size());
    auto scope            = arena.scope();
    Tensor control        = arena.alloc(DType::I32, {count, 3});
    std::array<std::int32_t, 3 * kMaximumConcurrency> host{};
    for (std::int32_t row = 0; row < count; ++row) {
        host[static_cast<std::size_t>(row)]             = rows[row].source_state_slot;
        host[static_cast<std::size_t>(count + row)]     = rows[row].destination_state_slot;
        host[static_cast<std::size_t>(2 * count + row)] = rows[row].commit_columns;
    }
    // The transfer reads the pageable array before returning, so it may go out of scope.
    CUDA_CHECK(cudaMemcpyAsync(control.data, host.data(), control.bytes(), cudaMemcpyHostToDevice,
                               stream));
    const Tensor& record   = runtime->ple_record;
    const Tensor inputs    = record.slice(2, 0, count);
    Tensor states          = runtime->ple_states;
    const Tensor sources   = control.slice(1, 0, 1).view({count});
    const Tensor targets   = control.slice(1, 1, 1).view({count});
    const Tensor committed = control.slice(1, 2, 1).view({count});
    ops::ple_conv_advance(
        inputs, &committed,
        static_cast<std::int32_t>(parameters.model.config().text.ple->conv_history()), states,
        sources, targets, stream);
}

void ProgramImpl::force_qwen4_tokens(const SequenceState& sequence, TokenId* tokens,
                                     std::int32_t count) const {
    for (const std::vector<TokenId>& line : qwen4_forced_tokens) {
        if (line.size() <= sequence.ledger.size() ||
            !std::equal(sequence.ledger.begin(), sequence.ledger.end(), line.begin())) {
            continue;
        }
        for (std::int32_t j = 0; j < count && sequence.ledger.size() + j < line.size(); ++j) {
            tokens[j] = line[sequence.ledger.size() + static_cast<std::size_t>(j)];
        }
        return;
    }
}

void ProgramImpl::write_qwen4_top2(const Tensor& logits_column, std::uint32_t lane,
                                   std::uint32_t position, int source) {
    std::vector<std::uint16_t> host(static_cast<std::size_t>(logits_column.ne[0]));
    CUDA_CHECK(cudaMemcpy(host.data(), logits_column.data, host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    const auto value = [](std::uint16_t bits) {
        const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
        float out;
        std::memcpy(&out, &word, sizeof(out));
        return out;
    };
    const auto domain = static_cast<std::size_t>(parameters.model.resources().public_token_count);
    std::size_t best = 0, second = 1;
    if (value(host[second]) > value(host[best])) { std::swap(best, second); }
    for (std::size_t i = 2; i < domain; ++i) {
        const float v = value(host[i]);
        if (v > value(host[best])) {
            second = best;
            best   = i;
        } else if (v > value(host[second])) {
            second = i;
        }
    }
    std::fprintf(qwen4_top2_file, "%u %u %d %zu %.6g %zu %.6g\n", lane, position, source, best,
                 static_cast<double>(value(host[best])), second,
                 static_cast<double>(value(host[second])));
    std::fflush(qwen4_top2_file);
}

ProgramImpl::~ProgramImpl() noexcept {
    if (qwen4_top2_file != nullptr) { std::fclose(qwen4_top2_file); }
    if (qwen4_mtp_calib != nullptr) { std::fclose(qwen4_mtp_calib); }
    if (qwen4_replay_record != nullptr) { std::fclose(qwen4_replay_record); }
    if (qwen4_round_log != nullptr) { std::fclose(qwen4_round_log); }
    if (peer) {
        if (peer->device.transfer_stream != nullptr) {
            (void)cudaStreamSynchronize(peer->device.transfer_stream);
        }
        if (peer->device.stream != nullptr) { (void)cudaStreamSynchronize(peer->device.stream); }
    }
    if (device.transfer_stream != nullptr) { (void)cudaStreamSynchronize(device.transfer_stream); }
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
}

std::vector<float> ProgramImpl::causal_score(PreparedPromptData&& prompt,
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
    const std::uint32_t entitlement            = kv_pages_for_frontier(predictor_count);
    if (entitlement == 0) { throw std::logic_error("causal score has no KV entitlement"); }

    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> address;
    const auto cleanup = [&] {
        bool released = true;
        if (address) {
            if (text_kv_addresses->active(*address)) { text_kv_addresses->deactivate(*address); }
            released = text_kv_addresses->release(*address) && released;
            address.reset();
        }
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
    std::vector<std::int32_t> staged_positions;
    staged_positions.reserve(kCausalScoreTile);
    std::uint32_t staged_columns = 0;
    if (qwen4_tfdump) {
        qwen4_tfdump->write_tokens(
            std::span<const std::int32_t>(prompt.token_ids.data(), prompt.token_ids.size()));
    }

    try {
        state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::bad_alloc(); }
        address = text_kv_addresses->create_active(entitlement, 0, device.stream);
        if (!address) { throw std::bad_alloc(); }
        if (text_kv_addresses->bound_row(*address) != 0) {
            throw std::logic_error("causal score did not bind the unique Main KV row");
        }
        text_kv_addresses->ensure_mapped_to_tokens(*address, predictor_count, device.stream);

        const std::int32_t state_slot = state_store->physical_slot(*state);
        const auto flush              = [&] {
            if (staged_columns == 0) { return; }
            if (staged_columns != staged_targets.size() || staged_columns > kCausalScoreTile) {
                throw std::logic_error("causal score staging has an invalid shape");
            }
            work.reset();
            if (peer) { peer->work.reset(); }
            mark_workspace_usage(workspace_plan.causal_score);
            const auto columns = static_cast<std::int32_t>(staged_columns);
            Tensor logits      = work.alloc(
                DType::BF16, {dimension(parameters.model.config().text.vocab_size), columns});
            Tensor target_ids = work.alloc(DType::I32, {columns});
            Tensor logprobs   = work.alloc(DType::FP32, {columns});
            Tensor hidden     = score_hidden->slice(1, 0, columns);
            if (parameters.qwen4 && parameters.model.config().text.hyper_connection) {
                // The carried hidden is the wide residual: the Text output mixer (replicated, on
                // rank 0), then the head. (Without hyper-connections it is the final-normed one.)
                Tensor mixed = work.alloc(
                    DType::BF16, {dimension(parameters.model.config().text.hidden_size), columns});
                ops::hyper_connection_mix(hidden, parameters.qwen4->head.weights,
                                                       parameters.model.config().text.rms_norm_eps, mixed,
                                                       nullptr, work, device.stream);
                hidden = mixed;
            }
            if (peer) {
                project_score_tile_split(hidden, logits);
            } else {
                execution::project(hidden, parameters.text.output_head, logits, work,
                                                device.stream);
            }
            CUDA_CHECK(cudaMemcpyAsync(target_ids.data, staged_targets.data(), target_ids.bytes(),
                                                    cudaMemcpyHostToDevice, device.stream));
            ops::target_logprobs(logits, target_ids,
                                              dimension(parameters.model.resources().public_token_count),
                                              logprobs, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(score_logprobs_host->data(), logprobs.data, logprobs.bytes(),
                                                    cudaMemcpyDeviceToHost, device.stream));
            synchronize_devices();
            if (qwen4_tfdump && qwen4_tfdump->logits_enabled()) {
                std::vector<std::uint16_t> host(static_cast<std::size_t>(logits.numel()));
                CUDA_CHECK(
                    cudaMemcpy(host.data(), logits.data, logits.bytes(), cudaMemcpyDeviceToHost));
                qwen4_tfdump->write_logits(host, logits.ne[0], staged_positions);
            }
            staged_positions.clear();
            const auto* host = static_cast<const float*>(score_logprobs_host->data());
            output.insert(output.end(), host, host + staged_columns);
            staged_targets.clear();
            staged_columns = 0;
            work.reset();
            if (peer) { peer->work.reset(); }
        };

        std::uint32_t cursor = 0;
        while (cursor < predictor_count) {
            const std::uint32_t nominal = std::min(prefill_chunk, predictor_count - cursor);
            // At tp 2 the chunk runs the Text forward on both ranks; rank 1's KV row and StateImage
            // slot are rank 0's, replayed by the mirrors.
            execution::PrefillContext schedule_state{
                {device, parameters, work, state_images->linear(), nullptr, io, prefill_hidden,
                 prefill_chunk, proposal_head, tp_binding(), nullptr, qwen4_runtime()},
                decoder->text_kv.execution_view(text_kv_addresses->execution_row(*address)),
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
            text_kv_addresses->commit_frontier(*address, cursor);

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
                CUDA_CHECK(::ninfer::memcpy_device_async(destination.data, source.data,
                                                         source.bytes(), device.stream));
                for (std::uint32_t column = 0; column < count; ++column) {
                    staged_targets.push_back(prompt.token_ids[selected + column + 1U]);
                    staged_positions.push_back(static_cast<std::int32_t>(selected + column));
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
            synchronize_devices();
        } catch (...) {}
        work.reset();
        if (peer) { peer->work.reset(); }
        try {
            cleanup();
        } catch (...) {}
        throw;
    }
}

void ProgramImpl::start_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].start();
}

void ProgramImpl::stop_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].record_stop();
}

runtime::ContextTransferObservation ProgramImpl::context_transfer_observation(
    runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
    TransferWork work, std::uint32_t page_count, std::uint64_t state_images) const {
    const double elapsed_ns =
        static_cast<double>(
            context_transfer_timers_[context_resource_index(resource)].elapsed_ms()) *
        1'000'000.0;
    const std::uint64_t measured_ns =
        elapsed_ns >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(elapsed_ns + 0.5));
    return runtime::ContextTransferObservation{
        .resource  = resource,
        .direction = direction,
        .units =
            resource == runtime::ContextResourceClass::State ? state_images : work.payload_bytes,
        .page_count = page_count,
        .work       = work,
        .elapsed_ns = measured_ns,
    };
}

MemorySummary ProgramImpl::memory_summary() const noexcept {
    MemorySummary out;
    out.device          = device.device;
    out.max_context     = capacity;
    out.kv_capacity     = kv_capacity;
    out.kv_cache        = kv_storage;
    const auto& weights = parameters.model.storage_stats();
    // The summary describes rank 0's device; at tp 2 the model's total spans both ranks.
    const std::size_t weight_bytes =
        peer ? weights.per_device_capacity_bytes[0] : weights.device_capacity_bytes;
    out.weights = ArenaMemorySummary{weight_bytes, weight_bytes, weight_bytes};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    // Rank 0's workspace; at tp 2 its Vision plan is the receiver's unless rank 0 holds the tower.
    const VisionWorkspacePlan* vision_plan = workspace_plan.rank_vision(0, vision_rank);
    std::size_t active_workspace_bytes     = work.used();
    if (vision_plan != nullptr && active_handoff_bytes != 0) {
        active_workspace_bytes = std::max(active_workspace_bytes,
                                          vision_plan->handoff_offset_bytes + active_handoff_bytes);
    }
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), active_workspace_bytes,
                                       std::max(work.peak_used(), workspace_logical_peak_bytes)};
    if (vision_plan != nullptr) {
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(capacity, kMaximumPromptVisionTokens)),
            .max_item_tokens        = vision_plan->max_merged_tokens,
            .general_capacity_bytes = vision_plan->general_capacity_bytes,
            .encode_peak_bytes      = vision_plan->encode_peak_bytes,
            .handoff_offset_bytes   = vision_plan->handoff_offset_bytes,
            .handoff_capacity_bytes = vision_plan->handoff_capacity_bytes,
            .handoff_active_bytes   = active_handoff_bytes,
            .handoff_peak_bytes     = vision_handoff_peak_bytes,
        };
    }
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.cuda_graph_observed_bytes    = graph_observed_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    out.routed_expert_slots =
        qwen4_expert_slots +
        (qwen4_elastic ? static_cast<std::uint32_t>(qwen4_elastic->stats().columns) : 0U);
    out.routed_expert_pool_bytes = qwen4_expert_pool_bytes;
    out.prefill_chunk            = prefill_chunk;
    out.routed_expert_prompt_workspace_bytes =
        qwen4_prompt_base_chunk > 0 ? workspace_plan.qwen4_prompt : 0;
    if (host_state_images) {
        out.host_state_capacity_slots = host_state_images->capacity();
        out.host_state_occupied_slots = host_state_images->occupied();
    }
    if (host_kv_arena) {
        out.host_kv_capacity_bytes = host_kv_arena->capacity_bytes();
        out.host_kv_occupied_bytes = host_kv_arena->occupied_bytes();
    }
    return out;
}

void ProgramImpl::reset_memory_peaks() noexcept {
    persistent.reset_peak();
    work.reset_peak();
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    vision_handoff_peak_bytes              = active_handoff_bytes;
    workspace_logical_peak_bytes           = work.used();
    const VisionWorkspacePlan* vision_plan = workspace_plan.rank_vision(0, vision_rank);
    if (vision_plan != nullptr && active_handoff_bytes != 0) {
        workspace_logical_peak_bytes = std::max(
            workspace_logical_peak_bytes, vision_plan->handoff_offset_bytes + active_handoff_bytes);
    }
}


} // namespace ninfer::models::qwen3_5::detail

#include "models/qwen3_5/program/tp2/program_impl_tp2.inc"
