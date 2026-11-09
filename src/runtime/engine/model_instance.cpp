#include "runtime/engine/model_instance.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/startup.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"
#include "models/qwen3_5/program/program.h"
#include "ops/sparse_moe/host/host_moe_executor.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

// The Engine option checks at tp 2 (tp2/model_instance_tp2.inc).
void validate_tensor_parallel_options(const EngineOptions& options);

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        const std::string name = options.artifact_path.filename().string();
        if (const auto part = name.find(".ninfer.part-"); part != std::string::npos) {
            throw std::invalid_argument(name + " is a continuation of a split artifact: pass " +
                                        name.substr(0, part + 7) + " (it finds its parts)");
        }
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts, not " + name);
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if ((options.vision_device || options.max_vision_tokens) && !options.enable_vision) {
        throw std::invalid_argument("Engine vision_device and max_vision_tokens require Vision");
    }
    if (options.vision_device && std::find(options.devices.begin(), options.devices.end(),
                                           *options.vision_device) == options.devices.end()) {
        throw std::invalid_argument("Engine vision_device must be one of devices");
    }
    if (options.max_vision_tokens && (*options.max_vision_tokens < kMinimumMaxVisionTokens ||
                                      *options.max_vision_tokens > kMaximumMaxVisionTokens)) {
        throw std::invalid_argument("Engine max_vision_tokens must be in [" +
                                    std::to_string(kMinimumMaxVisionTokens) + "," +
                                    std::to_string(kMaximumMaxVisionTokens) + "]");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
    if (options.tp == 2) { validate_tensor_parallel_options(options); }
}

std::size_t free_device_bytes(int device) {
    int previous = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    CUDA_CHECK(cudaSetDevice(device));
    std::size_t free_bytes   = 0;
    std::size_t total_bytes  = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    CUDA_CHECK(cudaSetDevice(previous));
    CUDA_CHECK(status);
    return free_bytes;
}

// Each rank's free device bytes, indexed by rank; one rank at width 1.
std::vector<std::size_t> rank_free_bytes(const DeviceContext& primary,
                                         const ExecutionContext* execution) {
    std::vector<std::size_t> free_bytes{free_device_bytes(primary.device)};
    if (execution != nullptr) {
        for (int rank = 1; rank < execution->tp; ++rank) {
            free_bytes.push_back(
                free_device_bytes(execution->dev[static_cast<std::size_t>(rank)]->device));
        }
    }
    return free_bytes;
}

// The bottleneck rank's free bytes.
std::size_t free_runtime_bytes(const DeviceContext& primary, const ExecutionContext* execution) {
    const auto free_bytes = rank_free_bytes(primary, execution);
    return *std::min_element(free_bytes.begin(), free_bytes.end());
}

void synchronize_ranks(const DeviceContext& primary, const ExecutionContext* execution) {
    if (execution != nullptr) {
        for (int rank = 1; rank < execution->tp; ++rank) {
            execution->dev[static_cast<std::size_t>(rank)]->synchronize();
        }
    }
    primary.synchronize();
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.vision_device.reset();
        options.max_vision_tokens.reset();
        options.use_cuda_graph = false;
        options.context_cache  = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    if (options.tp != 1 && options.tp != 2) {
        throw std::invalid_argument("Engine tp must be 1 or 2");
    }
    if (options.devices.empty()) {
        if (options.tp != 1) {
            throw std::invalid_argument("Engine tp 2 requires one device id per rank in devices");
        }
        options.devices = {options.device};
    }
    if (options.devices.size() != static_cast<std::size_t>(options.tp) ||
        options.devices.front() != options.device) {
        throw std::invalid_argument(
            "Engine devices must list one id per tensor-parallel rank, starting with device");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    if (!cache.enabled || cache.disk_directory.empty() || cache.disk_capacity_bytes == 0) {
        cache.disk_directory.clear();
        cache.disk_capacity_bytes = 0;
    } else {
        // Disk tier validity: the artifact and the executable that computed the records.
        const auto identity = [](const std::filesystem::path& path) {
            std::error_code error;
            const std::filesystem::path real = std::filesystem::weakly_canonical(path, error);
            const auto size                  = std::filesystem::file_size(real, error);
            const auto mtime                 = std::filesystem::last_write_time(real, error);
            return real.string() + ":" + std::to_string(error ? 0 : size) + ":" +
                   std::to_string(mtime.time_since_epoch().count());
        };
        cache.disk_identity =
            "artifact=" + identity(options.artifact_path) + ";exe=" + identity("/proc/self/exe");
    }
    if (!cache.enabled) {
        if ((cache.device_state_slots && *cache.device_state_slots != 0) ||
            (cache.max_private_continuations && *cache.max_private_continuations != concurrency) ||
            (cache.max_shared_prefixes && *cache.max_shared_prefixes != 0) ||
            (cache.max_long_anchors_per_continuation &&
             *cache.max_long_anchors_per_continuation != 0)) {
            throw std::invalid_argument("disabled context cache accepts only root-only capacities");
        }
        cache.device_state_slots                = 0;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        return options;
    }

    // At tp 2 the Host tiers are off unless asked for (serve: --profile 27b, --host-state-slots),
    // and then every endpoint, rewrite and anchor checkpoint must fit in Device StateImages. An
    // in-prefill private capture (TurnClosure rewrite, long anchor) that finds the Device State
    // pool full first releases idle private continuations without a live session, least valuable
    // and oldest first, as for every Program with no Host StateImages; a capture that still does
    // not fit is skipped, and the next turn re-prefills from an earlier checkpoint or the root. The
    // default pool matches the private catalog, one checkpoint image per retained continuation;
    // each image costs one StateImage per rank (about 73 MiB for Qwen3.8 27B).
    const std::uint32_t minimum_private = options.tp == 2 ? 8U : 0U;
    const std::uint64_t default_device_states =
        options.tp == 2 ? std::max<std::uint64_t>(2ULL * concurrency, minimum_private)
                        : concurrency;
    cache.device_state_slots =
        cache.device_state_slots.value_or(static_cast<std::uint32_t>(default_device_states));
    const std::uint64_t default_private =
        std::max<std::uint64_t>(2ULL * concurrency, minimum_private);
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes = cache.max_shared_prefixes.value_or(
        std::max(concurrency, static_cast<std::uint32_t>(kMaximumExplicitPromptCacheMarkers)));
    cache.max_long_anchors_per_continuation = cache.max_long_anchors_per_continuation.value_or(2U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      peer_parameters(
          model->device_count() == 2
              ? std::make_unique<const models::qwen3_5::execution::Parameters>(*model, 1)
              : nullptr),
      frontend(models::qwen3_5::make_frontend(
          model->resources(),
          {.chat_template_path       = options.chat_template_path,
           .vision_enabled           = options.enable_vision,
           .max_context              = options.max_context,
           .media_cache_bytes        = options.media_cache_bytes,
           .media_live_bytes         = options.media_live_bytes,
           .media_preprocess_threads = options.media_preprocess_threads,
           .max_item_vision_tokens   = options.max_vision_tokens.value_or(kMaximumMaxVisionTokens),
           .grammar_cache_bytes      = options.constraint_limits.cache_bytes,
           .constraint_limits        = options.constraint_limits})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

namespace {

// `execution` is null at tensor-parallel width 1, where `device` is the only device.
ConstructedModel construct_model_on(const EngineOptions& requested, DeviceContext& device,
                                    ExecutionContext* execution) {
    validate_options(requested);
    // Adjusted below once the routed-expert storage is known (ggml block banks).
    EngineOptions options = requested;
    if ((execution != nullptr) != (options.tp == 2)) {
        throw std::logic_error("Engine execution context does not match the tensor-parallel width");
    }
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    // The Program page-locks active KV in RAM's Host planes after the weights, inside the same
    // page-locked limit.
    plan.reserve_host_pinned_bytes(
        models::qwen3_5::active_kv_host_pinned_bytes(plan.config(), options));
    binding.complete();
    auto model = execution != nullptr
                     ? models::qwen3_5::materialize_model(std::move(plan), *execution,
                                                          &options.startup_observer)
                     : models::qwen3_5::materialize_model(std::move(plan), device,
                                                          &options.startup_observer);
    synchronize_ranks(device, execution);
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    if ((execution != nullptr) != (instance->peer_parameters != nullptr)) {
        throw std::logic_error("loaded Model device count does not match the Engine tp");
    }
    frontend.complete();
    const bool gguf_banks = models::qwen3_5::gguf_routed_experts(instance->parameters);
    if (instance->parameters.qwen4 &&
        instance->parameters.model.config().text.qwen4_routed_experts()) {
        // CPU threads of the missing experts: an explicit value (--expert-cpu-threads) wins, 0 is
        // resolved from this machine (v0.7.2, issue #11).
        auto& e                = options.routed_experts;
        std::uint32_t& threads = gguf_banks ? e.gguf_cpu_threads : e.cpu_threads;
        std::string reason     = "set by the caller (--expert-cpu-threads)";
        if (threads == 0) {
            threads = static_cast<std::uint32_t>(ops::host::auto_expert_cpu_threads(
                ops::host::host_cpu_topology(), gguf_banks, &reason));
            reason  = "auto: " + reason;
        }
        std::fprintf(stderr, "routed experts: %u CPU threads for the missing experts (%s)\n",
                     threads, reason.c_str());
    }
    if (gguf_banks) {
        // capy-flash v0.7: ggml block expert banks run without prefill expert streaming, the
        // router-ahead prefetch and the miss split (not implemented for them): plan and run so.
        auto& e = options.routed_experts;
        if (e.stream_min_tokens > 0 || e.prefetch_slots > 0 || e.device_reads > 0 ||
            e.miss_split.mode != MissSplitMode::Off) {
            std::fprintf(stderr,
                         "routed experts: ggml block banks; prefill experts zero-copy from "
                         "%u tokens, prefetch and miss split off\n",
                         e.stream_min_tokens);
        }
        e.gguf_stream_min_tokens = e.stream_min_tokens; // the GGUF prefill ring
        e.cpu_threads            = e.gguf_cpu_threads;
        e.stream_min_tokens      = 0;
        e.prefetch_slots         = 0;
        e.device_reads           = 0;
        e.miss_split.mode        = MissSplitMode::Off;
    }
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    const auto make_planner = [&](const EngineOptions& planned) {
        return execution != nullptr
                   ? models::qwen3_5::make_sequence_planner(
                         instance->parameters, *instance->peer_parameters, device, planned)
                   : models::qwen3_5::make_sequence_planner(instance->parameters, device, planned);
    };
    // Every rank reserves the same per-rank layout and addresses the same KV pages, so one page
    // count is resolved against the tightest rank's own free memory. A rank that allocates less
    // than that layout (the rank without the Vision tower skips its encode workspace) is credited
    // with the difference, so only the tower's rank pays for it.
    const auto budgets_of = [&](const auto& planner) {
        auto budgets = rank_free_bytes(device, execution);
        for (std::size_t rank = 0; rank < budgets.size(); ++rank) {
            budgets[rank] += planner.unallocated_reservation_bytes(static_cast<int>(rank));
        }
        return budgets;
    };
    EngineOptions planned_options = options;
    // T64 elastic KV: the reservation's KV pool keeps max_context's addresses, but only its first
    // grow step is backed at startup; the rest of its bytes are address space only.
    std::size_t elastic_unbacked = 0;
    // Each elastic KV plane and expert plane may cost one 2 MiB granule more than its bytes.
    constexpr std::size_t kElasticGranule = std::size_t(2) << 20;
    if (options.elastic_kv.enabled && (options.routed_experts.slots_per_layer == 0) !=
                                          (options.routed_experts.elastic_columns == 0)) {
        throw std::invalid_argument("elastic_kv: give both slots_per_layer and elastic_columns, or "
                                    "neither (resolved from the device memory)");
    }
    if (const std::int64_t slot_bytes =
            models::qwen3_5::offloaded_expert_slot_bytes(instance->parameters, options);
        slot_bytes > 0 && options.elastic_kv.enabled) {
        auto probe = make_planner(planned_options);
        // The pool the Program gets: an explicit kv_capacity (above max_context with several
        // requests), else the curve's minimum.
        const runtime::SequenceCapacityCurve& curve = probe.capacity_curve();
        const std::uint32_t pages =
            options.kv_capacity.mode == KvCapacityMode::Explicit && curve.main_page_tokens != 0
                ? std::max(curve.minimum_main_page_groups,
                           static_cast<std::uint32_t>(
                               (static_cast<std::uint64_t>(options.kv_capacity.explicit_tokens) +
                                curve.main_page_tokens - 1U) /
                               curve.main_page_tokens))
                : probe.kv_pool_pages();
        const std::size_t kv_full = probe.kv_pool_bytes(pages);
        const std::uint32_t step  = (options.elastic_kv.grow_tokens + 63U) / 64U;
        const std::size_t kv_init = pages == 0 ? kv_full : kv_full / pages * std::min(step, pages);
        elastic_unbacked          = kv_full - kv_init;
        if (options.routed_experts.slots_per_layer == 0) {
            const auto budgets = budgets_of(probe);
            const auto sizing =
                resolve_kv_capacity_symmetric(options.kv_capacity, probe.capacity_curve(), budgets);
            const std::size_t budget = *std::min_element(budgets.begin(), budgets.end());
            const std::size_t slack  = (probe.kv_plane_count() + 16) * kElasticGranule;
            // The Program without KV and without slots, plus the headroom and the extra slots.
            const std::size_t fixed =
                sizing.runtime_reservation_bytes - kv_full +
                options.routed_experts.slot_headroom_bytes + slack +
                static_cast<std::size_t>(models::qwen3_5::offloaded_expert_staging_bytes(
                    instance->parameters, options)) +
                static_cast<std::size_t>(
                    models::qwen3_5::offloaded_expert_split_bytes(instance->parameters, options));
            const std::size_t kv_max = options.elastic_kv.kv_budget_bytes > 0
                                           ? std::min(kv_full, options.elastic_kv.kv_budget_bytes)
                                           : kv_full;
            const auto slots_with    = [&](std::size_t kv) -> std::size_t {
                return budget > fixed + kv
                              ? (budget - fixed - kv) / static_cast<std::size_t>(slot_bytes)
                              : 0;
            };
            const std::size_t total = std::min<std::size_t>(slots_with(kv_init), 512);
            // The elastic columns must cover the KV's growth from kv_init to kv_max (plus one
            // granule per KV plane): the arena maps only the base slots, those columns and
            // kv_init, so flooring both slot counts can leave up to one column short and a
            // request near max_context then fails with the device memory exhausted. The total
            // slot count is unchanged: only its split between base slots and elastic columns.
            const std::size_t growth  = kv_max > kv_init ? kv_max - kv_init : 0;
            const std::size_t needed  = growth == 0
                                            ? 0
                                            : (growth + probe.kv_plane_count() * kElasticGranule +
                                              static_cast<std::size_t>(slot_bytes) - 1) /
                                                 static_cast<std::size_t>(slot_bytes);
            const std::size_t covered = total > needed ? total - needed : 0;
            const std::size_t base    = std::min<std::size_t>({slots_with(kv_max), covered, 512});
            if (base == 0) {
                throw std::invalid_argument(
                    "elastic_kv: the KV of max_context (" + std::to_string(kv_max) +
                    " bytes) leaves no expert slot per layer (" + std::to_string(budget) +
                    " bytes free after weights); lower max_context or set kv_budget_bytes");
            }
            planned_options.routed_experts.slots_per_layer = static_cast<std::uint32_t>(base);
            planned_options.routed_experts.elastic_columns =
                static_cast<std::uint32_t>(total - base);
            std::fprintf(
                stderr,
                "elastic KV: %zu base + %zu elastic expert slots per layer, KV %zu bytes at "
                "max context (%zu backed at start, budget %zu); slot %lld bytes, %zu bytes for "
                "slots and KV, KV growth needs %zu columns\n",
                base, total - base, kv_full, kv_init, kv_max, static_cast<long long>(slot_bytes),
                budget > fixed ? budget - fixed : 0, needed);
        }
    } else if (slot_bytes > 0 && options.routed_experts.slots_per_layer == 0) {
        // Offloaded experts: the device memory the rest of the Program leaves, minus the
        // headroom, becomes expert slots (one slot per layer at a time).
        auto probe         = make_planner(planned_options);
        const auto budgets = budgets_of(probe);
        const auto sizing =
            resolve_kv_capacity_symmetric(options.kv_capacity, probe.capacity_curve(), budgets);
        const std::size_t budget = *std::min_element(budgets.begin(), budgets.end());
        // Each of the pool's four planes may cost one arena alignment more.
        constexpr std::size_t kPoolSlack = std::size_t(1) << 20;
        // The prefetch staging slots (T25) and the staged miss split's scratch slots (T18) come
        // out of the same budget.
        const std::size_t used =
            sizing.runtime_reservation_bytes + options.routed_experts.slot_headroom_bytes +
            kPoolSlack +
            static_cast<std::size_t>(
                models::qwen3_5::offloaded_expert_staging_bytes(instance->parameters, options)) +
            static_cast<std::size_t>(
                models::qwen3_5::offloaded_expert_split_bytes(instance->parameters, options));
        const std::size_t slots =
            budget > used ? (budget - used) / static_cast<std::size_t>(slot_bytes) : 0;
        if (slots == 0) {
            throw std::invalid_argument(
                "offloaded experts: no device memory left for one expert slot per layer (" +
                std::to_string(budget) + " bytes free after weights, " +
                std::to_string(sizing.runtime_reservation_bytes) + " bytes of Program, " +
                std::to_string(options.routed_experts.slot_headroom_bytes) + " bytes headroom)");
        }
        planned_options.routed_experts.slots_per_layer =
            static_cast<std::uint32_t>(std::min<std::size_t>(slots, 512));
    }
    auto planner      = make_planner(planned_options);
    auto rank_budgets = budgets_of(planner);
    for (auto& b : rank_budgets) { b += elastic_unbacked; }
    auto resolution =
        resolve_kv_capacity_symmetric(options.kv_capacity, planner.capacity_curve(), rank_budgets);
    auto sequence = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    instance->kv_capacity_resolution = resolution;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program =
        execution != nullptr
            ? models::qwen3_5::create_program(instance->parameters, *instance->peer_parameters,
                                              std::move(sequence), *execution,
                                              options.startup_observer)
            : models::qwen3_5::create_program(instance->parameters, std::move(sequence), device,
                                              options.startup_observer);
    synchronize_ranks(device, execution);
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes =
        free_runtime_bytes(device, execution);
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    // Every rank's resident formats: at tp 2 the Vision tower may sit on rank 1 alone and the
    // drafter and proposal head on rank 0 alone, so neither rank's views list them all.
    std::set<std::string> formats;
    for (int rank = 0; rank < instance->model->device_count(); ++rank) {
        for (const auto& weight : instance->model->weight_data(rank)) {
            for (const auto& part : weight.view.parts) {
                formats.emplace(artifact::format_name(part.parent->geometry.format));
            }
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds             = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds           = stats.upload_seconds;
    summary.artifact_bytes_read      = stats.read_bytes;
    summary.host_to_device_bytes     = stats.h2d_bytes;
    summary.peak_staging_bytes       = stats.peak_staging_bytes;
    summary.device_object_count      = stats.device_object_count;
    summary.host_object_count        = stats.host_object_count;
    summary.host_mapped_bytes        = stats.host_mapped_bytes;
    summary.host_pageable_bytes      = stats.host_pageable_bytes;
    summary.host_pageable_objects    = stats.host_pageable_objects;
    summary.host_mapped_objects      = stats.host_mapped_objects;
    summary.host_pinned_limit_bytes  = stats.host_pinned_limit_bytes;
    summary.host_pinned_alloc_failed = stats.host_pinned_alloc_failed;
    for (int i = 0; i < stats.device_count; ++i) {
        const auto rank = static_cast<std::size_t>(i);
        summary.devices.push_back(
            {.device         = execution != nullptr ? execution->dev[rank]->device : device.device,
             .capacity_bytes = stats.per_device_capacity_bytes[rank],
             .host_to_device_bytes = stats.per_device_h2d_bytes[rank],
             .sharded_bytes        = stats.sharded_bytes[rank],
             .replicated_bytes     = stats.replicated_bytes[rank],
             .local_bytes          = stats.local_bytes[rank],
             .host_mapped_bytes    = stats.per_device_host_mapped_bytes[rank]});
    }
    summary.context_cost          = std::move(context_cost.summary);
    summary.tp_transport          = instance->program->tp_transport().transport;
    summary.tp_mailbox_probe_ms   = instance->program->tp_transport().probe_ms;
    summary.tp_mailbox_fallback   = instance->program->tp_transport().fallback;
    summary.tp_mailbox_kernel     = instance->program->tp_transport().exchange_kernel;
    summary.tp_mailbox_slot_bytes = instance->program->tp_transport().slot_bytes;
    summary.tp_proposal_head      = instance->program->tp_transport().proposal_head;
    summary.tp_drafter            = instance->program->tp_transport().drafter;
    summary.tp_eager_transport    = instance->program->tp_transport().eager_transport;
    summary.tp_prefill_overlap    = instance->program->tp_transport().prefill_overlap;
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace

ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device) {
    if (options.tp != 1) {
        throw std::invalid_argument("single-device Engine construction requires tp 1");
    }
    return construct_model_on(options, device, nullptr);
}

ConstructedModel construct_model(const EngineOptions& options, ExecutionContext& execution) {
    if (execution.tp != options.tp || !execution.dev[0]) {
        throw std::invalid_argument("Engine execution context does not match options.tp");
    }
    return construct_model_on(options, execution.primary(),
                              execution.tp == 1 ? nullptr : &execution);
}

} // namespace ninfer::runtime

#include "runtime/engine/tp2/model_instance_tp2.inc"
