#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
}

artifact::LoadProgress artifact_progress(const LoadProgress& progress) {
    return artifact::LoadProgress{.callback           = progress.callback,
                                  .min_interval_bytes = progress.min_interval_bytes};
}

void validate_device_budget(std::uint64_t weight_bytes, std::size_t sequence_bytes) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const std::uint64_t required =
        checked_add(weight_bytes, static_cast<std::uint64_t>(sequence_bytes),
                    "combined weight and sequence memory requirement overflows u64");
    if (required > free_bytes) {
        throw std::invalid_argument(
            "model weights and requested context require " + std::to_string(required) +
            " bytes of device memory (weights " + std::to_string(weight_bytes) +
            ", sequence/workspace/graphs " + std::to_string(sequence_bytes) + "), but only " +
            std::to_string(free_bytes) + " bytes are free before loading weights");
    }
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, DeviceContext& device,
                                       artifact::Reader& reader, Clock::time_point load_start) {
    const bool fallback_enabled = options.allow_context_fallback;
    const std::uint32_t fallback_min = options.context_fallback_min;
    const std::uint32_t fallback_step = options.context_fallback_step;

    EngineOptions local = options;
    ConstructedTarget constructed;
    while (true) {
        if (local.max_context == 0) {
            throw std::invalid_argument(
                "context fallback exhausted; no usable context length fits in device memory");
        }

        artifact::Binder binder(reader);
        auto sequence_plan = Target::plan_sequence(device, local);
        auto load_plan = Target::plan_load(binder, local);
        try {
            validate_device_budget(load_plan.materialization().device_capacity_bytes,
                                   sequence_plan.device_reservation_bytes());
        } catch (const std::invalid_argument&) {
            if (!fallback_enabled || local.max_context <= fallback_min) {
                throw;
            }
            if (local.max_context > fallback_min) {
                local.max_context = local.max_context > fallback_step
                                        ? local.max_context - fallback_step
                                        : fallback_min;
            } else {
                break;
            }
            continue;
        }

        auto progress     = artifact_progress(local.load_progress);
        auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                                  progress.callback ? &progress : nullptr);
        const artifact::MaterializationStats stats = materialized.stats();

        auto model    = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
        auto loaded   = std::make_unique<Loaded>(std::move(model));
        auto instance = std::make_unique<Instance>(std::move(loaded), std::move(sequence_plan), device);

        constructed.load.target               = std::string(Target::target_key);
        constructed.load.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
        constructed.load.upload_seconds       = stats.upload_seconds;
        constructed.load.artifact_bytes_read  = stats.file_bytes;
        constructed.load.host_to_device_bytes = stats.h2d_bytes;
        constructed.load.peak_staging_bytes   = stats.peak_staging_bytes;
        constructed.load.tensor_count         = stats.tensor_count;
        constructed.load.resource_count       = stats.resource_count;
        constructed.load.effective_max_context = local.max_context;
        constructed.active = ActiveTarget(std::move(instance));
        return constructed;
    }
    throw std::invalid_argument(
        "context fallback exhausted after reaching minimum context=" + std::to_string(fallback_min));
}

} // namespace

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         DeviceContext& device)
    : loaded(std::move(stable_loaded)), request_memory(device), capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               DeviceContext& device)
    : loaded(std::move(stable_loaded)), request_memory(device), capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    if (reader.model_id() == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start);
    }
    if (reader.model_id() == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, device, reader, load_start);
    }
    throw std::runtime_error("artifact model_id '" + reader.model_id() +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
