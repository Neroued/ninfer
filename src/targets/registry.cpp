#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets {
namespace {

using Clock  = std::chrono::steady_clock;
using Target = Qwen3_6_27BRtx5090;

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

} // namespace

LoadedQwen3_6_27BRtx5090::LoadedQwen3_6_27BRtx5090(
    std::unique_ptr<Qwen3_6_27BRtx5090::LoadedModel> stable_model)
    : model(std::move(stable_model)), frontend(Qwen3_6_27BRtx5090::make_frontend(*model)) {}

LoadedQwen3_6_27BRtx5090::~LoadedQwen3_6_27BRtx5090() = default;

Qwen3_6_27BRtx5090Instance::Qwen3_6_27BRtx5090Instance(
    runtime::ProductCookie product_cookie, std::unique_ptr<LoadedQwen3_6_27BRtx5090> stable_loaded,
    Qwen3_6_27BRtx5090::SequencePlan sequence_plan, DeviceContext& device)
    : cookie(product_cookie), loaded(std::move(stable_loaded)), request_memory(device),
      capacity(sequence_plan.capacity()),
      program(
          Qwen3_6_27BRtx5090::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_27BRtx5090Instance::~Qwen3_6_27BRtx5090Instance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    if (reader.model_id() != Target::model_id) {
        throw std::runtime_error("artifact model_id '" + reader.model_id() +
                                 "' has no registered target for this device");
    }
    Target::preflight(device, options);

    artifact::Binder binder(reader);
    auto load_plan    = Target::plan_load(binder);
    auto progress     = artifact_progress(options.load_progress);
    auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                              progress.callback ? &progress : nullptr);
    const artifact::MaterializationStats stats = materialized.stats();

    const runtime::ProductCookie cookie = runtime::allocate_product_cookie();
    auto model  = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    auto loaded = std::make_unique<LoadedQwen3_6_27BRtx5090>(std::move(model));
    auto sequence_plan = Target::plan_sequence(*loaded->model, device, options);
    auto instance      = std::make_unique<Qwen3_6_27BRtx5090Instance>(cookie, std::move(loaded),
                                                                      std::move(sequence_plan), device);

    LoadSummary summary;
    summary.target               = "qwen3_6_27b_rtx5090";
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    return ConstructedTarget{.active = ActiveTarget(std::move(instance)),
                             .load   = std::move(summary)};
}

} // namespace ninfer::targets
