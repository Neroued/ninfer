#pragma once
#include "runtime/engine/startup.inl"

namespace ninfer::runtime::detail {

// Native preparation owns weights, calibration and execution resources through activation.
// The same preparation contract serves local and distributed rank ownership.
template <class Setup>
BuiltModel<typename Setup::Types>
build_prepared_local_model(const artifact::Reader& reader, EngineOptions& options,
                           std::chrono::steady_clock::time_point start) {
    using Types = typename Setup::Types;
    std::unique_ptr<artifact::Reader> rank_reader;
    if (options.deployment && options.deployment->ranks.size() == 1) {
        const auto& node = options.deployment->nodes.front();
        for (const auto& [key, value] : node.environment)
            if (::setenv(key.c_str(), value.c_str(), 1))
                throw std::system_error(errno, std::generic_category(),
                                        "set local rank environment");
        rank_reader = std::make_unique<artifact::Reader>(node.artifact);
    }
    const auto& local_reader = rank_reader ? *rank_reader : reader;
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    Setup setup(reader, local_reader, options, Device::cuda(options.device));
    auto device = [&] {
        StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
        auto value = std::make_unique<DeviceContext>(options.device);
        phase.complete();
        return value;
    }();
    auto preparation       = setup.preparation(local_reader, options, *device);
    const auto requirement = preparation->memory();
    const auto initial     = device_memory_snapshot(device->device());
    StartupMemoryBudget budget(initial, device_budget(initial.available_bytes()),
                               options.memory_headroom_bytes, requirement.host_reservation_bytes);
    const auto available = budget.planning_available(initial);
    if (requirement.weight_bytes > available ||
        (initial.shared_physical_memory &&
         requirement.host_weight_bytes > available - requirement.weight_bytes) ||
        requirement.host_weight_bytes > budget.available_host(initial))
        throw DeviceAllocationError("model weights exceed local resource budget",
                                    requirement.weight_bytes);
    binding.complete();
    auto frontend = [&] {
        StartupPhaseScope phase(options.startup_observer, StartupPhase::FrontendInitialize);
        auto value = setup.frontend(options);
        phase.complete();
        return value;
    }();
    preparation->materialize();
    device->synchronize();
    const auto after_weights = device_memory_snapshot(device->device());
    if (preparation->report(budget.planning_available(after_weights)).preparation_required) {
        StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaGraphCalibration);
        preparation->prepare_probe();
        preparation->calibrate();
        preparation->prepare_rebuild_probe();
        preparation->calibrate_rebuild();
        phase.complete();
    }
    std::optional<typename Types::PreparedSession> session;
    ContextCapacityResolution capacity;
    std::size_t maximum_runtime = SIZE_MAX;
    auto description            = [&] {
        StartupPhaseScope phase(options.startup_observer, StartupPhase::ProgramInitialize);
        for (unsigned attempt = 0; attempt < 2; ++attempt) {
            const auto before  = device_memory_snapshot(device->device());
            const auto room    = budget.planning_available(before);
            const auto report  = preparation->report(room);
            const auto offered = std::min(report.capacity.available_bytes, maximum_runtime);
            capacity = resolve_context_capacity(options.context_capacity, report.capacity.plan,
                                                           offered, budget.available_host(before),
                                                           before.shared_physical_memory);
            std::optional<DeviceMemorySnapshot> ready;
            std::exception_ptr allocation_error;
            try {
                preparation->prepare({capacity.device_bytes}, budget.allocation_available(before),
                                                budget.available_host(before));
                preparation->prepare_execution();
                auto described = preparation->describe();
                session.emplace(
                    setup.session(described, options.context_cache.checkpoint_interval));
                device->synchronize();
                ready = device_memory_snapshot(device->device());
                if (budget.fits(*ready)) {
                    described.budget.limit_bytes            = budget.limit();
                    described.budget.headroom_bytes         = budget.headroom();
                    described.budget.host_reservation_bytes = budget.host_reservation();
                    described.budget.available_after_weights_bytes =
                        after_weights.available_bytes();
                    described.budget.available_after_startup_bytes = ready->available_bytes();
                    described.budget.retained_bytes =
                        budget.retained(before, requirement.weight_bytes);
                    described.budget.planned_slack_bytes   = capacity.planned_slack_bytes;
                    capacity.available_after_weights_bytes = after_weights.available_bytes();
                    capacity.available_after_startup_bytes = ready->available_bytes();
                    phase.complete();
                    return described;
                }
            } catch (const DeviceAllocationError&) { allocation_error = std::current_exception(); }
            std::string failure;
            if (ready) {
                const auto cost =
                    capacity.runtime_reservation_bytes +
                    (before.shared_physical_memory ? capacity.control_reservation_bytes : 0);
                maximum_runtime = budget.corrected_runtime_limit(*ready, cost);
                failure         = "prepared runtime exceeds startup budget by " +
                          std::to_string(budget.shortfall(*ready)) + " bytes";
            }
            session.reset();
            preparation->release_resources();
            device->synchronize();
            if (allocation_error) std::rethrow_exception(allocation_error);
            if (attempt == 1 || options.context_capacity.mode == ContextCapacityMode::Explicit)
                throw std::runtime_error(failure);
        }
        throw DeviceAllocationError("prepared runtime exceeds local resource budget");
    }();
    options.prefill_chunk                     = description.shape.prefill_chunk;
    options.context_cache.host_capacity_bytes = 0;
    auto instance                             = std::move(*preparation).finish();
    auto cost                                 = resolve_context_machine_cost(
        {context_cost_hardware_class(device->properties().name, device->properties().major,
                                                                     device->properties().minor),
                                         setup.prefill_signature(options.prefill_chunk)},
        options.context_cost.preset_path);
    const auto& weights    = instance->model->storage_stats();
    const auto& definition = setup.definition()->resource_stats();
    LoadSummary load;
    load.architecture           = models::architecture_name(setup.architecture());
    load.model_name             = setup.definition()->info().name;
    load.prefill_signature      = setup.prefill_signature(options.prefill_chunk);
    load.artifact_bytes_read    = weights.read_bytes + definition.read_bytes;
    load.host_to_device_bytes   = weights.h2d_bytes;
    load.upload_seconds         = weights.upload_seconds;
    load.max_rank_staging_bytes = weights.peak_staging_bytes;
    load.device_object_count    = weights.device_object_count;
    load.host_object_count      = weights.host_object_count + definition.host_object_count;
    load.cuda_sync_mode         = device->sync_mode();
    load.context_cost           = std::move(cost.summary);
    std::set<std::string> formats;
    for (const auto& weight : setup.definition()->weight_data())
        for (const auto& part : weight.view.parts)
            formats.emplace(artifact::format_name(part.parent->geometry.format));
    load.weight_formats.assign(formats.begin(), formats.end());
    load.load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    StartupMemorySummary memory;
    memory.context_capacities     = {capacity};
    memory.device_budget_bytes    = budget.limit();
    memory.startup_retained_bytes = description.budget.retained_bytes;
    memory.rank_budgets           = {description.budget};
    auto worker   = std::make_unique<RankWorker<Types>>(std::move(device), std::move(instance));
    auto executor = std::make_unique<LocalExecutor<Types>>(std::move(worker), next_session_epoch());
    auto inference = std::make_unique<InferenceInstance<Types>>(
        setup.definition(), std::move(frontend), std::move(executor), std::move(*session),
        std::move(memory));
    return {std::move(inference), std::move(load), std::move(cost.model)};
}

} // namespace ninfer::runtime::detail
