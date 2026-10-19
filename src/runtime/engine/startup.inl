#pragma once

#include "runtime/engine/instance.h"
#include "runtime/executor/local_executor.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/startup_memory.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "models/registry.h"
#include "core/startup.h"
#include "core/platform.h"

#include <algorithm>
#include <chrono>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <cerrno>
#include <system_error>
#include <set>

namespace ninfer::runtime::detail {


template <class Plan>
struct CalibratedPlan {
    Plan startup;
    GraphMemorySummary graph;
};

template <class Setup>
std::optional<CalibratedPlan<typename Setup::ExecutionPlan>>
calibrate_plan(const Setup& setup, RankModelInstance<typename Setup::Types>& instance,
               DeviceContext& device, const EngineOptions& options,
               const StartupMemoryBudget& budget, typename Setup::ExecutionPlan candidate,
               std::vector<typename Setup::Calibration>& measured, std::size_t minimum_rebuild = 0,
               std::size_t control_rebuild = 0) {
    for (;;) {
        auto probe_options          = options;
        probe_options.prefill_chunk = candidate.chunk;
        const auto key              = setup.graph_plan(probe_options, candidate.preparation);
        auto found                  = std::find_if(measured.begin(), measured.end(),
                                                   [&](const auto& item) { return item.plan == key; });
        if (found == measured.end()) {
            measured.push_back(
                setup.calibrate(instance.parameters, device, probe_options, candidate.preparation));
            found = std::prev(measured.end());
        }
        const auto graph_bytes = std::max(found->rebuild_bytes, minimum_rebuild);
        const GraphMemorySummary graph{
            .rebuild_estimate_bytes      = graph_bytes,
            .calibration_increment_bytes = found->preparation_increment_bytes,
            .calibration_retained_bytes  = found->retained_increment_bytes,
        };
        auto available = budget.available(device_memory_snapshot(device.device()));
        if (control_rebuild > available) return std::nullopt;
        available -= control_rebuild;
        if (graph_bytes > available) return std::nullopt;
        auto next = setup.plan(device, options, candidate.placement, available - graph_bytes);
        if (!next) return std::nullopt;
        const auto next_key = setup.graph_plan(probe_options, next->preparation);
        if (next_key == key) return CalibratedPlan{std::move(*next), graph};
        const bool revisited = std::any_of(measured.begin(), measured.end(),
                                           [&](const auto& item) { return item.plan == next_key; });
        if (revisited && candidate.sequence.device_reservation_bytes() <= available - graph_bytes) {
            // At a variant boundary choose the feasible measured side, then fill that interval.
            auto low  = candidate.sequence.device_reservation_bytes();
            auto high = available - graph_bytes;
            while (low < high) {
                const auto middle = low + (high - low + 1) / 2;
                auto trial        = setup.plan(device, options, candidate.placement, middle);
                if (trial && setup.graph_plan(probe_options, trial->preparation) == key) {
                    low       = middle;
                    candidate = std::move(*trial);
                } else
                    high = middle - 1;
            }
            return CalibratedPlan{std::move(candidate), graph};
        }
        candidate = std::move(*next);
    }
}

template <class Setup>
std::optional<CalibratedPlan<typename Setup::ExecutionPlan>>
prepare_program(const Setup& setup, RankModelInstance<typename Setup::Types>& instance,
                std::optional<typename Setup::Types::PreparedSession>& prepared_session,
                StartupMemorySummary& memory, DeviceContext& device, EngineOptions& options,
                const StartupMemoryBudget& budget,
                CalibratedPlan<typename Setup::ExecutionPlan> plan,
                std::vector<typename Setup::Calibration>& measured) {
    for (unsigned attempts = 0; attempts < 2; ++attempts) {
        auto& startup             = plan.startup;
        const auto explicit_bytes = startup.sequence.device_reservation_bytes();
        const auto host_bytes     = startup.sequence.host_capacity_bytes();
        const auto retained =
            budget.retained(device_memory_snapshot(device.device()),
                            instance.model->storage_stats().device_capacity_bytes);
        if (explicit_bytes != startup.capacity.runtime_reservation_bytes ||
            startup.sequence.kv_capacity() != startup.capacity.resolved_tokens)
            throw std::logic_error("resolved capacity differs from the finalized Program plan");
        std::optional<DeviceMemorySnapshot> ready;
        try {
            StartupPhaseScope phase(options.startup_observer, StartupPhase::ProgramInitialize);
            auto control_plan = plan_session(startup.sequence);
            instance.program = create_rank_program(instance.parameters, std::move(startup.sequence),
                                                   device, options.startup_observer);
            auto transfers   = instance.program->context_transfer_profiles();
            if (transfers.size() != control_plan.context.pools.size()) {
                throw std::logic_error("rank context pools differ from the session plan");
            }
            for (std::size_t i = 0; i < transfers.size(); ++i) {
                control_plan.context.pools[i].transfer = std::move(transfers[i]);
            }
            prepared_session.emplace(prepare_session(setup.definition(), std::move(control_plan)));
            device.synchronize();
            ready = device_memory_snapshot(device.device());
            if (budget.fits(*ready)) {
                options.prefill_chunk                                  = startup.chunk;
                options.context_cache.host_capacity_bytes              = host_bytes;
                memory.kv_capacity_resolution                          = startup.capacity;
                memory.kv_capacity_resolution.automatic_headroom_bytes = budget.headroom();
                memory.device_budget_bytes                             = budget.limit();
                memory.startup_retained_bytes                          = retained;
                memory.cuda_graph = instance.program->memory_summary().cuda_graph;
                if (memory.cuda_graph) {
                    plan.graph.preparation_increment_bytes =
                        memory.cuda_graph->preparation_increment_bytes;
                    memory.cuda_graph = plan.graph;
                }
                phase.complete();
                return plan;
            }
        } catch (const DeviceAllocationError&) {
            // A failed preparation invalidates its observation; the next probe measures it again.
            auto probe_options          = options;
            probe_options.prefill_chunk = startup.chunk;
            const auto key              = setup.graph_plan(probe_options, startup.preparation);
            std::erase_if(measured, [&](const auto& entry) { return entry.plan == key; });
        }
        prepared_session.reset();
        const auto after_control_cleanup = device_memory_snapshot(device.device());
        instance.program.reset();
        device.synchronize();
        if (attempts == 1) return std::nullopt;
        const auto after_cleanup = device_memory_snapshot(device.device());
        const auto available     = budget.available(after_cleanup);
        const auto recovered = ready ? device_memory_growth(after_cleanup, after_control_cleanup) -
                                           static_cast<std::int64_t>(explicit_bytes)
                                     : 0;
        const auto control_rebuild =
            ready ? static_cast<std::size_t>(std::max<std::int64_t>(
                        0, device_memory_growth(after_control_cleanup, *ready)))
                  : 0;
        const auto reserve =
            std::max(plan.graph.rebuild_estimate_bytes,
                     static_cast<std::size_t>(std::max<std::int64_t>(0, recovered)));
        if (reserve > available || control_rebuild > available - reserve) return std::nullopt;
        auto candidate =
            setup.plan(device, options, startup.placement, available - reserve - control_rebuild);
        if (!candidate) return std::nullopt;
        auto corrected = calibrate_plan(setup, instance, device, options, budget,
                                        std::move(*candidate), measured, reserve, control_rebuild);
        if (!corrected) return std::nullopt;
        plan = std::move(*corrected);
    }
    return std::nullopt;
}

template <class Types>
struct BuiltModel {
    std::unique_ptr<InferenceInstance<Types>> instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

template <class Setup>
std::optional<CalibratedPlan<typename Setup::ExecutionPlan>>
initialize_program(const Setup& setup, RankModelInstance<typename Setup::Types>& instance,
                   std::optional<typename Setup::Types::PreparedSession>& prepared_session,
                   StartupMemorySummary& memory, DeviceContext& device, EngineOptions& options,
                   const StartupMemoryBudget& budget, typename Setup::ExecutionPlan candidate) {
    std::vector<typename Setup::Calibration> measured;
    StartupPhaseScope phase(options.startup_observer, StartupPhase::TargetFinalize);
    auto calibrated =
        calibrate_plan(setup, instance, device, options, budget, std::move(candidate), measured);
    if (!calibrated) return {};
    phase.complete();
    return prepare_program(setup, instance, prepared_session, memory, device, options, budget,
                           std::move(*calibrated), measured);
}

template <class Setup>
BuiltModel<typename Setup::Types> build_local_model(const artifact::Reader& reader,
                                                    EngineOptions& options,
                                                    std::chrono::steady_clock::time_point start) {
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    std::unique_ptr<artifact::Reader> rank_reader;
    if (options.deployment && options.deployment->ranks.size() == 1) {
        const auto& node = options.deployment->nodes.front();
        for (const auto& [key, value] : node.environment)
            if (::setenv(key.c_str(), value.c_str(), 1))
                throw std::system_error(errno, std::generic_category(),
                                        "set local rank environment");
        rank_reader = std::make_unique<artifact::Reader>(node.artifact);
    }
    Setup setup(reader, rank_reader ? *rank_reader : reader, options, Device::cuda(options.device));
    std::unique_ptr<DeviceContext> device_owner;
    {
        StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
        device_owner = std::make_unique<DeviceContext>(options.device);
        phase.complete();
    }
    auto& device       = *device_owner;
    const auto initial = device_memory_snapshot(device.device());
    const auto limit   = device_budget(initial.free_bytes);
    const auto headroom =
        std::max(kDefaultKvCapacityHeadroomBytes, options.kv_capacity.automatic_headroom_bytes);
    const auto host_reservation = (NINFER_SHARED_PHYSICAL_MEMORY != 0) && options.enable_vision
                                      ? options.media_live_bytes
                                      : 0;
    StartupMemoryBudget budget(initial, limit, headroom, host_reservation);
    const auto choices   = setup.placements();
    std::size_t selected = 0;
    const auto choose    = [&](std::size_t available) {
        for (; selected < choices.size(); ++selected) {
            const auto& choice = choices[selected];
            if (choice.device_bytes > available) continue;
            auto plan =
                setup.plan(device, options, choice.placement, available - choice.device_bytes);
            if (plan) return plan;
        }
        return std::optional<typename Setup::ExecutionPlan>{};
    };
    auto candidate = choose(budget.available(device_memory_snapshot(device.device())));
    if (!candidate)
        throw std::invalid_argument(
            "Device budget cannot hold model weights and required execution resources");
    binding.complete();
    using Types = typename Setup::Types;
    std::unique_ptr<RankModelInstance<Types>> instance;
    std::optional<typename Types::PreparedSession> prepared_session;
    StartupMemorySummary memory;
    std::optional<CalibratedPlan<typename Setup::ExecutionPlan>> calibrated;
    std::size_t available_after_weights = 0;
    auto frontend                       = [&] {
        StartupPhaseScope phase(options.startup_observer, StartupPhase::FrontendInitialize);
        auto result = setup.frontend(options);
        phase.complete();
        return result;
    }();
    const auto& definition_stats = setup.definition()->resource_stats();
    LoadSummary summary;
    summary.artifact_bytes_read = definition_stats.read_bytes;
    for (;;) {
        try {
            auto model =
                setup.materialize(device, choices[selected].placement, options.startup_observer);
            device.synchronize();
            available_after_weights = device_memory_snapshot(device.device()).free_bytes;
            const auto& stats       = model->storage_stats();
            summary.upload_seconds += stats.upload_seconds;
            summary.artifact_bytes_read += stats.read_bytes;
            summary.host_to_device_bytes += stats.h2d_bytes;
            summary.peak_staging_bytes =
                std::max(summary.peak_staging_bytes, stats.peak_staging_bytes);
            instance = std::make_unique<RankModelInstance<Types>>(std::move(model), setup.profile);
            calibrated = initialize_program(setup, *instance, prepared_session, memory, device,
                                            options, budget, std::move(*candidate));
        } catch (const DeviceAllocationError&) {
            if (selected + 1 == choices.size()) throw;
            calibrated.reset();
        }
        if (calibrated) break;
        instance.reset();
        device.synchronize();
        if (++selected == choices.size())
            throw std::invalid_argument(
                "Device budget cannot hold the calibrated runtime and requested KV capacity");
        candidate = choose(budget.available(device_memory_snapshot(device.device())));
        if (!candidate)
            throw std::invalid_argument("Device budget cannot hold model execution resources");
    }
    memory.kv_capacity_resolution.available_after_weights_bytes = available_after_weights;
    memory.kv_capacity_resolution.available_after_startup_bytes =
        device_memory_snapshot(device.device()).free_bytes;
    const auto signature =
        setup.prefill_signature(calibrated->startup.preparation, options.prefill_chunk);
    auto context_cost = resolve_context_machine_cost(
        {.hardware_class = context_cost_hardware_class(
             device.properties().name, device.properties().major, device.properties().minor),
         .prefill_signature = signature},
        options.context_cost.preset_path);
    const auto& stats         = instance->model->storage_stats();
    summary.architecture      = models::architecture_name(setup.architecture());
    summary.model_name        = setup.definition()->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : setup.definition()->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    summary.device_object_count = stats.device_object_count;
    summary.host_object_count   = stats.host_object_count + definition_stats.host_object_count;
    summary.context_cost        = std::move(context_cost.summary);
    summary.cuda_sync_mode      = device.sync_mode();
    auto worker = std::make_unique<RankWorker<Types>>(std::move(device_owner), std::move(instance));
    auto executor = std::make_unique<LocalExecutor<Types>>(std::move(worker), next_session_epoch());
    auto inference = std::make_unique<InferenceInstance<Types>>(
        setup.definition(), std::move(frontend), std::move(executor), std::move(*prepared_session),
        std::move(memory));
    return {std::move(inference), std::move(summary), std::move(context_cost.model)};
}

} // namespace ninfer::runtime::detail
