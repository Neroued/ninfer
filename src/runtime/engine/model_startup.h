#pragma once

#include "runtime/engine/model_instance.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/startup_memory.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "models/registry.h"
#include "core/startup.h"

#include <algorithm>
#include <chrono>
#include <set>

namespace ninfer::runtime::detail {

template <class Plan>
struct CalibratedPlan {
    Plan startup;
    GraphMemorySummary graph;
};

template <class Setup>
std::optional<CalibratedPlan<typename Setup::ExecutionPlan>>
calibrate_plan(const Setup& setup, ModelInstance<typename Setup::Types>& instance,
               DeviceContext& device, const EngineOptions& options,
               const StartupMemoryBudget& budget, typename Setup::ExecutionPlan candidate,
               std::vector<typename Setup::Calibration>& measured,
               std::size_t minimum_rebuild = 0) {
    // Explicit template arguments: MSVC fails to deduce aggregate CTAD here
    // (CalibratedPlan{plan, graph}) inside this member-template context.
    using ExecutionPlan = typename Setup::ExecutionPlan;
    for (;;) {
        auto probe_options          = options;
        probe_options.prefill_chunk = candidate.chunk;
        const auto key              = setup.graph_plan(probe_options, candidate.moe.get());
        auto found                  = std::find_if(measured.begin(), measured.end(),
                                                   [&](const auto& item) { return item.plan == key; });
        if (found == measured.end()) {
            measured.push_back(
                setup.calibrate(instance.parameters, device, probe_options, candidate.moe));
            found = std::prev(measured.end());
        }
        const auto graph_bytes = std::max(found->rebuild_bytes, minimum_rebuild);
        const GraphMemorySummary graph{
            .rebuild_estimate_bytes      = graph_bytes,
            .calibration_increment_bytes = found->preparation_increment_bytes,
            .calibration_retained_bytes  = found->retained_increment_bytes,
        };
        const auto available = budget.available(device_memory_snapshot(device.device()));
        if (graph_bytes > available) return std::nullopt;
        auto next = setup.plan(device, options, candidate.host_experts, available - graph_bytes);
        if (!next) return std::nullopt;
        const auto next_key = setup.graph_plan(probe_options, next->moe.get());
        if (next_key == key) return CalibratedPlan<ExecutionPlan>{std::move(*next), graph};
        const bool revisited = std::any_of(measured.begin(), measured.end(),
                                           [&](const auto& item) { return item.plan == next_key; });
        if (revisited && candidate.sequence.device_reservation_bytes() <= available - graph_bytes) {
            // At a variant boundary choose the feasible measured side, then fill that interval.
            auto low  = candidate.sequence.device_reservation_bytes();
            auto high = available - graph_bytes;
            while (low < high) {
                const auto middle = low + (high - low + 1) / 2;
                auto trial        = setup.plan(device, options, candidate.host_experts, middle);
                if (trial && setup.graph_plan(probe_options, trial->moe.get()) == key) {
                    low       = middle;
                    candidate = std::move(*trial);
                } else
                    high = middle - 1;
            }
            return CalibratedPlan<ExecutionPlan>{std::move(candidate), graph};
        }
        candidate = std::move(*next);
    }
}

template <class Setup>
std::optional<CalibratedPlan<typename Setup::ExecutionPlan>>
prepare_program(const Setup& setup, ModelInstance<typename Setup::Types>& instance,
                DeviceContext& device, EngineOptions& options, const StartupMemoryBudget& budget,
                CalibratedPlan<typename Setup::ExecutionPlan> plan,
                std::vector<typename Setup::Calibration>& measured, unsigned& attempts) {
    while (attempts < 2) {
        ++attempts;
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
            instance.program = create_program(instance.parameters, std::move(startup.sequence),
                                              device, options.startup_observer);
            device.synchronize();
            ready = device_memory_snapshot(device.device());
            if (budget.fits(*ready)) {
                options.prefill_chunk                                    = startup.chunk;
                options.context_cache.host_capacity_bytes                = host_bytes;
                instance.kv_capacity_resolution                          = startup.capacity;
                instance.kv_capacity_resolution.automatic_headroom_bytes = budget.headroom();
                instance.device_budget_bytes                             = budget.limit();
                instance.startup_retained_bytes                          = retained;
                instance.cuda_graph = instance.program->memory_summary().cuda_graph;
                if (instance.cuda_graph) {
                    plan.graph.preparation_increment_bytes =
                        instance.cuda_graph->preparation_increment_bytes;
                    instance.cuda_graph = plan.graph;
                }
                phase.complete();
                return plan;
            }
        } catch (const DeviceAllocationError&) {
            // A failed preparation invalidates its observation; the next probe measures it again.
            auto probe_options          = options;
            probe_options.prefill_chunk = startup.chunk;
            const auto key              = setup.graph_plan(probe_options, startup.moe.get());
            std::erase_if(measured, [&](const auto& entry) { return entry.plan == key; });
        }
        instance.program.reset();
        device.synchronize();
        if (attempts == 2) return std::nullopt;
        const auto after_cleanup = device_memory_snapshot(device.device());
        const auto available     = budget.available(after_cleanup);
        const auto recovered     = ready ? device_memory_growth(after_cleanup, *ready) -
                                           static_cast<std::int64_t>(explicit_bytes)
                                         : 0;
        const auto reserve =
            std::max(plan.graph.rebuild_estimate_bytes,
                     static_cast<std::size_t>(std::max<std::int64_t>(0, recovered)));
        if (reserve > available) return std::nullopt;
        auto candidate = setup.plan(device, options, startup.host_experts, available - reserve);
        if (!candidate) return std::nullopt;
        auto corrected = calibrate_plan(setup, instance, device, options, budget,
                                        std::move(*candidate), measured, reserve);
        if (!corrected) return std::nullopt;
        plan = std::move(*corrected);
    }
    return std::nullopt;
}

template <class Types>
struct BuiltModel {
    std::unique_ptr<ModelInstance<Types>> instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

template <class Setup>
std::optional<CalibratedPlan<typename Setup::ExecutionPlan>>
initialize_program(const Setup& setup, ModelInstance<typename Setup::Types>& instance,
                   DeviceContext& device, EngineOptions& options, const StartupMemoryBudget& budget,
                   typename Setup::ExecutionPlan candidate, unsigned& attempts) {
    if constexpr (Setup::supports_graph) {
        std::vector<typename Setup::Calibration> measured;
        StartupPhaseScope phase(options.startup_observer, StartupPhase::TargetFinalize);
        auto calibrated = calibrate_plan(setup, instance, device, options, budget,
                                         std::move(candidate), measured);
        if (!calibrated) return {};
        phase.complete();
        return prepare_program(setup, instance, device, options, budget, std::move(*calibrated),
                               measured, attempts);
    } else {
        std::size_t reserve = 0;
        while (attempts < 2) {
            ++attempts;
            const auto before    = device_memory_snapshot(device.device());
            const auto available = budget.available(before);
            if (reserve > available) return {};
            auto plan = setup.plan(device, options, candidate.host_experts, available - reserve);
            if (!plan) return {};
            const auto explicit_bytes = plan->sequence.device_reservation_bytes();
            const auto host_bytes     = plan->sequence.host_capacity_bytes();
            if (explicit_bytes != plan->capacity.runtime_reservation_bytes ||
                plan->sequence.kv_capacity() != plan->capacity.resolved_tokens)
                throw std::logic_error("eager plan differs from its resolved capacity");
            std::optional<DeviceMemorySnapshot> ready;
            try {
                StartupPhaseScope phase(options.startup_observer, StartupPhase::ProgramInitialize);
                instance.program = create_program(instance.parameters, std::move(plan->sequence),
                                                  device, options.startup_observer);
                device.synchronize();
                ready = device_memory_snapshot(device.device());
                if (budget.fits(*ready)) {
                    options.prefill_chunk                                    = plan->chunk;
                    options.context_cache.host_capacity_bytes                = host_bytes;
                    instance.kv_capacity_resolution                          = plan->capacity;
                    instance.kv_capacity_resolution.automatic_headroom_bytes = budget.headroom();
                    instance.device_budget_bytes                             = budget.limit();
                    instance.startup_retained_bytes                          = budget.retained(
                        before, instance.model->storage_stats().device_capacity_bytes);
                    phase.complete();
                    return CalibratedPlan<typename Setup::ExecutionPlan>{std::move(*plan), {}};
                }
            } catch (const DeviceAllocationError&) {}
            instance.program.reset();
            device.synchronize();
            if (ready) {
                const auto after = device_memory_snapshot(device.device());
                reserve = std::max(reserve, static_cast<std::size_t>(std::max<std::int64_t>(
                                                0, device_memory_growth(after, *ready) -
                                                       static_cast<std::int64_t>(explicit_bytes))));
            }
        }
        return {};
    }
}

template <class Setup>
BuiltModel<typename Setup::Types>
build_model(const artifact::Reader& reader, EngineOptions& options, DeviceContext& device,
            const StartupMemoryBudget& budget, std::chrono::steady_clock::time_point start) {
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    Setup setup(reader, options, device.device());
    const auto& weights          = setup.weights;
    const auto initial_available = budget.available(device_memory_snapshot(device.device()));
    auto candidate =
        weights.resident_bytes && *weights.resident_bytes <= initial_available
            ? setup.plan(device, options, false, initial_available - *weights.resident_bytes)
            : std::nullopt;
    bool host_experts = !candidate.has_value();
    if (host_experts && weights.offload && weights.fixed_bytes <= initial_available)
        candidate = setup.plan(device, options, true, initial_available - weights.fixed_bytes);
    if (!candidate)
        throw std::invalid_argument(
            "Device budget cannot hold model weights and required execution resources");
    binding.complete();
    std::unique_ptr<ModelInstance<typename Setup::Types>> instance;
    std::optional<CalibratedPlan<typename Setup::ExecutionPlan>> calibrated;
    unsigned preparation_attempts       = 0;
    std::size_t available_after_weights = 0;
    LoadSummary summary;
    for (;;) {
        auto model = setup.materialize(device, host_experts, options.startup_observer);
        device.synchronize();
        available_after_weights = device_memory_snapshot(device.device()).free_bytes;
        const auto& stats       = model->storage_stats();
        summary.upload_seconds += stats.upload_seconds;
        summary.artifact_bytes_read += stats.read_bytes;
        summary.host_to_device_bytes += stats.h2d_bytes;
        summary.peak_staging_bytes = std::max(summary.peak_staging_bytes, stats.peak_staging_bytes);
        {
            StartupPhaseScope phase(options.startup_observer, StartupPhase::FrontendInitialize);
            auto frontend = setup.frontend(*model, options);
            instance      = std::make_unique<ModelInstance<typename Setup::Types>>(
                std::move(model), std::move(frontend), setup.profile);
            phase.complete();
        }
        try {
            calibrated = initialize_program(setup, *instance, device, options, budget,
                                            std::move(*candidate), preparation_attempts);
        } catch (const DeviceAllocationError&) {
            if (host_experts || !weights.offload) throw;
            calibrated.reset();
        }
        if (calibrated) break;
        instance.reset();
        device.synchronize();
        if (host_experts || !weights.offload || preparation_attempts == 2)
            throw std::invalid_argument(
                "Device budget cannot hold the calibrated runtime and requested KV capacity");
        host_experts         = true;
        const auto available = budget.available(device_memory_snapshot(device.device()));
        if (weights.fixed_bytes > available)
            throw std::invalid_argument("Device budget cannot hold fixed model weights");
        candidate = setup.plan(device, options, true, available - weights.fixed_bytes);
        if (!candidate)
            throw std::invalid_argument("Device budget cannot hold MoE execution resources");
    }
    instance->kv_capacity_resolution.available_after_weights_bytes = available_after_weights;
    instance->kv_capacity_resolution.available_after_startup_bytes =
        device_memory_snapshot(device.device()).free_bytes;
    const auto signature =
        prefill_signature(*instance->model, calibrated->startup.moe.get(), options.prefill_chunk);
    auto context_cost = resolve_context_machine_cost(
        {.hardware_class = context_cost_hardware_class(
             device.properties().name, device.properties().major, device.properties().minor),
         .prefill_signature = signature},
        options.context_cost.preset_path);
    const auto& stats         = instance->model->storage_stats();
    summary.architecture      = models::architecture_name(setup.architecture());
    summary.model_name        = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    summary.device_object_count = stats.device_object_count;
    summary.host_object_count   = stats.host_object_count;
    summary.context_cost        = std::move(context_cost.summary);
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace ninfer::runtime::detail
