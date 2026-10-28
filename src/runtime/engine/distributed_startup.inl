#pragma once

#include "runtime/engine/startup.inl"
#include "runtime/executor/distributed_executor.h"
#include "runtime/executor/preparation.h"

#include <iostream>
#include <set>

namespace ninfer::runtime::detail {

template <class Setup>
BuiltModel<typename Setup::Types>
build_distributed_model(const artifact::Reader& reader, EngineOptions& options,
                        std::chrono::steady_clock::time_point start) {
    using Types       = typename Setup::Types;
    using Distributed = DistributedExecutor<typename Types::ExecutionTypes, typename Setup::Codec>;
    StartupPhaseScope plan(options.startup_observer, StartupPhase::TargetPlan);
    Setup setup(reader, options);
    plan.complete();
    StartupPhaseScope frontend_phase(options.startup_observer, StartupPhase::FrontendInitialize);
    auto frontend = setup.frontend(options);
    frontend_phase.complete();
    auto group = std::make_unique<distributed::ExecutionGroup>(
        *options.deployment, setup.topology(), options.startup_observer,
        [](auto rank, std::string_view message) {
            std::cerr << "[rank " << rank << "] " << message << '\n';
        });
    std::optional<typename Types::PreparedSession> session;
    auto prepared = prepare_execution_group(*group, setup.requests(options),
                                            options.context_capacity, std::chrono::minutes(30),
                                            {[&](auto descriptions) {
                                                 setup.resolve(descriptions, options);
                                                 session.emplace(setup.session());
                                             },
                                             [&] { session.reset(); }});
    setup.resolve(prepared.layouts, options);
    auto codec = setup.codec();
    protocol::validate_control_capacity(codec.control_budget(), setup.capacity(), group->limits(),
                                        setup.topology().ranks.size());
    activate_execution_group(*group, prepared,
                             std::chrono::milliseconds(options.deployment->connect_timeout_ms));
    auto executor = std::make_unique<Distributed>(std::move(group), setup.capacity(),
                                                  std::move(codec), setup.observations());
    executor->bind_to_current_thread();
    StartupMemorySummary memory;
    memory.rank_budgets       = setup.memory_budgets();
    memory.context_capacities = prepared.capacities;
    for (std::size_t rank = 0; rank < memory.context_capacities.size(); ++rank) {
        auto& capacity                         = memory.context_capacities[rank];
        const auto& budget                     = memory.rank_budgets.at(rank);
        capacity.available_after_weights_bytes = prepared.reports.at(rank).free_after_weights_bytes;
        capacity.available_after_startup_bytes = budget.available_after_startup_bytes;
        capacity.automatic_headroom_bytes      = budget.headroom_bytes;
    }
    LoadSummary load;
    load.architecture        = models::architecture_name(setup.architecture());
    load.model_name          = setup.definition()->info().name;
    load.prefill_signature   = setup.prefill_signature(options.prefill_chunk);
    load.artifact_bytes_read = setup.definition()->resource_stats().read_bytes;
    load.host_object_count   = setup.definition()->resource_stats().host_object_count;
    std::string hardware;
    for (const auto& report : prepared.reports) {
        load.artifact_bytes_read += report.read_bytes;
        load.host_to_device_bytes += report.upload_bytes;
        load.upload_seconds         = std::max(load.upload_seconds, report.load_seconds);
        load.max_rank_staging_bytes = std::max(load.max_rank_staging_bytes, report.staging_bytes);
        load.device_object_count += report.device_objects;
        load.host_object_count += report.host_objects;
        if (!hardware.empty()) hardware += "+";
        hardware += report.hardware_class;
    }
    load.cuda_sync_mode = prepared.reports.front().cuda_sync_mode;
    auto cost           = resolve_context_machine_cost({hardware, load.prefill_signature},
                                                       options.context_cost.preset_path);
    load.context_cost   = std::move(cost.summary);
    std::set<std::string> formats;
    for (const auto& weight : setup.definition()->weight_data())
        for (const auto& part : weight.view.parts)
            formats.emplace(artifact::format_name(part.parent->geometry.format));
    load.weight_formats.assign(formats.begin(), formats.end());
    load.load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    auto instance = std::make_unique<InferenceInstance<Types>>(
        setup.definition(), std::move(frontend), std::move(executor), std::move(*session),
        std::move(memory));
    return {std::move(instance), std::move(load), std::move(cost.model)};
}

} // namespace ninfer::runtime::detail
