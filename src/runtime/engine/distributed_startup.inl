#pragma once

#include "runtime/engine/startup.inl"
#include "runtime/executor/distributed_executor.h"
#include "runtime/executor/preparation.h"

#include <iostream>
#include <set>
#include <vector>

namespace ninfer::runtime::detail {

template <class Setup>
BuiltModel<typename Setup::Types>
build_distributed_model(const artifact::Reader& reader, EngineOptions& options,
                        std::chrono::steady_clock::time_point start) {
    using Types       = typename Setup::Types;
    using Distributed = DistributedExecutor<typename Types::ExecutionTypes, typename Setup::Codec>;
    auto& host_capacity = options.context_cache.host_capacity_bytes;
    if (host_capacity.value_or(0) > 0)
        throw std::invalid_argument("distributed multi-worker execution does not support Host Context cache");
    if (!host_capacity) host_capacity = 0;
    StartupPhaseScope plan(options.startup_observer, StartupPhase::TargetPlan);
    Setup setup(reader, options);
    plan.complete();
    StartupPhaseScope frontend_phase(options.startup_observer, StartupPhase::FrontendInitialize);
    auto frontend = setup.frontend(options);
    frontend_phase.complete();
    auto group = std::make_shared<distributed::ExecutionGroup>(
        *options.deployment, setup.topology(), options.startup_observer,
        [](auto rank, std::string_view message) {
            std::cerr << "[rank " << rank << "] " << message << '\n';
        });
    const bool want_narrow = setup.narrow_targets(options);
    const auto requests = setup.requests(options, want_narrow);
    const std::size_t frontend_host_bytes =
        (options.enable_vision ? options.media_live_bytes : 0) +
        (options.purpose == EnginePurpose::Generation ? options.grammar_cache_bytes : 0) +
        setup.coordinator_payload_bytes(options);
    std::vector<typename Types::PreparedSession> sessions;
    const auto requested = options;
    GroupPreparationOptions preparation_options{
        .memory_headroom_bytes = options.memory_headroom_bytes,
        .frontend_host_bytes   = frontend_host_bytes,
        .coordinator_node      = options.deployment->coordinator_node,
        .try_narrow_targets    = want_narrow,
        .purpose               = options.purpose};
    if constexpr (requires { setup.placement_names(); }) {
        preparation_options.placement_names  = setup.placement_names();
        preparation_options.maximize_context = setup.prefer_context_capacity(options);
    }
    auto prepared = prepare_execution_group(*group, requests, options.context_capacity,
                                            preparation_options, std::chrono::minutes(30),
                                            {[&](auto descriptions) {
                                                 options = requested;
                                                 setup.resolve(descriptions, options);
                                                 sessions = setup.sessions();
                                             },
                                             [&] {
                                                 sessions.clear();
                                                 options = requested;
                                             }});
    const std::uint32_t dp_count =
        static_cast<std::uint32_t>(group->topology().replicas.size());
    if (sessions.size() != dp_count) {
        throw std::logic_error("distributed startup prepared session count does not match dp");
    }
    const std::uint32_t local_lanes =
        (options.max_concurrency + dp_count - 1U) / dp_count;

    activate_execution_group(*group, prepared,
                             std::chrono::milliseconds(options.deployment->connect_timeout_ms));
    group->clear_startup_observer();

    StartupMemorySummary cluster_memory;
    cluster_memory.rank_budgets       = setup.memory_budgets();
    cluster_memory.context_capacities = prepared.capacities;
    auto all_observations = setup.observations();
    for (std::size_t rank = 0; rank < cluster_memory.context_capacities.size(); ++rank) {
        auto& capacity     = cluster_memory.context_capacities[rank];
        auto& budget       = cluster_memory.rank_budgets.at(rank);
        const auto& sample = prepared.memory_samples.at(rank);
        all_observations.at(rank).physical_memory = sample;
        budget.available_after_startup_bytes =
            sample.shared_physical_memory ? std::min(sample.cuda_total_bytes,
                                                    sample.system_available_bytes)
                                          : sample.cuda_free_bytes;
        capacity.available_after_weights_bytes = prepared.reports.at(rank).free_after_weights_bytes;
        capacity.available_after_startup_bytes = budget.available_after_startup_bytes;
    }

    ResolvedContextMachineCost first_cost;
    std::vector<BuiltReplica<Types>> built_replicas;
    built_replicas.reserve(dp_count);
    for (std::uint32_t r = 0; r < dp_count; ++r) {
        const auto& rep_topo = group->topology().replica(r);
        // Each executor reserves runtime control capacity against its own replica transport.
        std::unique_ptr<Executor<typename Types::ExecutionTypes>> executor = std::make_unique<Distributed>(
            group->take_replica(r), setup.capacity(r), setup.codec(r), local_lanes, all_observations);
        if constexpr (requires { setup.coordinate(r, std::move(executor)); })
            executor = setup.coordinate(r, std::move(executor));
        executor->bind_to_current_thread();

        StartupMemorySummary rep_memory;
        std::string hardware;
        for (const auto global_rank : rep_topo.ranks) {
            rep_memory.context_capacities.push_back(cluster_memory.context_capacities.at(global_rank));
            rep_memory.rank_budgets.push_back(cluster_memory.rank_budgets.at(global_rank));
            if (!hardware.empty()) hardware += "+";
            hardware += prepared.reports.at(global_rank).hardware_class;
        }
        auto rep_cost = resolve_context_machine_cost(
            {std::move(hardware), setup.prefill_signature(options.prefill_chunk, r)},
            options.context_cost.preset_path);
        if (r == 0) first_cost = rep_cost;
        built_replicas.push_back(BuiltReplica<Types>{
            .replica_id       = r,
            .executor         = std::move(executor),
            .prepared_session = std::move(sessions[r]),
            .memory           = std::move(rep_memory),
            .context_cost     = std::move(rep_cost.model),
        });
    }

    LoadSummary load;
    load.architecture        = models::architecture_name(setup.architecture());
    load.model_name          = setup.definition()->info().name;
    load.prefill_signature   = first_cost.summary.prefill_signature;
    load.artifact_bytes_read = setup.definition()->resource_stats().read_bytes;
    load.host_object_count   = setup.definition()->resource_stats().host_object_count;
    for (const auto& report : prepared.reports) {
        load.artifact_bytes_read += report.read_bytes;
        load.host_to_device_bytes += report.upload_bytes;
        load.upload_seconds         = std::max(load.upload_seconds, report.load_seconds);
        load.max_rank_staging_bytes = std::max(load.max_rank_staging_bytes, report.staging_bytes);
        load.device_object_count += report.device_objects;
        load.host_object_count += report.host_objects;
    }
    load.cuda_sync_mode = prepared.reports.front().cuda_sync_mode;
    load.context_cost   = std::move(first_cost.summary);
    std::set<std::string> formats;
    for (const auto& weight : setup.definition()->weight_data())
        for (const auto& part : weight.view.parts)
            formats.emplace(artifact::format_name(part.parent->geometry.format));
    load.weight_formats.assign(formats.begin(), formats.end());
    load.load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    auto instance = std::make_unique<InferenceInstance<Types>>(
        setup.definition(), std::move(frontend), std::move(built_replicas),
        std::move(cluster_memory), std::move(group));
    return {std::move(instance), std::move(load), std::move(first_cost.model)};
}

} // namespace ninfer::runtime::detail
