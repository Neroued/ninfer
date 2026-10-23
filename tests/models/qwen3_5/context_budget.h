#pragma once

#include "artifact/reader.h"
#include "core/device.h"
#include "models/qwen3_5/definition.h"
#include "models/qwen3_5/planning.h"
#include "ninfer/observability/snapshot.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::test::qwen3_5 {

[[nodiscard]] inline runtime::ContextCapacityPlan
context_capacity_plan(const EngineOptions& options) {
    artifact::Reader reader(options.artifact_path);
    const auto definition = models::qwen3_5::read_definition(reader, models::load_options(options));
    const auto profile    = models::qwen3_5::execution::describe_model(*definition);
    DeviceContext device(options.device);
    const auto planner = models::qwen3_5::make_sequence_planner(profile, device, options);
    return planner.capacity_plan();
}

// Optional checkpoints compete with each complete production working set, including its
// write/commit peak. Record packing, rather than payload-byte sums, determines the budget.
[[nodiscard]] inline std::size_t context_capacity_bytes(const runtime::ContextCapacityPlan& plan,
                                                        std::uint32_t retained_states = 0) {
    const auto geometry = describe_context_pool(plan.records);
    auto bytes          = plan.minimum_capacity().device_bytes;
    for (const auto& set : plan.working_sets) {
        auto records = set.records;
        records.push_back({geometry.spec_classes.at(0), retained_states});
        bytes = std::max(bytes, geometry.required_bytes(records));
    }
    return bytes;
}

inline void set_context_capacity(EngineOptions& options, std::uint32_t retained_states = 0) {
    options.context_capacity = ContextCapacityPolicy::explicit_capacity(
        context_capacity_bytes(context_capacity_plan(options), retained_states));
}

[[nodiscard]] inline std::size_t checkpoint_capacity_bytes(const runtime::ContextCapacityPlan& plan,
                                                           std::uint32_t checkpoints,
                                                           std::uint32_t tokens_per_checkpoint) {
    const auto geometry        = describe_context_pool(plan.records);
    constexpr auto page_tokens = static_cast<std::uint32_t>(kPagedKVPageSize);
    const auto pages           = (tokens_per_checkpoint + page_tokens - 1) / page_tokens;
    std::vector<RecordRequest> records{{geometry.spec_classes.at(0), checkpoints}};
    for (std::size_t index = 1; index < geometry.spec_classes.size(); ++index)
        records.push_back({geometry.spec_classes[index], checkpoints * pages});
    return geometry.required_bytes(records);
}

// Two private requests start together, then their growth exceeds the shared arena. The
// startup minimum must still support either complete request and its recovery by itself.
// Choose the starting length from the actual record packing, not a separate KV token quota.
[[nodiscard]] inline std::uint32_t configure_two_request_pressure(EngineOptions& options,
                                                                  std::uint32_t output_tokens) {
    if (options.max_concurrency != 2 || options.context_cache.enabled)
        throw std::invalid_argument("pressure fixture requires two uncached requests");
    for (options.max_context = 8192; options.max_context <= 65536; options.max_context *= 2) {
        const auto plan            = context_capacity_plan(options);
        const auto geometry        = describe_context_pool(plan.records);
        const auto capacity        = plan.minimum_capacity().device_bytes;
        constexpr auto page_tokens = static_cast<std::uint32_t>(kPagedKVPageSize);
        const auto records         = [&](std::uint32_t requests, std::uint32_t tokens) {
            const auto pages = (tokens + page_tokens - 1) / page_tokens;
            std::vector<RecordRequest> demand{{geometry.spec_classes.at(0), requests}};
            for (std::size_t index = 1; index < geometry.spec_classes.size(); ++index)
                demand.push_back({geometry.spec_classes[index], requests * pages});
            return demand;
        };
        std::uint32_t prompt = 128;
        for (; prompt + output_tokens < options.max_context; prompt += 128) {
            if (geometry.required_bytes(records(2, prompt + options.speculative.draft_tokens +
                                                       page_tokens + 1)) > capacity)
                break;
            if (geometry.required_bytes(records(
                    2, prompt + output_tokens - options.speculative.draft_tokens - 1)) <= capacity)
                continue;
            options.context_capacity = ContextCapacityPolicy::explicit_capacity(capacity);
            if (options.context_cache.host_capacity_bytes.value_or(0))
                options.context_cache.host_capacity_bytes =
                    std::max(options.context_cache.host_capacity_bytes.value(),
                             geometry.required_bytes(records(
                                 1, prompt + output_tokens + options.speculative.draft_tokens +
                                        page_tokens + 1)));
            return prompt;
        }
    }
    throw std::runtime_error("could not form a two-request Context pressure workload");
}

[[nodiscard]] inline ContextPoolUsage
context_usage(const RuntimeStats& stats, ContextMemoryDomain domain, std::uint32_t rank = 0) {
    for (const auto& pool : stats.context_pools)
        if (pool.domain == domain && pool.rank == rank) return pool;
    return {.rank = rank, .domain = domain};
}

[[nodiscard]] inline std::uint64_t occupied_bytes(const ContextPoolUsage& pool) {
    return pool.live_bytes + pool.reserved_bytes + pool.pending_retirement_bytes;
}

} // namespace ninfer::test::qwen3_5
