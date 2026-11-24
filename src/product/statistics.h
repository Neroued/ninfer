#pragma once

#include "ninfer/types.h"
#include <nlohmann/json.hpp>
#include <span>

namespace ninfer::product {

[[nodiscard]] double seconds(std::uint64_t nanoseconds) noexcept;
[[nodiscard]] double rate(std::uint64_t count, double elapsed_seconds) noexcept;

struct ExecutionWindow {
    std::uint64_t units = 0, rows = 0, input_tokens = 0, scheduled_positions = 0;
    double elapsed_seconds = 0;

    [[nodiscard]] std::optional<double> mean_seconds() const {
        if (!units) return {};
        return elapsed_seconds / units;
    }
};

[[nodiscard]] ExecutionWindow execution_window(const ExecutionCounters& before,
                                               const ExecutionCounters& after);

struct GenerationRates {
    double prefill_tokens_per_second = 0, output_tokens_per_second = 0;
};

[[nodiscard]] GenerationRates generation_rates(const RequestStats&, std::uint64_t output_tokens);
[[nodiscard]] nlohmann::json execution_json(const ExecutionExposure&);
[[nodiscard]] nlohmann::json execution_json(const ExecutionSample&);
[[nodiscard]] nlohmann::json request_execution_json(const RequestStats&);
[[nodiscard]] MemoryAllocations sum_allocations(std::span<const RankSnapshot>);
[[nodiscard]] nlohmann::json resource_json(const RankSnapshot&);
[[nodiscard]] const char* context_domain_name(ContextMemoryDomain) noexcept;
[[nodiscard]] nlohmann::json context_capacity_json(std::span<const ContextPoolCapacity>);
[[nodiscard]] nlohmann::json context_usage_json(const RuntimeStats&);
[[nodiscard]] nlohmann::json startup_resources_json(const EngineDescription&, const StartupReport&,
                                                    const EngineSnapshot&);

} // namespace ninfer::product
