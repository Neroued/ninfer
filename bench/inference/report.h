#pragma once

#include "options.h"

#include <nlohmann/json.hpp>

namespace ninfer::bench {

struct Repetition {
    double wall_seconds = 0;
    EngineSnapshot before, after;
    std::vector<GenerationResult> requests;
};

// Execution durations are global batch observations; request exposures provide window maxima.
[[nodiscard]] nlohmann::json summarize(std::span<const Repetition> repetitions,
                                       std::uint32_t target_batch);
[[nodiscard]] nlohmann::json
make_report(const Options& options, const EngineDescription& description,
            const StartupReport& startup, const EngineSnapshot& initial,
            std::span<const std::uint64_t> seeds, std::span<const Repetition> repetitions);
[[nodiscard]] std::string format_report(const nlohmann::json& report, OutputFormat format);

} // namespace ninfer::bench
