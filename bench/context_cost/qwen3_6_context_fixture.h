#pragma once

#include "context_cost_measure.h"

#include "core/transfer_work.h"
#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ninfer::bench::context_cost {

enum class TransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    DeviceToDevice,
};

struct ArtifactProfile {
    std::filesystem::path path;
    std::string model_id;
    std::string weights_id;
    std::string target_key;
};

struct MeasurementOptions {
    std::filesystem::path artifact;
    std::filesystem::path corpus;
    int device                  = 0;
    std::uint32_t max_context   = 8192;
    std::uint32_t prefill_chunk = 1024;
    int transfer_warmup         = 2;
    int transfer_repetitions    = 9;
    int prefill_repetitions     = 5;
};

struct TransferMeasurement {
    std::string label;
    TransferDirection direction = TransferDirection::DeviceToHost;
    TransferWork work;
    std::uint32_t page_count      = 0;
    std::uint32_t contiguous_runs = 0;
    bool validation               = false;
    SampleSummary timing;
};

struct PrefillMeasurement {
    std::string label;
    std::uint32_t prefix_tokens   = 0;
    std::uint32_t suffix_tokens   = 0;
    std::uint32_t chunks          = 0;
    std::uint64_t attention_pairs = 0;
    std::uint64_t vision_items    = 0;
    std::uint64_t vision_patches  = 0;
    bool validation               = false;
    SampleSummary timing;
};

struct TransferSuiteResult {
    std::vector<TransferMeasurement> measurements;
};

struct PrefillSuiteResult {
    std::vector<PrefillMeasurement> measurements;
    LoadSummary load;
};

[[nodiscard]] ArtifactProfile inspect_artifact(const std::filesystem::path& artifact);

[[nodiscard]] TransferSuiteResult measure_context_transfers(const MeasurementOptions& options);

[[nodiscard]] PrefillSuiteResult measure_prefill(const ArtifactProfile& artifact,
                                                 const MeasurementOptions& options);

[[nodiscard]] const char* transfer_direction_name(TransferDirection direction) noexcept;

} // namespace ninfer::bench::context_cost
