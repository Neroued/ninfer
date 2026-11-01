#pragma once

// Single-device (tp 1) tests on NINFER_TEST_ARTIFACT. The suite's 27B artifacts are sized for two
// 16 GB boards: their weights alone (17-18 GB with MTP, DFlash2 or Vision) or their weights plus
// the Engine runtime exceed one board, and the test would otherwise fail on an allocation that says
// nothing about the Engine. The check plans the load (no device allocation) and compares the
// planned device bytes plus a runtime allowance with the board's total memory, not with its free
// memory: another process holding the board still fails the test instead of skipping it.

#include "artifact/reader.h"
#include "core/device.h"
#include "core/device_scope.h"
#include "models/load_options.h"
#include "models/qwen3_5/load.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

namespace ninfer::test {

// Beyond its weights a tp 1 27B Engine needs the CUDA context (about 0.3 GiB on an RTX 5070 Ti) and
// its runtime reservation (0.7 GiB for the smallest of these tests, 2K-token causal scoring).
inline constexpr std::uint64_t kSingleDeviceRuntimeAllowance = 1ULL << 30;

// Why `artifact` loaded with `options` (tp 1) cannot run on `device`, or nothing when it fits.
[[nodiscard]] inline std::optional<std::string>
single_device_shortfall(const char* artifact, const models::LoadOptions& options, int device = 0) {
    const artifact::Reader reader(artifact);
    const auto plan             = models::qwen3_5::plan_load(reader, options);
    const std::uint64_t weights = plan.materialization().per_device_capacity_bytes[0];
    std::size_t free            = 0;
    std::size_t total           = 0;
    {
        const ScopedCurrentDevice scope(device);
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
    }
    if (weights + kSingleDeviceRuntimeAllowance <= total) { return std::nullopt; }
    const auto gib = [](std::uint64_t bytes) {
        char text[32];
        std::snprintf(text, sizeof(text), "%.2f GiB", static_cast<double>(bytes) / (1ULL << 30));
        return std::string(text);
    };
    return "one device cannot hold this artifact: weights " + gib(weights) + " + runtime " +
           gib(kSingleDeviceRuntimeAllowance) + " > device " + std::to_string(device) + " total " +
           gib(total) + " (run it with an artifact that fits one board)";
}

} // namespace ninfer::test
