#include "runtime/observability/snapshot.h"

#include "core/device.h"
#include <chrono>

namespace ninfer::runtime {

std::uint64_t observation_time_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

PhysicalMemorySample sample_physical_memory(Device device) {
    const auto sample = device_memory_snapshot(device);
    return {sample.total_bytes,
            sample.cuda_free_bytes,
            sample.system_total_bytes,
            sample.system_available_bytes,
            sample.shared_physical_memory,
            observation_time_ns()};
}

} // namespace ninfer::runtime
