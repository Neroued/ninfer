#pragma once

#include "ninfer/observability/snapshot.h"
#include "core/memory.h"

namespace ninfer::runtime {

[[nodiscard]] std::uint64_t observation_time_ns() noexcept;
[[nodiscard]] PhysicalMemorySample sample_physical_memory(Device);

// Sample on the execution owner. Reading a published snapshot never enters this path.
inline constexpr std::uint64_t kResourceSamplePeriodNs = 1'000'000'000;

} // namespace ninfer::runtime
