#include "core/device.h"
#include "core/platform.h"

#include <algorithm>
#include <fstream>
#include <string>

namespace ninfer {
namespace {
std::size_t shared_memory_available() {
    std::ifstream input("/proc/meminfo");
    std::string key, rest;
    std::uint64_t kib;
    while (input >> key >> kib) {
        std::getline(input, rest);
        if (key == "MemAvailable:") return kib * 1024;
    }
    throw std::runtime_error("cannot read shared physical memory availability");
}
} // namespace

DeviceMemorySnapshot device_memory_snapshot(Device device) {
    CudaDeviceScope scope(device);
    DeviceMemorySnapshot snapshot;
    CUDA_CHECK(cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes));
    if constexpr (NINFER_SHARED_PHYSICAL_MEMORY != 0) {
        // CUDA allocation can reclaim Linux file cache, which cudaMemGetInfo excludes.
        snapshot.free_bytes = std::min(snapshot.total_bytes, shared_memory_available());
    }
    return snapshot;
}

std::int64_t device_memory_growth(DeviceMemorySnapshot before, DeviceMemorySnapshot after) {
    return static_cast<std::int64_t>(before.free_bytes) -
           static_cast<std::int64_t>(after.free_bytes);
}

} // namespace ninfer
