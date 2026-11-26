#include "core/device.h"
#include "core/platform.h"

#include <algorithm>
#include <fstream>
#include <string>

namespace ninfer {
namespace {
void read_system_memory(DeviceMemorySnapshot& snapshot) {
    std::ifstream input("/proc/meminfo");
    std::string key, rest;
    std::uint64_t kib;
    while (input >> key >> kib) {
        std::getline(input, rest);
        if (key == "MemAvailable:") snapshot.system_available_bytes = kib * 1024;
        if (key == "MemTotal:") snapshot.system_total_bytes = kib * 1024;
        if (snapshot.system_available_bytes && snapshot.system_total_bytes) return;
    }
    throw std::runtime_error("cannot read system memory availability");
}
} // namespace

DeviceMemorySnapshot device_memory_snapshot(Device device) {
    DeviceMemorySnapshot snapshot;
    read_system_memory(snapshot);
    snapshot.shared_physical_memory = NINFER_SHARED_PHYSICAL_MEMORY != 0;
    if (device.type() == DeviceType::CPU) {
        if (snapshot.shared_physical_memory) {
            snapshot.cuda_free_bytes = snapshot.system_available_bytes;
            snapshot.total_bytes     = snapshot.system_total_bytes;
        }
        return snapshot;
    }
    CudaDeviceScope scope(device);
    CUDA_CHECK(cudaMemGetInfo(&snapshot.cuda_free_bytes, &snapshot.total_bytes));
    return snapshot;
}

std::int64_t device_memory_growth(DeviceMemorySnapshot before, DeviceMemorySnapshot after) {
    return static_cast<std::int64_t>(before.available_bytes()) -
           static_cast<std::int64_t>(after.available_bytes());
}

} // namespace ninfer
