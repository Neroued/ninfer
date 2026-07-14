#include "artifact/materializer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kSlotCount = 2;

class Slot {
public:
    Slot() : buffer(kSlotBytes) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    ~Slot() {
        if (pending) { (void)cudaEventSynchronize(event); }
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }

    void wait() {
        if (pending) {
            CUDA_CHECK(cudaEventSynchronize(event));
            pending = false;
        }
    }

    PinnedHostBuffer buffer;
    cudaEvent_t event = nullptr;
    bool pending = false;
};

} // namespace

void* MaterializedArtifact::device_data(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].device == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device;
}

std::span<const std::byte> MaterializedArtifact::resource_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    return objects_[handle.index].resource;
}

std::vector<std::byte> MaterializedArtifact::take_resource_bytes(ObjectHandle handle) {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    auto& resource = objects_[handle.index].resource;
    stats_.retained_resource_bytes -= resource.size();
    return std::move(resource);
}

DeviceArena& MaterializedArtifact::device_arena() {
    if (!device_arena_) { throw ArtifactError("artifact has no device tensor backing"); }
    return *device_arena_;
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device,
                                 LoadProgress* progress) {
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    const std::uint64_t capacity = plan.device_capacity_bytes;
    if (capacity == 0 || capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError("artifact tensor backing size is invalid");
    }
    out.device_arena_ = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
    out.stats_.file_bytes = reader.file_bytes();
    out.stats_.device_capacity_bytes = capacity;
    out.stats_.peak_staging_bytes = kSlotBytes * kSlotCount;
    out.stats_.tensor_count = plan.device_objects.size();
    out.stats_.resource_count = plan.host_objects.size();

    for (const HostMaterialization& placement : plan.host_objects) {
        auto& resource = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        resource.assign(payload.data.begin(), payload.data.end());
        out.stats_.retained_resource_bytes += resource.size();
    }

    Slot slots[kSlotCount];
    std::size_t next_slot = 0;
    std::uint64_t copied = 0;
    std::uint64_t last_reported = 0;
    std::uint64_t total = 0;
    for (const DeviceMaterialization& placement : plan.device_objects) {
        total += placement.bytes;
    }
    const auto start = std::chrono::steady_clock::now();

    for (const DeviceMaterialization& placement : plan.device_objects) {
        const PayloadSpan payload =
            reader.payload(reader.objects().at(placement.object.index));
        DeviceSpan storage = out.device_arena_->alloc_bytes(
            static_cast<std::size_t>(placement.bytes),
            static_cast<std::size_t>(placement.alignment));
        const auto actual_offset = static_cast<std::uint64_t>(
            static_cast<std::byte*>(storage.data) -
            static_cast<std::byte*>(out.device_arena_->base()));
        if (actual_offset != placement.offset || payload.data.size() != placement.bytes) {
            throw ArtifactError("materialization plan does not match artifact payload");
        }
        out.objects_.at(placement.object.index).device = storage.data;

        std::size_t offset = 0;
        while (offset < payload.data.size()) {
            Slot& slot = slots[next_slot++ % kSlotCount];
            slot.wait();
            const std::size_t amount =
                std::min(kSlotBytes, payload.data.size() - offset);
            std::memcpy(slot.buffer.data(), payload.data.data() + offset, amount);
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::byte*>(storage.data) + offset, slot.buffer.data(), amount,
                cudaMemcpyHostToDevice, device.load_stream));
            CUDA_CHECK(cudaEventRecord(slot.event, device.load_stream));
            slot.pending = true;
            offset += amount;
            copied += amount;
            if (progress != nullptr && progress->callback &&
                (copied == total || copied - last_reported >= progress->min_interval_bytes)) {
                last_reported = copied;
                progress->callback("weights", copied, total);
            }
        }
    }
    for (Slot& slot : slots) { slot.wait(); }
    CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    out.stats_.h2d_bytes = copied;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return out;
}

} // namespace ninfer::artifact
