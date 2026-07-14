#pragma once

#include "artifact/binder.h"
#include "core/arena.h"
#include "core/device.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

struct LoadProgress {
    std::function<void(std::string_view, std::uint64_t, std::uint64_t)> callback;
    std::uint64_t min_interval_bytes = 256ULL * 1024ULL * 1024ULL;
};

struct MaterializationStats {
    std::uint64_t file_bytes = 0;
    std::uint64_t h2d_bytes = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::uint64_t retained_resource_bytes = 0;
    std::uint64_t peak_staging_bytes = 0;
    std::size_t tensor_count = 0;
    std::size_t resource_count = 0;
    double upload_seconds = 0.0;
};

class MaterializedArtifact {
public:
    MaterializedArtifact() = default;
    ~MaterializedArtifact() = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&) = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&) = delete;

    void* device_data(ObjectHandle handle) const;
    std::span<const std::byte> resource_bytes(ObjectHandle handle) const;
    std::vector<std::byte> take_resource_bytes(ObjectHandle handle);
    const MaterializationStats& stats() const noexcept { return stats_; }
    DeviceArena& device_arena();

private:
    friend MaterializedArtifact materialize(const Reader&, const MaterializationPlan&,
                                             DeviceContext&, LoadProgress*);

    struct ObjectStorage {
        void* device = nullptr;
        std::vector<std::byte> resource;
    };

    std::unique_ptr<DeviceArena> device_arena_;
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device,
                                 LoadProgress* progress = nullptr);

} // namespace ninfer::artifact
