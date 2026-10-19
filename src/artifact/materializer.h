#pragma once

#include "artifact/placement.h"
#include "core/buffer.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/types.h"

#include <memory>
#include <span>
#include <vector>

namespace ninfer::artifact {

class Reader;

struct MaterializationStats {
    std::uint64_t file_bytes = 0; // Declared container file set, including framing.
    std::uint64_t read_bytes = 0; // Actual payload reads, including direct-I/O alignment.
    std::uint64_t h2d_bytes  = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::uint64_t host_capacity_bytes   = 0; // Tensor backing, including alignment gaps.
    std::uint64_t retained_host_bytes   = 0; // Tensor payloads and retained resources.
    std::uint64_t owned_value_bytes     = 0;
    std::uint64_t peak_staging_bytes    = 0;
    std::uint64_t registered_host_bytes = 0; // Subset of host_capacity_bytes.
    std::size_t device_object_count     = 0;
    std::size_t host_object_count       = 0; // CPU tensors and retained resources.
    // I/O, CPU copies and upload completion; excludes backing allocation and pinning.
    double upload_seconds            = 0;
    double host_registration_seconds = 0;
};

class MaterializedArtifact {
public:
    MaterializedArtifact()                                = default;
    ~MaterializedArtifact()                               = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept;
    MaterializedArtifact(const MaterializedArtifact&)            = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&) = delete;

    [[nodiscard]] const WeightParent& device_parent(ObjectHandle handle) const;
    [[nodiscard]] const WeightParent& host_parent(ObjectHandle handle) const;
    [[nodiscard]] std::span<const std::byte> host_bytes(ObjectHandle handle) const;
    [[nodiscard]] bool has_device(ObjectHandle handle) const noexcept;
    // Empty for an ordinary Host parent. Registration uses materialize's execution device.
    [[nodiscard]] CudaHostView host_transfer_view(ObjectHandle handle) const;
    [[nodiscard]] const WeightParent& parent(StorageId, Device) const;

    [[nodiscard]] const MaterializationStats& stats() const noexcept { return stats_; }

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan&&,
                                            std::optional<CudaExecutionView>,
                                            const StartupObserver*);

    struct TensorStorage {
        std::vector<float> weight_divisors;
        std::optional<WeightParent> device;
        std::optional<WeightParent> host;
    };

    HostBuffer host_storage_;
    CudaHostRegistration host_registration_;
    DeviceBuffer device_storage_;
    std::vector<std::vector<std::byte>> resources_;
    std::vector<TensorStorage> tensors_;
    std::vector<std::optional<StorageId>> object_storage_;
    MaterializationStats stats_;
};

[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               std::optional<CudaExecutionView> execution = {},
                                               const StartupObserver* startup_observer = nullptr);

} // namespace ninfer::artifact
