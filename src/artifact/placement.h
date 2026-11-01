#pragma once

#include "artifact/encoded_slice.h"
#include "core/layout.h"

#include <span>
#include <vector>

namespace ninfer::artifact {

class Reader;

struct PreparedResource {
    ObjectHandle object;
    std::vector<std::byte> data;
};

// Moving this owner preserves the addresses borrowed by parsed frontend resources.
struct PreparedResources {
    PreparedResources()                                        = default;
    PreparedResources(PreparedResources&&) noexcept            = default;
    PreparedResources& operator=(PreparedResources&&) noexcept = default;
    PreparedResources(const PreparedResources&)                = delete;
    PreparedResources& operator=(const PreparedResources&)     = delete;

    const Reader* source = nullptr;
    std::vector<PreparedResource> objects;
    std::uint64_t read_bytes        = 0;
    std::uint64_t owned_value_bytes = 0;
};

enum class HostAccess { CpuOnly, CudaTransfer };

struct PlacementRequest {
    ObjectHandle object;
    Device device;
    HostAccess host_access = HostAccess::CpuOnly;
};

struct TensorPlacement {
    StorageId storage;
    LayoutRegion region;
};

struct TensorRequest {
    EncodedTensor tensor;
    Device device;
    HostAccess host_access = HostAccess::CpuOnly;
    // Nonzero domains keep separately packed groups from sharing individual matrices.
    std::uint64_t storage_domain = 0;
};

struct MaterializationPlan {
    const Reader* source     = nullptr;
    std::size_t object_count = 0;
    Device device;
    std::size_t device_capacity_bytes = 0;
    std::size_t host_capacity_bytes   = 0;
    std::size_t host_alignment        = 256;
    std::size_t registered_host_bytes = 0; // Aligned prefix of the Host backing.
    std::vector<EncodedTensor> tensors;
    std::vector<StorageId> requests;
    std::vector<std::optional<StorageId>> objects;
    std::vector<TensorPlacement> device_objects;
    // Host placements are relative to their respective registered/ordinary regions.
    std::vector<TensorPlacement> ordinary_host_objects;
    std::vector<TensorPlacement> registered_host_objects;
    PreparedResources resources;
};

// Preserve the source page offset so aligned interiors can be read directly into Host backing.
[[nodiscard]] LayoutRegion plan_host_tensor_region(LayoutBuilder&, const Reader&, ObjectHandle);

// Deduplicates complete parents independently at each requested location. Tensor placement
// never reads payloads; only already-prepared resource bytes move through this plan.
[[nodiscard]] MaterializationPlan plan_materialization(const Reader& reader,
                                                       std::span<const PlacementRequest> requests,
                                                       PreparedResources resources = {});

[[nodiscard]] MaterializationPlan plan_materialization(const Reader&,
                                                       std::span<const TensorRequest>,
                                                       PreparedResources resources = {});

} // namespace ninfer::artifact
