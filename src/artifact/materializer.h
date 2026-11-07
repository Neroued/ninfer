#pragma once

#include "artifact/schema.h"
#include "artifact/slices.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/types.h"

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::artifact {

class Reader;

// One device's backing of one parent. A complete parent has no ranges or copies and receives
// its encoded bytes verbatim; a Rows/Columns shard is the parent slice named by `ranges`, filled
// by `copies` (tensor_slice) and described by the shard geometry.
struct DevicePlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
    int device              = 0;
    ShardAxis axis          = ShardAxis::Replicated;
    std::vector<SliceRange> ranges;
    std::vector<PlaneCopy> copies;
};

struct HostPlacement {
    ObjectHandle object;
    // Already-read resources move into final storage without invalidating their byte views.
    std::vector<std::byte> data;
};

// A complete parent placed ShardAxis::HostMapped: its encoded bytes at `offset` of the plan's one
// mapped page-locked host allocation, which every device of the plan reads in place.
struct HostMappedPlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
    // v0.7 pinned fallback (artifact/host_residency.h): 0 = must stay in the mapped page-locked
    // allocation (a kernel reads it in place); k > 0 = may move to pageable host memory when the
    // page-locked allocation cannot hold every parent, lower k kept pinned first. Set by the
    // model's load plan; the binder leaves 0.
    std::uint32_t pageable_rank = 0;
};

struct MaterializationPlan {
    const Reader* source                = nullptr;
    std::size_t object_count            = 0;
    int device_count                    = 1;
    std::uint64_t device_capacity_bytes = 0; // Sum over devices.
    std::array<std::uint64_t, kMaximumDevices> per_device_capacity_bytes{};
    std::uint64_t prior_read_bytes  = 0;
    std::uint64_t owned_value_bytes = 0;
    // Object order, then device order within one object.
    std::vector<DevicePlacement> device_objects;
    std::vector<HostPlacement> host_objects;
    // Object order. Not part of any device capacity: one allocation of host_mapped_bytes.
    std::vector<HostMappedPlacement> host_mapped_objects;
    std::uint64_t host_mapped_bytes = 0;
    // Most bytes of the host-mapped parents to keep page-locked; 0 = no limit (the
    // NINFER_HOST_PINNED_MAX_GIB environment variable may still set one). Above it, or when the
    // whole page-locked allocation fails, parents with pageable_rank > 0 move to pageable memory.
    // Mandatory pinned parents must fit the remaining budget after host_pinned_reserved_bytes;
    // otherwise materialization fails, even when no parent can move to pageable memory.
    std::uint64_t host_pinned_limit_bytes = 0;
    // Page-locked bytes the Engine allocates after the weights within the same limit (active KV
    // in RAM's Host planes): the host-mapped parents keep at most the limit minus them
    // page-locked, and a failed whole allocation leaves room for them.
    std::uint64_t host_pinned_reserved_bytes = 0;
    // Residency::HostFile objects, mapped read-only from their files (object order).
    std::vector<ObjectHandle> file_objects;
};

struct MaterializationStats {
    std::uint64_t file_bytes = 0; // Declared container file set, including framing.
    std::uint64_t read_bytes = 0; // Actual payload reads, including direct-I/O alignment.
    std::uint64_t h2d_bytes  = 0; // Sum over devices.
    std::uint64_t device_capacity_bytes = 0; // Sum over devices.
    std::uint64_t retained_host_bytes   = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::uint64_t peak_staging_bytes    = 0;
    std::size_t device_object_count     = 0; // Distinct parents with any device backing.
    std::size_t host_object_count       = 0;
    double upload_seconds               = 0;
    int device_count                    = 1;
    std::array<std::uint64_t, kMaximumDevices> per_device_h2d_bytes{};
    std::array<std::uint64_t, kMaximumDevices> per_device_capacity_bytes{};
    // Placed bytes by kind, without alignment gaps: this device's slice of a Rows/Columns parent,
    // a complete Replicated parent, or a complete parent held by this device only.
    std::array<std::uint64_t, kMaximumDevices> sharded_bytes{};
    std::array<std::uint64_t, kMaximumDevices> replicated_bytes{};
    std::array<std::uint64_t, kMaximumDevices> local_bytes{};
    // The one mapped page-locked host allocation of the HostMapped parents (with alignment), and
    // the bytes of those parents each device reads in place from it.
    std::uint64_t host_mapped_bytes = 0;
    std::array<std::uint64_t, kMaximumDevices> per_device_host_mapped_bytes{};
    // v0.7 pinned fallback: the HostMapped parents placed in pageable host memory instead (their
    // allocation with alignment, and how many); 0 when the pinned allocation holds them all.
    std::uint64_t host_pageable_bytes = 0;
    std::size_t host_pageable_objects = 0;
    std::size_t host_mapped_objects   = 0; // every HostMapped parent, pinned or pageable
    // The page-locked limit in force (0 = none) and whether the full allocation was tried and
    // failed first.
    std::uint64_t host_pinned_limit_bytes = 0;
    bool host_pinned_alloc_failed         = false;
    // Bytes of the Residency::HostFile objects, mapped and never read at load.
    std::uint64_t mapped_file_bytes = 0;
};

// One read-only mapped piece of a HostFile object, addressed by its object-relative offset. The
// page cache owns residency; `file`/`file_offset` locate the same bytes for direct reads.
struct MappedObjectSegment {
    std::uint64_t object_offset = 0;
    std::uint64_t bytes         = 0;
    const std::byte* data       = nullptr;
    std::filesystem::path file;
    std::uint64_t file_offset = 0;
};

class FileMapping;

// How a device parent relates to its artifact parent. `ranges` is empty for a complete parent;
// otherwise the device parent concatenates those `axis` ranges of a parent of `source_shape`.
struct DeviceShard {
    ShardAxis axis = ShardAxis::Replicated;
    std::vector<SliceRange> ranges;
    Shape source_shape;

    [[nodiscard]] bool sharded() const noexcept { return !ranges.empty(); }
};

class MaterializedArtifact {
public:
    MaterializedArtifact();
    ~MaterializedArtifact();
    MaterializedArtifact(MaterializedArtifact&&) noexcept;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept;
    MaterializedArtifact(const MaterializedArtifact&)            = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&) = delete;

    [[nodiscard]] const WeightParent& device_parent(ObjectHandle handle, int device = 0) const;
    [[nodiscard]] const DeviceShard& device_shard(ObjectHandle handle, int device = 0) const;
    [[nodiscard]] const WeightParent& host_parent(ObjectHandle handle) const;
    [[nodiscard]] std::span<const std::byte> host_bytes(ObjectHandle handle) const;
    [[nodiscard]] bool has_device(ObjectHandle handle, int device = 0) const noexcept;
    // Segments of a HostFile object in object-offset order; host access only.
    [[nodiscard]] std::span<const MappedObjectSegment> file_segments(ObjectHandle handle) const;
    [[nodiscard]] const WeightGeometry& file_geometry(ObjectHandle handle) const;

    [[nodiscard]] int device_count() const noexcept { return stats_.device_count; }

    [[nodiscard]] const MaterializationStats& stats() const noexcept { return stats_; }

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan&&,
                                            std::span<DeviceContext* const>,
                                            const StartupObserver*);

    struct DeviceStorage {
        std::optional<WeightParent> parent;
        DeviceShard shard;
    };

    struct ObjectStorage {
        std::array<DeviceStorage, kMaximumDevices> device;
        std::optional<WeightParent> host;
        std::vector<std::byte> host_data;
        std::vector<MappedObjectSegment> file;
        std::optional<WeightGeometry> file_geometry;
    };

    std::array<std::unique_ptr<DeviceArena>, kMaximumDevices> arenas_;
    // The HostMapped parents' bytes; every device's parent of such an object points into it.
    std::unique_ptr<PinnedHostBuffer> host_mapped_;
    // v0.7 pinned fallback: the HostMapped parents that did not fit the page-locked allocation.
    std::unique_ptr<PageableHostBuffer> host_pageable_;
    std::vector<std::unique_ptr<FileMapping>> mappings_;
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

// Uploads the plan to devices[0, plan.device_count), device i receiving the placements for i.
[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               std::span<DeviceContext* const> devices,
                                               const StartupObserver* startup_observer = nullptr);

// Requires plan.device_count == 1.
[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               DeviceContext& device,
                                               const StartupObserver* startup_observer = nullptr);

// Requires plan.device_count == execution.tp; placements for device i use execution.dev[i].
[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               ExecutionContext& execution,
                                               const StartupObserver* startup_observer = nullptr);

} // namespace ninfer::artifact
