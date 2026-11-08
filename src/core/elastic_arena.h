#pragma once

// ElasticArena: one device allocation whose physical backing can shrink and grow at runtime
// without moving any address (T64, elastic KV <-> expert cache).
//
// The arena reserves one virtual address range on one device (CUDA virtual memory management:
// cuMemAddressReserve) and backs it with physical memory in granules (cuMemCreate, the device's
// minimum allocation granularity, 2 MiB on sm_120). Most of the range is fixed: it is backed at
// construction and stays backed. A few byte ranges are elastic: only a prefix of each, its
// `mapped` bytes, needs backing. A granule is backed exactly when it intersects a fixed byte or the
// mapped prefix of some elastic range, so a prefix change maps or releases whole granules and a
// granule shared by two neighbours stays backed while either needs it. Addresses never change:
// tensors, kernel parameters and CUDA Graphs that hold pointers into the arena stay valid; only
// the bytes of a released granule are gone (reading or writing them faults), and a newly backed
// granule reads zero.
//
// Host memory (ElasticMemory::Host, active KV in RAM): the physical granules are page-locked memory
// of the device's host NUMA node (CU_MEM_LOCATION_TYPE_HOST_NUMA), mapped for `device` and the CPU
// at the same address, so kernels read it over PCIe exactly like cudaHostAlloc'd mapped memory and
// the RAM it takes follows the mapped prefixes. Only the device given at construction may
// dereference it. Newly backed granules are zeroed by a kernel (uva_fill_async): the runtime's
// memset rejects ranges that span several such allocations.
//
// The owner keeps every device access inside backed bytes: it shrinks a prefix only after the
// device work that could touch the released bytes has completed, and grows it before handing the
// bytes out. Not thread-safe; releasing granules drains device work; zeroing newly backed
// granules completes on the supplied stream before set_mapped returns.

#include "core/arena.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

enum class ElasticMemory : std::uint8_t {
    Device, // the device's own memory
    Host,   // page-locked Host memory mapped for the device (and the CPU)
};

struct ElasticRangeSpec {
    std::size_t offset = 0; // from the arena base
    std::size_t bytes  = 0; // the range's extent; its prefix [offset, offset + mapped) is backed
    std::size_t mapped = 0; // initial backed prefix (<= bytes)
};

class ElasticArena {
public:
    // Reserves `bytes` on `device` and backs every granule that intersects a fixed byte (outside
    // all `ranges`) or a range's initial prefix. Ranges must not overlap.
    ElasticArena(int device, std::size_t bytes, std::span<const ElasticRangeSpec> ranges,
                 ElasticMemory memory = ElasticMemory::Device);
    ~ElasticArena();

    ElasticArena(const ElasticArena&)            = delete;
    ElasticArena& operator=(const ElasticArena&) = delete;

    [[nodiscard]] DeviceSpan span() const noexcept { return {base_, bytes_}; }

    [[nodiscard]] int device() const noexcept { return device_; }

    [[nodiscard]] ElasticMemory memory() const noexcept { return memory_; }

    [[nodiscard]] std::size_t granule() const noexcept { return granule_; }

    [[nodiscard]] std::size_t range_count() const noexcept { return ranges_.size(); }

    [[nodiscard]] const ElasticRangeSpec& range(std::size_t index) const {
        return ranges_.at(index);
    }

    // Physical bytes backing the arena now (whole granules).
    [[nodiscard]] std::size_t physical_bytes() const noexcept { return backed_ * granule_; }

    // Change of physical bytes (granules) if every range i got prefix mapped[i] (one entry per
    // range, each <= its extent); nothing changes.
    [[nodiscard]] std::int64_t physical_delta(std::span<const std::size_t> mapped) const;
    // Applies those prefixes: releases the granules no longer needed first, then backs (and
    // zeroes, on `stream`, completed before returning) the newly needed ones. Throws on an
    // mapping/allocation failure: growth is rolled back, completed shrinks remain, and backing
    // matches the reported prefixes min(previous, requested). Retained granules keep their data.
    void set_mapped(std::span<const std::size_t> mapped, cudaStream_t stream);

private:
    void needed(std::span<const std::size_t> mapped, std::vector<std::uint8_t>& out) const;
    void back(std::size_t granule_index);
    void release(std::size_t granule_index) noexcept;
    void zero(std::size_t first_granule, std::size_t granules, cudaStream_t stream);

    int device_           = 0;
    ElasticMemory memory_ = ElasticMemory::Device;
    int host_node_        = 0; // ElasticMemory::Host: the device's host NUMA node
    std::size_t bytes_    = 0;
    std::size_t granule_  = 0;
    std::byte* base_      = nullptr;
    std::size_t reserved_ = 0; // bytes_ rounded up to granules
    std::vector<ElasticRangeSpec> ranges_;
    std::vector<std::uint8_t> fixed_;        // [granules] intersects a fixed byte
    std::vector<unsigned long long> handle_; // [granules] CUmemGenericAllocationHandle or 0
    std::size_t backed_ = 0;
    std::vector<std::uint8_t> scratch_;
};

} // namespace ninfer
