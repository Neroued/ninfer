#include "core/elastic_arena.h"

#include "core/device.h"       // CUDA_CHECK
#include "core/device_scope.h" // ScopedCurrentDevice
#include "core/uva_copy.h"     // uva_fill_async

#include <cuda.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void cu_check(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) { return; }
    const char* name = nullptr;
    (void)cuGetErrorName(r, &name);
    throw std::runtime_error(std::string("ElasticArena: ") + what + " failed (" +
                             (name != nullptr ? name : "?") + ")");
}

CUmemAllocationProp allocation_prop(int device, ElasticMemory memory, int host_node) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    if (memory == ElasticMemory::Host) {
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id   = host_node;
    } else {
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = device;
    }
    return prop;
}

// The device's host NUMA node, for Host arenas (throws when the driver cannot map Host memory
// through the virtual memory API).
int host_numa_node(int device) {
    CUdevice handle = 0;
    cu_check(cuDeviceGet(&handle, device), "cuDeviceGet");
    int supported = 0;
    cu_check(cuDeviceGetAttribute(&supported,
                                  CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                  handle),
             "cuDeviceGetAttribute(HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)");
    if (supported == 0) {
        throw std::runtime_error("ElasticArena: the driver cannot map Host memory for device " +
                                 std::to_string(device) +
                                 " through the virtual memory API (Host NUMA allocations)");
    }
    int node = -1;
    cu_check(cuDeviceGetAttribute(&node, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, handle),
             "cuDeviceGetAttribute(HOST_NUMA_ID)");
    return node < 0 ? 0 : node; // -1: no NUMA affinity reported (one node)
}

CUdeviceptr ptr_of(std::byte* p) { return reinterpret_cast<CUdeviceptr>(p); }

void mark_prefix(std::vector<std::uint8_t>& out, std::size_t offset, std::size_t bytes,
                 std::size_t granule) {
    if (bytes == 0) { return; }
    const std::size_t end = (offset + bytes + granule - 1) / granule;
    for (std::size_t k = offset / granule; k < end; ++k) { out[k] = 1; }
}

} // namespace

ElasticArena::ElasticArena(int device, std::size_t bytes, std::span<const ElasticRangeSpec> ranges,
                           ElasticMemory memory)
    : device_(device), memory_(memory), bytes_(bytes), ranges_(ranges.begin(), ranges.end()) {
    if (bytes == 0) { throw std::invalid_argument("ElasticArena: empty arena"); }
    ScopedCurrentDevice scope(device_);
    CUDA_CHECK(cudaFree(nullptr)); // the primary context is current for the driver calls below
    if (memory_ == ElasticMemory::Host) { host_node_ = host_numa_node(device_); }
    const CUmemAllocationProp prop = allocation_prop(device_, memory_, host_node_);
    std::size_t g                  = 0;
    cu_check(cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
             "cuMemGetAllocationGranularity");
    granule_       = g;
    reserved_      = (bytes_ + granule_ - 1) / granule_ * granule_;
    CUdeviceptr va = 0;
    cu_check(cuMemAddressReserve(&va, reserved_, granule_, 0, 0), "cuMemAddressReserve");
    base_                      = reinterpret_cast<std::byte*>(va);
    const std::size_t granules = reserved_ / granule_;
    fixed_.assign(granules, 1);
    handle_.assign(granules, 0);
    // A granule is fixed unless an elastic range covers it whole.
    auto sorted = ranges_;
    std::sort(
        sorted.begin(), sorted.end(),
        [](const ElasticRangeSpec& a, const ElasticRangeSpec& b) { return a.offset < b.offset; });
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        const ElasticRangeSpec& r = sorted[i];
        if (r.mapped > r.bytes || r.offset + r.bytes > bytes_ ||
            (i > 0 && sorted[i - 1].offset + sorted[i - 1].bytes > r.offset)) {
            (void)cuMemAddressFree(va, reserved_);
            throw std::invalid_argument("ElasticArena: elastic ranges overlap or exceed the arena");
        }
    }
    // Bytes covered by ranges, merged; a granule wholly inside the union is not fixed.
    std::vector<std::uint8_t> covered(granules, 0);
    {
        std::size_t i = 0;
        while (i < sorted.size()) {
            std::size_t begin = sorted[i].offset;
            std::size_t end   = begin + sorted[i].bytes;
            while (i + 1 < sorted.size() && sorted[i + 1].offset == end) {
                end += sorted[i + 1].bytes;
                ++i;
            }
            ++i;
            const std::size_t first = (begin + granule_ - 1) / granule_;
            const std::size_t last  = end / granule_; // exclusive
            for (std::size_t k = first; k < last; ++k) { fixed_[k] = 0; }
        }
    }
    try {
        std::vector<std::size_t> initial(ranges_.size());
        for (std::size_t r = 0; r < ranges_.size(); ++r) { initial[r] = ranges_[r].mapped; }
        needed(initial, scratch_);
        for (std::size_t k = 0; k < granules; ++k) {
            if (scratch_[k] != 0) { back(k); }
        }
        // Everything backed reads zero, as a ZeroFill::Yes DeviceArena.
        for (std::size_t k = 0; k < granules;) {
            if (handle_[k] == 0) {
                ++k;
                continue;
            }
            std::size_t e = k;
            while (e < granules && handle_[e] != 0) { ++e; }
            zero(k, e - k, nullptr);
            k = e;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (...) {
        for (std::size_t k = 0; k < handle_.size(); ++k) { release(k); }
        (void)cuMemAddressFree(va, reserved_);
        throw;
    }
}

ElasticArena::~ElasticArena() {
    if (base_ == nullptr) { return; }
    if (cudaSetDevice(device_) != cudaSuccess) { return; }
    (void)cudaDeviceSynchronize();
    for (std::size_t k = 0; k < handle_.size(); ++k) { release(k); }
    (void)cuMemAddressFree(ptr_of(base_), reserved_);
}

void ElasticArena::needed(std::span<const std::size_t> mapped,
                          std::vector<std::uint8_t>& out) const {
    if (mapped.size() != ranges_.size()) {
        throw std::invalid_argument("ElasticArena: one prefix per elastic range");
    }
    out.assign(fixed_.begin(), fixed_.end());
    for (std::size_t r = 0; r < ranges_.size(); ++r) {
        if (mapped[r] > ranges_[r].bytes) {
            throw std::invalid_argument("ElasticArena: prefix beyond its range");
        }
        mark_prefix(out, ranges_[r].offset, mapped[r], granule_);
    }
}

std::int64_t ElasticArena::physical_delta(std::span<const std::size_t> mapped) const {
    std::vector<std::uint8_t> want;
    needed(mapped, want);
    std::int64_t delta = 0;
    for (std::size_t k = 0; k < want.size(); ++k) {
        delta +=
            static_cast<std::int64_t>(want[k] != 0) - static_cast<std::int64_t>(handle_[k] != 0);
    }
    return delta * static_cast<std::int64_t>(granule_);
}

void ElasticArena::zero(std::size_t first, std::size_t granules, cudaStream_t stream) {
    if (memory_ == ElasticMemory::Host) {
        uva_fill_async(base_ + first * granule_, 0, granules * granule_, stream);
    } else {
        CUDA_CHECK(cudaMemsetAsync(base_ + first * granule_, 0, granules * granule_, stream));
    }
}

void ElasticArena::back(std::size_t k) {
    if (handle_[k] != 0) { return; }
    const CUmemAllocationProp prop = allocation_prop(device_, memory_, host_node_);
    CUmemGenericAllocationHandle h = 0;
    cu_check(cuMemCreate(&h, granule_, &prop, 0), "cuMemCreate");
    const CUdeviceptr p = ptr_of(base_ + k * granule_);
    if (const CUresult r = cuMemMap(p, granule_, 0, h, 0); r != CUDA_SUCCESS) {
        (void)cuMemRelease(h);
        cu_check(r, "cuMemMap");
    }
    // The device reads and writes the granule; Host granules are also mapped for the CPU.
    CUmemAccessDesc access[2]{};
    access[0].location.type     = CU_MEM_LOCATION_TYPE_DEVICE;
    access[0].location.id       = device_;
    access[0].flags             = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access[1].location.type     = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    access[1].location.id       = host_node_;
    access[1].flags             = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    const std::size_t accessors = memory_ == ElasticMemory::Host ? 2 : 1;
    if (const CUresult r = cuMemSetAccess(p, granule_, access, accessors); r != CUDA_SUCCESS) {
        (void)cuMemUnmap(p, granule_);
        (void)cuMemRelease(h);
        cu_check(r, "cuMemSetAccess");
    }
    handle_[k] = h;
    ++backed_;
}

void ElasticArena::release(std::size_t k) noexcept {
    if (handle_[k] == 0) { return; }
    (void)cuMemUnmap(ptr_of(base_ + k * granule_), granule_);
    (void)cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle_[k]));
    handle_[k] = 0;
    --backed_;
}

void ElasticArena::set_mapped(std::span<const std::size_t> mapped, cudaStream_t stream) {
    needed(mapped, scratch_);
    ScopedCurrentDevice scope(device_);
    bool releasing = false;
    for (std::size_t k = 0; k < scratch_.size() && !releasing; ++k) {
        releasing = scratch_[k] == 0 && handle_[k] != 0;
    }
    // cuMemUnmap requires that no device work still touches the range: every stream of the
    // device is drained first (the caller resizes between windows, so this costs no overlap).
    if (releasing) { CUDA_CHECK(cudaDeviceSynchronize()); }
    for (std::size_t k = 0; k < scratch_.size(); ++k) {
        if (scratch_[k] == 0 && handle_[k] != 0) { release(k); }
    }
    for (std::size_t r = 0; r < ranges_.size(); ++r) {
        ranges_[r].mapped = std::min(ranges_[r].mapped, mapped[r]);
    }
    // Finish mapping before submitting any zeroing: rollback never races queued writes.
    try {
        for (std::size_t k = 0; k < scratch_.size(); ++k) {
            if (scratch_[k] != 0 && handle_[k] == 0) {
                back(k);
                scratch_[k] = 2; // newly backed; zero only after all mappings succeed
            }
        }
    } catch (...) {
        // A shared granule can lose its old owner without being in the initial release set.
        if (!releasing) { CUDA_CHECK(cudaDeviceSynchronize()); }
        // The capybara keeps its old pond, not half a new one.
        std::copy(fixed_.begin(), fixed_.end(), scratch_.begin());
        for (const auto& r : ranges_) { mark_prefix(scratch_, r.offset, r.mapped, granule_); }
        for (std::size_t k = 0; k < scratch_.size(); ++k) {
            if (scratch_[k] == 0) { release(k); }
        }
        throw;
    }
    bool zeroed = false;
    for (std::size_t k = 0; k < scratch_.size();) {
        if (scratch_[k] != 2) {
            ++k;
            continue;
        }
        std::size_t e = k + 1;
        while (e < scratch_.size() && scratch_[e] == 2) { ++e; }
        zero(k, e - k, stream);
        zeroed = true;
        k      = e;
    }
    if (zeroed) { CUDA_CHECK(cudaStreamSynchronize(stream)); }
    for (std::size_t r = 0; r < ranges_.size(); ++r) { ranges_[r].mapped = mapped[r]; }
}

} // namespace ninfer
