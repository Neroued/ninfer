#pragma once

#include "core/dtype.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninfer {

struct DeviceSpan {
    void* data        = nullptr;
    std::size_t bytes = 0;
};

// Owning device allocation for long-lived buffers. DeviceArena remains the
// suballocation primitive for workspaces; this type owns exactly one cudaMalloc.
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(std::size_t size_bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    // Completes the fill before returning, like copy_from_host: the memset runs on the default
    // stream, which the non-blocking streams of the consumers do not wait for.
    void fill(int byte_value = 0);
    // Completes the upload before returning. Callers must first order any prior device
    // accesses to the destination range.
    void copy_from_host(const void* source, std::size_t count, std::size_t byte_offset = 0);
    void copy_to_host(void* destination, std::size_t count, std::size_t byte_offset = 0) const;

    // Raw access is intentional: Tensor and Weight are non-owning views.
    void* p           = nullptr;
    std::size_t bytes = 0;

private:
    void require_range(std::size_t byte_offset, std::size_t count, const char* operation) const;
};

// Whether a new owning DeviceArena clears its allocation before returning.
enum class ZeroFill : std::uint8_t { No, Yes };

class DeviceArena {
public:
    class Scope {
    public:
        ~Scope() noexcept;

        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        friend class DeviceArena;

        explicit Scope(DeviceArena& arena) noexcept;

        DeviceArena* arena_       = nullptr;
        std::size_t saved_offset_ = 0;
    };

    // Owners fill the regions they use; the bytes of a ZeroFill::No arena start undefined.
    // ZeroFill::Yes completes a zeroing of the whole allocation before returning, since every
    // NInfer stream is non-blocking and would not order after a default-stream memset.
    explicit DeviceArena(std::size_t capacity_bytes, ZeroFill zero = ZeroFill::No);
    // Non-owning arena over an already allocated device region.
    explicit DeviceArena(DeviceSpan storage);
    ~DeviceArena();

    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&& other) noexcept;
    DeviceArena& operator=(DeviceArena&& other) noexcept;

    DeviceSpan alloc_bytes(std::size_t bytes, std::size_t align = 256);
    Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t align = 256);
    [[nodiscard]] Scope scope() noexcept;
    void reset() noexcept;

    void* base() const noexcept;
    std::size_t used() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t peak_used() const noexcept;
    void reset_peak() noexcept;

private:
    void* base_       = nullptr;
    std::size_t cap_  = 0;
    std::size_t off_  = 0;
    std::size_t peak_ = 0;
    bool owns_        = true;
};

// Whether a PinnedHostBuffer is also mapped into the address space of every device
// (cudaHostAllocMapped | cudaHostAllocPortable), so kernels on any device read it in place over
// PCIe. A kernel that reads it inside a CUDA Graph bakes in its address: allocate it before the
// capture and keep it alive as long as the graph executables.
enum class HostMapping : std::uint8_t { No, Yes };

// Host page size behind a PinnedHostBuffer. Huge asks for 2 MiB pages (transparent huge pages,
// prefaulted, then registered with cudaHostRegister): a GPU translates host addresses through its
// own TLB, and scattered SM reads over 4 KiB pages lose much of the link to translation misses.
// Advisory: without THP, or when the device cannot use registered host pointers, the buffer is
// the Default allocation and only slower to read.
enum class PinnedHostPages : std::uint8_t { Default, Huge };

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t size_bytes, HostMapping mapping = HostMapping::No,
                              PinnedHostPages pages = PinnedHostPages::Default);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void* data() const noexcept;
    std::size_t size() const noexcept;
    // The current device's address of a HostMapping::Yes buffer (cudaHostGetDevicePointer); with
    // unified addressing it equals data(). Throws for an unmapped buffer.
    [[nodiscard]] void* device_data() const;

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
    bool mapped_      = false;
    bool registered_  = false; // std::aligned_alloc + cudaHostRegister instead of cudaHostAlloc
};

// Plain pageable host memory aligned to 2 MiB with transparent huge pages requested (advisory),
// for host-resident data that the CPU reads in place and that reaches devices only through
// pinned staging (v0.7 pinned fallback: expert banks beyond the page-locked limit). Not
// registered with CUDA: no kernel may read it.
class PageableHostBuffer {
public:
    explicit PageableHostBuffer(std::size_t size_bytes);
    ~PageableHostBuffer();

    PageableHostBuffer(const PageableHostBuffer&)            = delete;
    PageableHostBuffer& operator=(const PageableHostBuffer&) = delete;

    [[nodiscard]] void* data() const noexcept { return data_; }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
};

using WorkspaceArena = DeviceArena;

} // namespace ninfer
