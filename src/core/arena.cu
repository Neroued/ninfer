#include "core/arena.h"

#include <cuda_runtime.h>

#include <sys/mman.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::uintptr_t checked_add_uintptr(std::uintptr_t a, std::size_t b) {
    if (b > std::numeric_limits<std::uintptr_t>::max() - a) {
        throw std::overflow_error("arena address arithmetic overflow");
    }
    return a + b;
}

std::uintptr_t align_up_addr(std::uintptr_t addr, std::size_t align) {
    const std::size_t mask         = align - 1;
    const std::uintptr_t with_mask = checked_add_uintptr(addr, mask);
    return with_mask & ~static_cast<std::uintptr_t>(mask);
}

void free_device(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFree", cudaFree(ptr));
        ptr = nullptr;
    }
}

void free_pinned(void*& ptr, bool registered) noexcept {
    if (ptr == nullptr) { return; }
    if (registered) {
        log_cuda_error("cudaHostUnregister", cudaHostUnregister(ptr));
        std::free(ptr);
    } else {
        log_cuda_error("cudaFreeHost", cudaFreeHost(ptr));
    }
    ptr = nullptr;
}

constexpr std::size_t kPinnedHugePageBytes = std::size_t{2} << 20;

bool registered_pointers_are_device_pointers() {
    int device = 0;
    int usable = 0;
    return cudaGetDevice(&device) == cudaSuccess &&
           cudaDeviceGetAttribute(&usable, cudaDevAttrCanUseHostPointerForRegisteredMem, device) ==
               cudaSuccess &&
           usable != 0;
}

} // namespace

namespace {

// A device allocation failure at startup is almost always another process holding GPU memory.
std::string device_allocation_failure(cudaError_t err) {
    std::string text = cuda_error_message("cudaMalloc failed", err);
    if (err == cudaErrorMemoryAllocation) {
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            char numbers[96];
            std::snprintf(numbers, sizeof(numbers), " (%.1f of %.1f GiB free on this GPU",
                          static_cast<double>(free_bytes) / (1ULL << 30),
                          static_cast<double>(total_bytes) / (1ULL << 30));
            text += numbers;
        } else {
            text += " (";
        }
        text += "; is another process using it? nvidia-smi lists it)";
    }
    return text;
}

} // namespace

DeviceBuffer::DeviceBuffer(std::size_t size_bytes) : bytes(size_bytes) {
    if (bytes == 0) { return; }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) { throw std::runtime_error(device_allocation_failure(err)); }
    p = ptr;
}

DeviceBuffer::~DeviceBuffer() { free_device(p); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p), bytes(other.bytes) {
    other.p     = nullptr;
    other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    free_device(p);
    p     = other.p;
    bytes = other.bytes;

    other.p     = nullptr;
    other.bytes = 0;
    return *this;
}

void DeviceBuffer::fill(int byte_value) {
    if (bytes == 0) { return; }
    const cudaError_t err = cudaMemset(p, byte_value, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemset failed", err));
    }
    const cudaError_t settled = cudaStreamSynchronize(nullptr);
    if (settled != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("device fill synchronization failed", settled));
    }
}

void DeviceBuffer::copy_from_host(const void* source, std::size_t count, std::size_t byte_offset) {
    require_range(byte_offset, count, "host-to-device copy");
    if (count == 0) { return; }
    void* destination     = static_cast<std::uint8_t*>(p) + byte_offset;
    const cudaError_t err = cudaMemcpy(destination, source, count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy host-to-device failed", err));
    }
    // Pageable H2D copies may return before the device transfer completes. Settle the
    // default stream before callers submit consumers on non-blocking streams.
    const cudaError_t settled = cudaStreamSynchronize(nullptr);
    if (settled != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("host-to-device copy synchronization failed", settled));
    }
}

void DeviceBuffer::copy_to_host(void* destination, std::size_t count,
                                std::size_t byte_offset) const {
    require_range(byte_offset, count, "device-to-host copy");
    if (count == 0) { return; }
    const void* source    = static_cast<const std::uint8_t*>(p) + byte_offset;
    const cudaError_t err = cudaMemcpy(destination, source, count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy device-to-host failed", err));
    }
}

void DeviceBuffer::require_range(std::size_t byte_offset, std::size_t count,
                                 const char* operation) const {
    if (byte_offset > bytes || count > bytes - byte_offset) {
        throw std::out_of_range(std::string(operation) + " exceeds device buffer");
    }
}

DeviceArena::Scope::Scope(DeviceArena& arena) noexcept
    : arena_(&arena), saved_offset_(arena.off_) {}

DeviceArena::Scope::~Scope() noexcept {
    if (arena_ != nullptr && saved_offset_ <= arena_->off_) { arena_->off_ = saved_offset_; }
}

DeviceArena::Scope::Scope(Scope&& other) noexcept
    : arena_(other.arena_), saved_offset_(other.saved_offset_) {
    other.arena_ = nullptr;
}

DeviceArena::DeviceArena(std::size_t capacity_bytes, ZeroFill zero) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("DeviceArena capacity must be nonzero");
    }

    void* ptr       = nullptr;
    cudaError_t err = cudaMalloc(&ptr, capacity_bytes);
    if (err != cudaSuccess) { throw std::runtime_error(device_allocation_failure(err)); }
    if (zero == ZeroFill::Yes) {
        err = cudaMemset(ptr, 0, capacity_bytes);
        if (err == cudaSuccess) { err = cudaStreamSynchronize(nullptr); }
        if (err != cudaSuccess) {
            free_device(ptr);
            throw std::runtime_error(cuda_error_message("cudaMemset failed", err));
        }
    }

    base_ = ptr;
    cap_  = capacity_bytes;
    off_  = 0;
}

DeviceArena::DeviceArena(DeviceSpan storage)
    : base_(storage.data), cap_(storage.bytes), owns_(false) {
    if (base_ == nullptr || cap_ == 0) {
        throw std::invalid_argument("borrowed DeviceArena storage must be non-empty");
    }
}

DeviceArena::~DeviceArena() {
    if (owns_) { free_device(base_); }
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : base_(other.base_), cap_(other.cap_), off_(other.off_), peak_(other.peak_),
      owns_(other.owns_) {
    other.base_ = nullptr;
    other.cap_  = 0;
    other.off_  = 0;
    other.peak_ = 0;
    other.owns_ = true;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
    if (this == &other) { return *this; }

    if (owns_) { free_device(base_); }
    base_ = other.base_;
    cap_  = other.cap_;
    off_  = other.off_;
    peak_ = other.peak_;
    owns_ = other.owns_;

    other.base_ = nullptr;
    other.cap_  = 0;
    other.off_  = 0;
    other.peak_ = 0;
    other.owns_ = true;
    return *this;
}

DeviceSpan DeviceArena::alloc_bytes(std::size_t bytes, std::size_t align) {
    if (base_ == nullptr) { throw std::runtime_error("DeviceArena has no backing allocation"); }
    if (!is_power_of_two(align)) {
        throw std::invalid_argument("arena alignment must be a nonzero power of two");
    }
    if (bytes == 0) { throw std::invalid_argument("arena allocation must be nonzero"); }

    const std::uintptr_t base_addr           = reinterpret_cast<std::uintptr_t>(base_);
    const std::uintptr_t current_addr        = checked_add_uintptr(base_addr, off_);
    const std::uintptr_t aligned_addr        = align_up_addr(current_addr, align);
    const std::uintptr_t aligned_offset_addr = aligned_addr - base_addr;
    if (aligned_offset_addr > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("arena aligned offset overflows size_t");
    }
    const auto aligned_offset = static_cast<std::size_t>(aligned_offset_addr);
    if (bytes > std::numeric_limits<std::size_t>::max() - aligned_offset) {
        throw std::overflow_error("arena allocation end offset overflows size_t");
    }
    const std::size_t end = aligned_offset + bytes;
    if (end > cap_) { throw std::bad_alloc(); }

    auto* ptr = static_cast<unsigned char*>(base_) + aligned_offset;
    off_      = end;
    if (off_ > peak_) { peak_ = off_; }
    return DeviceSpan{ptr, bytes};
}

Tensor DeviceArena::alloc(DType dtype, std::initializer_list<std::int32_t> shape,
                          std::size_t align) {
    Tensor view(nullptr, dtype, shape);
    const DeviceSpan storage = alloc_bytes(view.bytes(), align);
    return Tensor(storage.data, dtype, shape);
}

DeviceArena::Scope DeviceArena::scope() noexcept { return Scope(*this); }

void DeviceArena::reset() noexcept { off_ = 0; }

void* DeviceArena::base() const noexcept { return base_; }

std::size_t DeviceArena::used() const noexcept { return off_; }

std::size_t DeviceArena::capacity() const noexcept { return cap_; }

std::size_t DeviceArena::peak_used() const noexcept { return peak_; }

void DeviceArena::reset_peak() noexcept { peak_ = off_; }

PinnedHostBuffer::PinnedHostBuffer(std::size_t size_bytes, HostMapping mapping,
                                   PinnedHostPages pages) {
    if (size_bytes == 0) { throw std::invalid_argument("PinnedHostBuffer size must be nonzero"); }

    if (pages == PinnedHostPages::Huge && registered_pointers_are_device_pointers()) {
        if (size_bytes > std::numeric_limits<std::size_t>::max() - kPinnedHugePageBytes) {
            throw std::overflow_error("PinnedHostBuffer size overflow");
        }
        const std::size_t reserved =
            (size_bytes + kPinnedHugePageBytes - 1) / kPinnedHugePageBytes * kPinnedHugePageBytes;
        void* ptr = std::aligned_alloc(kPinnedHugePageBytes, reserved);
        if (ptr == nullptr) { throw std::bad_alloc(); }
        // Populate every page before registration, so the pinned mapping is built once over its
        // final (huge) pages rather than faulted in piecewise by the registration itself.
        (void)madvise(ptr, reserved, MADV_HUGEPAGE);
        if (madvise(ptr, reserved, MADV_POPULATE_WRITE) != 0) { std::memset(ptr, 0, reserved); }
        const unsigned flags =
            cudaHostRegisterPortable | (mapping == HostMapping::Yes ? cudaHostRegisterMapped : 0U);
        const cudaError_t err = cudaHostRegister(ptr, reserved, flags);
        if (err != cudaSuccess) {
            std::free(ptr);
            throw std::runtime_error(cuda_error_message("cudaHostRegister failed", err));
        }
        data_       = ptr;
        size_       = size_bytes;
        mapped_     = mapping == HostMapping::Yes;
        registered_ = true;
        return;
    }

    void* ptr = nullptr;
    if (mapping == HostMapping::Yes) {
        const cudaError_t err =
            cudaHostAlloc(&ptr, size_bytes, cudaHostAllocMapped | cudaHostAllocPortable);
        if (err != cudaSuccess) {
            throw std::runtime_error(cuda_error_message("cudaHostAlloc (mapped) failed", err));
        }
    } else {
        const cudaError_t err = cudaMallocHost(&ptr, size_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(cuda_error_message("cudaMallocHost failed", err));
        }
    }

    data_   = ptr;
    size_   = size_bytes;
    mapped_ = mapping == HostMapping::Yes;
}

PinnedHostBuffer::~PinnedHostBuffer() { free_pinned(data_, registered_); }

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_), mapped_(other.mapped_),
      registered_(other.registered_) {
    other.data_       = nullptr;
    other.size_       = 0;
    other.mapped_     = false;
    other.registered_ = false;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    free_pinned(data_, registered_);
    data_       = other.data_;
    size_       = other.size_;
    mapped_     = other.mapped_;
    registered_ = other.registered_;

    other.data_       = nullptr;
    other.size_       = 0;
    other.mapped_     = false;
    other.registered_ = false;
    return *this;
}

PageableHostBuffer::PageableHostBuffer(std::size_t size_bytes) {
    if (size_bytes == 0) { throw std::invalid_argument("PageableHostBuffer size must be nonzero"); }
    if (size_bytes > std::numeric_limits<std::size_t>::max() - kPinnedHugePageBytes) {
        throw std::overflow_error("PageableHostBuffer size overflow");
    }
    const std::size_t reserved =
        (size_bytes + kPinnedHugePageBytes - 1) / kPinnedHugePageBytes * kPinnedHugePageBytes;
    data_ = std::aligned_alloc(kPinnedHugePageBytes, reserved);
    if (data_ == nullptr) { throw std::bad_alloc(); }
    (void)madvise(data_, reserved, MADV_HUGEPAGE);
    size_ = size_bytes;
}

PageableHostBuffer::~PageableHostBuffer() { std::free(data_); }

void* PinnedHostBuffer::data() const noexcept { return data_; }

std::size_t PinnedHostBuffer::size() const noexcept { return size_; }

void* PinnedHostBuffer::device_data() const {
    if (!mapped_ || data_ == nullptr) {
        throw std::logic_error("PinnedHostBuffer is not mapped into the device address space");
    }
    void* device          = nullptr;
    const cudaError_t err = cudaHostGetDevicePointer(&device, data_, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaHostGetDevicePointer failed", err));
    }
    return device;
}

} // namespace ninfer
