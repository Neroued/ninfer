#include "core/host_kv_arena.h"

#include "core/device.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace ninfer {

HostKvArena::HostKvArena(std::size_t slab_count, std::size_t slab_bytes)
    : slab_bytes_(slab_bytes) {
    if (slab_count == 0 || slab_bytes == 0) {
        throw std::invalid_argument("host KV arena requires a non-zero slab count and size");
    }

    const std::size_t total = slab_count * slab_bytes;
    if (total / slab_bytes != slab_count) {
        throw std::invalid_argument("host KV arena size overflows");
    }

    // One contiguous pinned registration rather than slab_count of them: fewer
    // page-table entries and a single failure point at startup. A failure must
    // throw so the server reports it through its normal startup error path;
    // CUDA_CHECK would abort the process instead.
    void* backing = nullptr;
    const cudaError_t err = cudaHostAlloc(&backing, total, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("host KV arena: cudaHostAlloc failed (") +
                                 cudaGetErrorString(err) + ")");
    }
    // The metadata allocations below can throw; until they complete the
    // pinned registration must be released on unwind, or a failed Engine
    // construction leaks a multi-gigabyte pinned registration.
    struct BackingGuard {
        void* pointer = nullptr;
        ~BackingGuard() {
            if (pointer != nullptr) { (void)cudaFreeHost(pointer); }
        }
    } guard{backing};
    slabs_.reserve(slab_count);
    free_.reserve(slab_count);
    for (std::size_t i = 0; i < slab_count; ++i) {
        auto* base = static_cast<unsigned char*>(backing) + i * slab_bytes;
        slabs_.push_back(std::make_unique<HostKvSlab>(base, slab_bytes));
        free_.push_back(slabs_.back().get());
    }
    backing_ = guard.pointer;
    guard.pointer = nullptr;  // ownership moved into the member
}

HostKvArena::~HostKvArena() {
    if (backing_ != nullptr) { (void)cudaFreeHost(backing_); }
}

HostKvSlab* HostKvArena::acquire() noexcept {
    if (free_.empty()) { return nullptr; }
    HostKvSlab* slab = free_.back();
    free_.pop_back();
    slab->set_filled(0);
    return slab;
}

void HostKvArena::release(HostKvSlab* slab) noexcept {
    if (slab == nullptr) { return; }
    slab->set_filled(0);
    free_.push_back(slab);
}

}  // namespace ninfer
