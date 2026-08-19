#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer {

// One pinned host buffer able to hold a parked sequence's paged KV bytes.
// Slabs are fixed-size and reused; a slab is either free or owned by one parked
// sequence. Pinned memory is required: pageable host transfers measured ~3x
// slower and cannot overlap asynchronously.
class HostKvSlab {
public:
    HostKvSlab(void* base, std::size_t bytes) noexcept : base_(base), bytes_(bytes) {}

    [[nodiscard]] void* base() const noexcept { return base_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    // Bytes actually written by the last park, so restore copies no more than
    // was stored.
    [[nodiscard]] std::size_t filled() const noexcept { return filled_; }
    void set_filled(std::size_t bytes) noexcept { filled_ = bytes; }

private:
    void* base_        = nullptr;
    std::size_t bytes_ = 0;
    std::size_t filled_ = 0;
};

// Fixed pool of pinned slabs, allocated once at startup. Allocating pinned
// memory on the hot path would stall; the arena trades resident RAM for
// predictable park/restore latency.
class HostKvArena {
public:
    // Allocates slab_count buffers of slab_bytes each. Throws if pinned
    // allocation fails, so a too-large request fails at startup, not mid-serve.
    HostKvArena(std::size_t slab_count, std::size_t slab_bytes);
    ~HostKvArena();

    HostKvArena(const HostKvArena&)            = delete;
    HostKvArena& operator=(const HostKvArena&) = delete;
    HostKvArena(HostKvArena&&)                 = delete;
    HostKvArena& operator=(HostKvArena&&)      = delete;

    [[nodiscard]] std::size_t slab_count() const noexcept { return slabs_.size(); }
    [[nodiscard]] std::size_t slab_bytes() const noexcept { return slab_bytes_; }
    [[nodiscard]] std::size_t free_slabs() const noexcept { return free_.size(); }

    // Returns nullptr when every slab is in use; callers fall back to discarding
    // the KV (today's behaviour) rather than failing the request.
    [[nodiscard]] HostKvSlab* acquire() noexcept;
    void release(HostKvSlab* slab) noexcept;

private:
    void* backing_          = nullptr;
    std::size_t slab_bytes_ = 0;
    std::vector<std::unique_ptr<HostKvSlab>> slabs_;
    std::vector<HostKvSlab*> free_;
};

}  // namespace ninfer
