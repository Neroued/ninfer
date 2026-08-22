#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer {

// One pinned host buffer able to hold a parked sequence's paged KV bytes.
// A slab is a variable-size view over the budget; it is either free (its range
// is in the allocator's free list) or owned by one parked sequence. Pinned
// memory is required: pageable host transfers measured ~3x slower and cannot
// overlap asynchronously.
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

// Budget-bounded pool of pinned host memory. One contiguous cudaHostAlloc of
// budget_bytes, carved into variable-size regions by a coalescing free-range
// list. Each parked sequence takes only the bytes it needs (its real page
// count), so a small session no longer occupies a full max-size slab.
class HostKvBudget {
public:
    // Allocates one pinned buffer of budget_bytes. Throws if the allocation
    // fails or the budget is zero, so a too-large request fails at startup,
    // not mid-serve.
    explicit HostKvBudget(std::size_t budget_bytes);
    ~HostKvBudget();

    HostKvBudget(const HostKvBudget&)            = delete;
    HostKvBudget& operator=(const HostKvBudget&) = delete;
    HostKvBudget(HostKvBudget&&)                 = delete;
    HostKvBudget& operator=(HostKvBudget&&)      = delete;

    [[nodiscard]] std::size_t budget_bytes() const noexcept { return budget_bytes_; }
    // Bytes not currently handed out (sum of the free ranges).
    [[nodiscard]] std::size_t used_bytes() const noexcept;
    // Largest single contiguous free range. Exposes fragmentation so a
    // fit-driven over-eviction (total free bytes suffice but no single range
    // fits) is diagnosable.
    [[nodiscard]] std::size_t largest_free_range() const noexcept;
    // Offset of a slab handed out by acquire() within the budget's backing.
    [[nodiscard]] std::size_t slab_offset(const HostKvSlab* slab) const noexcept;
    // True when acquire(bytes) can ever succeed if every region except
    // [protected_offset, protected_offset + protected_bytes) were returned to
    // the free list. acquire() needs one CONTIGUOUS range, not a sum of bytes:
    // a protected entry in the middle of the budget splits the address space,
    // so the largest span obtainable is the larger of the space before it and
    // the space after it (the whole budget when nothing is protected). Pass
    // (0, 0) for "no protected entry"; a zero-byte region does not split the
    // address space, so (offset, 0) also answers the whole budget. Out-of-range
    // input (the protected region past the end of the budget, or a request
    // larger than the budget) answers false rather than underflowing. If the
    // aligned request exceeds the span, no eviction can make room, so the
    // caller bails out rather than evict the cache for a park that cannot
    // happen.
    [[nodiscard]] bool can_satisfy(std::size_t bytes, std::size_t protected_offset,
                                   std::size_t protected_bytes) const;

    // Returns a slab of exactly ceil(bytes/64)*64 bytes, or nullptr when no
    // free range can satisfy it. The slab view is owned for the budget's
    // lifetime (never freed while the cache lives), so the pointer stays valid
    // for the parked entry and any in-flight park. May throw std::bad_alloc
    // when the (tiny) view object cannot be allocated.
    [[nodiscard]] HostKvSlab* acquire(std::size_t bytes);
    // Returns the slab's range to the free list, coalescing with the adjacent
    // free ranges. The view object itself stays in the pool (not recycled).
    void release(HostKvSlab* slab) noexcept;

private:
    // Sum of the free ranges (the budget minus what is handed out).
    [[nodiscard]] std::size_t free_total() const noexcept;

    void* backing_        = nullptr;
    std::size_t budget_bytes_ = 0;
    struct FreeRange {
        std::size_t offset;
        std::size_t bytes;
    };
    std::vector<FreeRange> free_;  // sorted by offset, coalesced (no adjacent ranges)
    // Lifetime-owned slab views, one per acquire() and never recycled: the
    // pool grows without bound (~40 B per park, append-only). At the observed
    // park rate that is a few MB over a year of uptime, so it is not reclaimed;
    // the trade is a fixed-size HostKvSlab (no setter) for an unbounded pool.
    std::vector<std::unique_ptr<HostKvSlab>> views_;
};

}  // namespace ninfer
