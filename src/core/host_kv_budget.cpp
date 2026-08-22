#include "core/host_kv_budget.h"

#include "core/device.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace ninfer {

namespace {

constexpr std::size_t kSlabAlignment = 64;

std::size_t align_up(std::size_t bytes, std::size_t alignment) {
    return (bytes + (alignment - 1)) & ~(alignment - 1);
}

}  // namespace

HostKvBudget::HostKvBudget(std::size_t budget_bytes) : budget_bytes_(budget_bytes) {
    if (budget_bytes_ == 0) {
        throw std::invalid_argument("host KV budget requires a non-zero byte budget");
    }

    // One contiguous pinned registration rather than per-entry allocations:
    // fewer page-table entries and a single failure point at startup. A
    // failure must throw so the server reports it through its normal startup
    // error path; CUDA_CHECK would abort the process instead.
    void* backing = nullptr;
    const cudaError_t err = cudaHostAlloc(&backing, budget_bytes_, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("host KV budget: cudaHostAlloc failed (") +
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
    free_.push_back({0, budget_bytes_});  // the whole budget is free at start
    backing_ = guard.pointer;
    guard.pointer = nullptr;  // ownership moved into the member
}

HostKvBudget::~HostKvBudget() {
    if (backing_ != nullptr) { (void)cudaFreeHost(backing_); }
}

std::size_t HostKvBudget::free_total() const noexcept {
    std::size_t total = 0;
    for (const FreeRange& r : free_) { total += r.bytes; }
    return total;
}

std::size_t HostKvBudget::used_bytes() const noexcept {
    return budget_bytes_ - free_total();
}

std::size_t HostKvBudget::largest_free_range() const noexcept {
    std::size_t largest = 0;
    for (const FreeRange& r : free_) { largest = std::max(largest, r.bytes); }
    return largest;
}

std::size_t HostKvBudget::slab_offset(const HostKvSlab* slab) const noexcept {
    return static_cast<const unsigned char*>(slab->base()) -
           static_cast<const unsigned char*>(backing_);
}

bool HostKvBudget::can_satisfy(std::size_t bytes, std::size_t protected_offset,
                               std::size_t protected_bytes) const {
    // acquire() needs one contiguous range, not a sum of bytes. The only
    // immovable region is the protected entry; releasing every other region
    // leaves the space before it and the space after it. The largest contiguous
    // span obtainable is the larger of the two (the whole budget when nothing
    // is protected). If the aligned request exceeds it, no eviction can make
    // room, so the caller should not evict at all.
    //
    // Wrong input must yield a conservative answer, not a permissive one: an
    // out-of-range protected region would underflow the span arithmetic below
    // (wrapping to a near-SIZE_MAX span), and a request larger than the budget
    // can never fit (the alignment only grows it). Reject both before the
    // arithmetic.
    if (bytes > budget_bytes_) { return false; }
    if (protected_offset > budget_bytes_ ||
        protected_bytes > budget_bytes_ - protected_offset) {
        return false;
    }
    const std::size_t needed = align_up(bytes, kSlabAlignment);
    const std::size_t max_span =
        (protected_bytes == 0)
            ? budget_bytes_
            : std::max(protected_offset,
                       budget_bytes_ - protected_offset - protected_bytes);
    return needed <= max_span;
}

HostKvSlab* HostKvBudget::acquire(std::size_t bytes) {
    const std::size_t needed = align_up(bytes, kSlabAlignment);
    if (needed == 0 || needed > budget_bytes_) { return nullptr; }

    // First-fit over the coalesced free ranges: take the first range that
    // fits, keep its tail as a (possibly empty) free range.
    for (std::size_t i = 0; i < free_.size(); ++i) {
        FreeRange& r = free_[i];
        if (r.bytes < needed) { continue; }
        const std::size_t offset = r.offset;
        auto* base = static_cast<unsigned char*>(backing_) + offset;
        // Allocate the view (and grow the view pool) BEFORE carving the range,
        // so a bad_alloc leaves the free range intact instead of leaking the
        // region. Grow geometrically (not by exactly one) so the pool's
        // amortized cost stays O(1) per park; an exact reserve would make every
        // acquire reallocate the whole array, O(n) in total parks.
        if (views_.size() == views_.capacity()) {
            views_.reserve(views_.size() * 2 + 16);
        }
        auto view = std::make_unique<HostKvSlab>(base, needed);
        const std::size_t remainder = r.bytes - needed;
        if (remainder == 0) {
            free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(i));
        } else {
            r.offset += needed;  // the remainder starts where the taken region ends
            r.bytes    = remainder;
        }
        views_.push_back(std::move(view));  // cannot throw: capacity reserved above
        return views_.back().get();
    }
    return nullptr;  // no single free range fits (exhaustion or fragmentation)
}

void HostKvBudget::release(HostKvSlab* slab) noexcept {
    if (slab == nullptr) { return; }
    const std::size_t offset =
        static_cast<const unsigned char*>(slab->base()) -
        static_cast<const unsigned char*>(backing_);
    const std::size_t bytes = slab->bytes();
    // A double-release would re-insert a range that is already free (or
    // overlaps one); under a coalescing list that silently doubles the free
    // bytes and hands overlapping regions to two live entries. Guard it.
    for (const FreeRange& r : free_) {
        const bool overlaps = offset < r.offset + r.bytes && r.offset < offset + bytes;
        if (overlaps) {
            // A double-release would re-insert a range that is already free (or
            // overlaps one); under a coalescing list that silently doubles the
            // free bytes and hands overlapping regions to two live entries. This
            // is a memory-safety invariant, so fail hard in every build (a plain
            // assert is compiled out under NDEBUG, which is the production build).
            std::fprintf(stderr, "host KV budget: double-release of an overlapping range\n");
            std::abort();
        }
    }
    slab->set_filled(0);

    // Insert [offset, offset+bytes) in order, then coalesce with the adjacent
    // ranges (the previous range ends at offset, the next starts at
    // offset+bytes).
    auto it = std::lower_bound(free_.begin(), free_.end(), offset,
                               [](const FreeRange& r, std::size_t off) { return r.offset < off; });
    FreeRange inserted{offset, bytes};
    if (it != free_.end() && it->offset == offset + bytes) {  // coalesce with next
        inserted.bytes += it->bytes;
        it = free_.erase(it);
    }
    if (it != free_.begin()) {  // coalesce with previous
        auto prev = std::prev(it);
        if (prev->offset + prev->bytes == offset) {
            prev->bytes += inserted.bytes;
            return;  // merged into the previous range; nothing to insert
        }
    }
    free_.insert(it, inserted);
}

}  // namespace ninfer
