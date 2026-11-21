#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::context {

class MetadataCatalog;

struct MetadataRelease {
    const MetadataCatalog* catalog                           = nullptr;
    std::uint32_t index                                      = 0;
    std::uint64_t generation                                 = 0;
    friend bool operator==(MetadataRelease, MetadataRelease) = default;
};

class MetadataCatalog {
public:
    virtual ~MetadataCatalog() = default;
    [[nodiscard]] virtual std::size_t released_metadata(std::span<const MetadataRelease>) const = 0;
};

class MetadataExhausted : public std::runtime_error {
public:
    explicit MetadataExhausted(std::size_t bytes)
        : std::runtime_error("Context metadata budget exhausted"), required_bytes(bytes) {}

    std::size_t required_bytes;
};

class MetadataBudget {
public:
    explicit MetadataBudget(std::size_t limit, MetadataBudget* parent = nullptr)
        : limit_(limit), idle_limit_(limit), parent_(parent) {
        if (parent_) parent_->reserve(limit_);
    }

    ~MetadataBudget() {
        if (parent_) {
            if (allocated_ || reserved_) std::terminate();
            parent_->cancel(limit_);
        }
    }

    MetadataBudget(const MetadataBudget&)            = delete;
    MetadataBudget& operator=(const MetadataBudget&) = delete;

    [[nodiscard]] std::size_t limit() const noexcept { return limit_; }

    void grow(std::size_t limit) {
        if (limit <= limit_) {
            idle_limit_ = std::max(idle_limit_, limit);
            return;
        }
        if (parent_) parent_->reserve(limit - limit_);
        limit_      = limit;
        idle_limit_ = std::max(idle_limit_, limit);
    }

    // Live decision metadata pins its existing guarantee. Return excess reservation only
    // after all users of that guarantee retire, including in-flight counterfactual views.
    void trim_when_idle(std::size_t limit) noexcept {
        idle_limit_ = std::min(limit, limit_);
        trim_idle();
    }

    [[nodiscard]] std::size_t allocated() const noexcept { return allocated_; }

    [[nodiscard]] std::size_t reserved() const noexcept { return reserved_; }

    [[nodiscard]] std::size_t available() const noexcept { return limit_ - allocated_ - reserved_; }

    void acquire(std::size_t bytes) {
        if (bytes > available()) throw MetadataExhausted(bytes);
        if (parent_) parent_->consume(bytes);
        allocated_ += bytes;
    }

    void release(std::size_t bytes) noexcept {
        if (bytes > allocated_) std::terminate();
        allocated_ -= bytes;
        if (parent_) {
            parent_->release(bytes);
            parent_->reserve(bytes);
        }
        trim_idle();
    }

    void reserve(std::size_t bytes) {
        if (bytes > available()) throw MetadataExhausted(bytes);
        reserved_ += bytes;
    }

    void consume(std::size_t bytes) noexcept {
        if (bytes > reserved_) std::terminate();
        if (parent_) parent_->consume(bytes);
        reserved_ -= bytes;
        allocated_ += bytes;
    }

    void cancel(std::size_t bytes) noexcept {
        if (bytes > reserved_) std::terminate();
        reserved_ -= bytes;
        trim_idle();
    }

private:
    void trim_idle() noexcept {
        if (allocated_ || reserved_ || idle_limit_ >= limit_) return;
        if (parent_) parent_->cancel(limit_ - idle_limit_);
        limit_ = idle_limit_;
    }

    std::size_t limit_, idle_limit_;
    std::size_t allocated_  = 0;
    std::size_t reserved_   = 0;
    MetadataBudget* parent_ = nullptr;
};

// Accounts auxiliary containers owned by one directory. Payload segment storage is charged by
// DirectorySlots; a container retained for reuse remains allocated until explicitly released.
class MetadataAccount {
public:
    explicit MetadataAccount(MetadataBudget& budget) : budget_(&budget) {}

    ~MetadataAccount() { budget_->release(bytes_); }

    MetadataAccount(const MetadataAccount&)            = delete;
    MetadataAccount& operator=(const MetadataAccount&) = delete;

    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    [[nodiscard]] MetadataBudget& budget() const noexcept { return *budget_; }

    void acquire(std::size_t bytes) {
        budget_->acquire(bytes);
        bytes_ += bytes;
    }

    void release(std::size_t bytes) noexcept {
        if (bytes > bytes_) std::terminate();
        budget_->release(bytes);
        bytes_ -= bytes;
    }

    void transfer(std::size_t bytes, MetadataAccount& destination) noexcept {
        if (budget_ != destination.budget_ || bytes > bytes_) std::terminate();
        bytes_ -= bytes;
        destination.bytes_ += bytes;
    }

    template <class T>
    void reserve(std::vector<T>& values, std::size_t capacity) {
        if (capacity <= values.capacity()) return;
        const auto old_bytes = values.capacity() * sizeof(T);
        const auto bytes     = capacity * sizeof(T);
        acquire(bytes);
        try {
            values.reserve(capacity);
        } catch (...) {
            release(bytes);
            throw;
        }
        release(old_bytes);
    }
private:
    MetadataBudget* budget_;
    std::size_t bytes_ = 0;
};

class MetadataOwner {
public:
    virtual ~MetadataOwner()                                                   = default;
    virtual void release_metadata(std::span<const std::uint32_t> ids) noexcept = 0;
};

// Storage whose capacity must be prepared before its producer publishes new identities.
class MetadataCapacity {
public:
    virtual ~MetadataCapacity()                                               = default;
    [[nodiscard]] virtual std::size_t metadata_requirement(std::size_t) const = 0;
    virtual void reserve_metadata(std::size_t)                                = 0;

    virtual void trim_metadata(std::size_t) noexcept {}
};

// The index prepares reusable evaluation storage. Extra simultaneous evaluations own separately
// charged buffers; all evaluations must retire before the index. Idle scratch is allocated
// metadata, not an in-flight reservation.
class MetadataScratch {
public:
    explicit MetadataScratch(MetadataBudget& budget) : budget_(&budget) {}

    ~MetadataScratch() {
        if (readers_) std::terminate();
        if (data_) budget_->release(capacity_);
    }

    MetadataScratch(const MetadataScratch&)            = delete;
    MetadataScratch& operator=(const MetadataScratch&) = delete;

    [[nodiscard]] std::size_t growth_bytes(std::size_t bytes) const noexcept {
        return bytes > capacity_ ? bytes : 0;
    }

    void reserve(std::size_t bytes) {
        if (bytes <= capacity_) return;
        budget_->acquire(bytes);
        std::unique_ptr<std::byte[]> next;
        try {
            next = std::make_unique<std::byte[]>(bytes);
        } catch (...) {
            budget_->release(bytes);
            throw;
        }
        if (data_) {
            data_.reset();
            budget_->release(capacity_);
        }
        data_     = std::move(next);
        capacity_ = bytes;
    }

    class Lease {
    public:
        Lease(MetadataScratch& owner, std::size_t bytes) : owner_(&owner), bytes_(bytes) {
            if (owner.data_ && bytes <= owner.capacity_) {
                bytes_ = owner.capacity_;
                data_  = std::move(owner.data_);
            } else {
                owner.budget_->acquire(bytes);
                try {
                    data_ = std::make_unique<std::byte[]>(bytes);
                } catch (...) {
                    owner.budget_->release(bytes);
                    throw;
                }
            }
            ++owner.readers_;
        }

        ~Lease() {
            if (!owner_) return;
            --owner_->readers_;
            if (!owner_->data_ && bytes_ == owner_->capacity_)
                owner_->data_ = std::move(data_);
            else {
                data_.reset();
                owner_->budget_->release(bytes_);
            }
        }

        Lease(Lease&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), bytes_(other.bytes_),
              data_(std::move(other.data_)) {}

        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;

        [[nodiscard]] std::byte* data() noexcept { return data_.get(); }
    private:
        MetadataScratch* owner_;
        std::size_t bytes_;
        std::unique_ptr<std::byte[]> data_;
    };

    [[nodiscard]] Lease acquire(std::size_t bytes) { return Lease(*this, bytes); }
private:
    MetadataBudget* budget_;
    std::size_t capacity_ = 0, readers_ = 0;
    std::unique_ptr<std::byte[]> data_;
};

class MetadataReservation {
public:
    MetadataReservation() noexcept = default;

    MetadataReservation(MetadataOwner& owner, std::vector<std::uint32_t> ids) noexcept
        : owner_(&owner), ids_(std::move(ids)) {}

    ~MetadataReservation() { reset(); }

    MetadataReservation(MetadataReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), ids_(std::move(other.ids_)) {}

    MetadataReservation& operator=(MetadataReservation&& other) noexcept {
        if (this != &other) {
            reset();
            owner_ = std::exchange(other.owner_, nullptr);
            ids_   = std::move(other.ids_);
        }
        return *this;
    }

    MetadataReservation(const MetadataReservation&)            = delete;
    MetadataReservation& operator=(const MetadataReservation&) = delete;

    [[nodiscard]] MetadataOwner* owner() const noexcept { return owner_; }

    [[nodiscard]] std::size_t remaining() const noexcept { return ids_.size(); }

    [[nodiscard]] std::uint32_t last() const { return ids_.at(ids_.size() - 1); }

    void prepare_returns(std::size_t count) { ids_.reserve(ids_.size() + count); }

    [[nodiscard]] std::uint32_t take(MetadataOwner& owner) {
        if (owner_ != &owner || ids_.empty())
            throw std::logic_error("Context metadata does not belong to this reservation");
        const auto result = ids_.back();
        ids_.pop_back();
        return result;
    }

    void restore(MetadataOwner& owner, std::uint32_t id) {
        if (owner_ != &owner || ids_.size() == ids_.capacity())
            throw std::logic_error("Context metadata reservation cannot accept this descriptor");
        ids_.push_back(id);
    }

    [[nodiscard]] MetadataReservation split(std::size_t count) {
        count = std::min(count, ids_.size());
        std::vector<std::uint32_t> ids(ids_.end() - count, ids_.end());
        ids_.resize(ids_.size() - count);
        return owner_ ? MetadataReservation(*owner_, std::move(ids)) : MetadataReservation{};
    }

    void shrink(std::size_t count) noexcept {
        if (count >= ids_.size()) return;
        if (owner_) owner_->release_metadata(std::span<const std::uint32_t>(ids_).subspan(count));
        ids_.resize(count);
    }

    void reset() noexcept {
        if (owner_) owner_->release_metadata(ids_);
        owner_ = nullptr;
        ids_.clear();
    }
private:
    MetadataOwner* owner_ = nullptr;
    std::vector<std::uint32_t> ids_;
};

// Only the shortage path collects segments the quote expects to reuse. Pricing never
// changes the directory's occupancy or the order of its free lists.
class ProtectedMetadataSegments {
public:
    void add(const void* directory, std::uint32_t segment) {
        const Key key{reinterpret_cast<std::uintptr_t>(directory), segment};
        if (entries_.empty() || entries_.back() != key) entries_.push_back(key);
    }

    void finish() {
        std::sort(entries_.begin(), entries_.end());
        entries_.erase(std::unique(entries_.begin(), entries_.end()), entries_.end());
    }

    [[nodiscard]] bool contains(const void* directory, std::uint32_t segment) const noexcept {
        return std::binary_search(entries_.begin(), entries_.end(),
                                  Key{reinterpret_cast<std::uintptr_t>(directory), segment});
    }
private:
    using Key = std::pair<std::uintptr_t, std::uint32_t>;
    std::vector<Key> entries_;
};

struct MetadataReclaim {
    std::size_t bytes    = 0;
    std::size_t segments = 0, examined = 0;

    void add(MetadataReclaim other) noexcept {
        bytes += other.bytes;
        segments += other.segments;
        examined += other.examined;
    }
};

// Stable indices and addresses. Payload segments are materialized during admission, within the
// shared metadata budget; publication never grows this store.
template <class T, std::uint32_t SegmentSize = 256, bool TrackOccupancy = false>
class DirectorySlots {
    struct Untracked {};

    struct Links {
        std::uint32_t occupied = 0, prev = UINT32_MAX, next = UINT32_MAX;
    };

    struct Segment {
        std::unique_ptr<T[]> data;
        [[no_unique_address]] std::conditional_t<TrackOccupancy, Links, Untracked> links;
    };
public:
    DirectorySlots(std::uint32_t limit, MetadataBudget& budget) : limit_(limit), budget_(&budget) {}

    ~DirectorySlots() { budget_->release(storage_bytes()); }

    DirectorySlots(const DirectorySlots&)            = delete;
    DirectorySlots& operator=(const DirectorySlots&) = delete;

    [[nodiscard]] std::uint32_t limit() const noexcept { return limit_; }

    [[nodiscard]] std::uint32_t size() const noexcept { return size_; }

    [[nodiscard]] std::size_t storage_bytes() const noexcept {
        return payload_bytes_ + segments_.capacity() * sizeof(Segment);
    }

    [[nodiscard]] std::size_t growth_bytes(std::uint32_t minimum_size) const {
        if (minimum_size > limit_) throw std::runtime_error("Context descriptor limit exhausted");
        if (minimum_size <= size_) return 0;
        const auto segments = (std::uint64_t(minimum_size) + SegmentSize - 1) / SegmentSize;
        const auto target   = std::min<std::uint64_t>(limit_, segments * SegmentSize);
        return (target - size_) * sizeof(T) +
               (segments > segments_.capacity() ? segments : 0) * sizeof(Segment);
    }

    [[nodiscard]] std::size_t ensure_bytes(std::uint32_t index) const {
        if (index >= size_) return growth_bytes(index + 1);
        if (is_allocated(index)) return 0;
        return std::size_t(std::min(SegmentSize, limit_ - index / SegmentSize * SegmentSize)) *
               sizeof(T);
    }

    void grow(std::uint32_t minimum_size) {
        if (minimum_size > limit_) throw std::runtime_error("Context metadata budget exhausted");
        const auto segments = (std::uint64_t(minimum_size) + SegmentSize - 1) / SegmentSize;
        if (segments > segments_.capacity()) {
            const auto before = segments_.capacity() * sizeof(Segment);
            const auto bytes  = segments * sizeof(Segment);
            budget_->acquire(bytes);
            try {
                segments_.reserve(segments);
            } catch (...) {
                budget_->release(bytes);
                throw;
            }
            budget_->release(before);
        }
        while (size_ < minimum_size) {
            const auto count = std::min(SegmentSize, limit_ - size_);
            const auto bytes = std::size_t(count) * sizeof(T);
            budget_->acquire(bytes);
            try {
                segments_.push_back({std::make_unique<T[]>(count), {}});
                if constexpr (TrackOccupancy) link(size_ / SegmentSize, empty_);
            } catch (...) {
                budget_->release(bytes);
                throw;
            }
            payload_bytes_ += bytes;
            size_ += count;
        }
    }

    [[nodiscard]] bool is_allocated(std::uint32_t index) const noexcept {
        return index < size_ && segments_[index / SegmentSize].data != nullptr;
    }

    void ensure(std::uint32_t index) {
        if (index >= size_) {
            grow(index + 1);
            return;
        }
        auto& segment = segments_[index / SegmentSize];
        if (segment.data) return;
        const auto count = std::min(SegmentSize, limit_ - index / SegmentSize * SegmentSize);
        const auto bytes = std::size_t(count) * sizeof(T);
        budget_->acquire(bytes);
        try {
            segment.data = std::make_unique<T[]>(count);
        } catch (...) {
            budget_->release(bytes);
            throw;
        }
        if constexpr (TrackOccupancy) {
            unlink(index / SegmentSize, missing_);
            link(index / SegmentSize, empty_);
        }
        payload_bytes_ += bytes;
    }

    static constexpr std::size_t segment_entry_bytes() noexcept { return sizeof(Segment); }

    [[nodiscard]] bool idle(std::uint32_t index) const noexcept
        requires(TrackOccupancy)
    {
        return is_allocated(index) && !segments_[index / SegmentSize].links.occupied;
    }

    void occupy(std::uint32_t index) noexcept
        requires(TrackOccupancy)
    {
        const auto s = index / SegmentSize;
        if (segments_[s].links.occupied++ == 0) unlink(s, empty_);
    }

    void vacate(std::uint32_t index) noexcept
        requires(TrackOccupancy)
    {
        const auto s = index / SegmentSize;
        if (!segments_[s].links.occupied) std::terminate();
        if (!--segments_[s].links.occupied) link(s, empty_);
    }

    void protect(std::uint32_t index, ProtectedMetadataSegments* pins) const
        requires(TrackOccupancy)
    {
        if (pins && idle(index)) pins->add(this, index / SegmentSize);
    }

    template <class ReleaseTail>
    MetadataReclaim reclaim(std::size_t needed, const ProtectedMetadataSegments* keep,
                            ReleaseTail release_tail) noexcept
        requires(TrackOccupancy)
    {
        MetadataReclaim result;
        for (auto s = empty_; s != UINT32_MAX && result.bytes < needed;) {
            const auto next = segments_[s].links.next;
            ++result.examined;
            if (!keep || !keep->contains(this, s)) {
                const auto before = budget_->available();
                const auto first  = s * SegmentSize;
                const auto count  = std::min(SegmentSize, size_ - first);
                for (std::uint32_t i = 0; i < count; ++i) release_tail((*this)[first + i]);
                release_segment(first);
                result.bytes += budget_->available() - before;
                ++result.segments;
            }
            s = next;
        }
        return result;
    }

    // Count only idle segments and segments explicitly named by projected releases.
    template <class TailBytes>
    [[nodiscard]] std::size_t reclaimable_bytes(std::vector<std::uint32_t> released,
                                                TailBytes tail_bytes) const
        requires(TrackOccupancy)
    {
        const auto bytes = [&](std::uint32_t s) {
            const auto first  = s * SegmentSize;
            const auto count  = std::min(SegmentSize, size_ - first);
            std::size_t total = count * sizeof(T);
            for (std::uint32_t i = 0; i < count; ++i) total += tail_bytes((*this)[first + i]);
            return total;
        };
        std::size_t total = 0;
        for (auto s = empty_; s != UINT32_MAX; s = segments_[s].links.next) total += bytes(s);
        std::sort(released.begin(), released.end());
        released.erase(std::unique(released.begin(), released.end()), released.end());
        for (std::size_t i = 0; i < released.size();) {
            const auto s     = released[i] / SegmentSize;
            const auto begin = i++;
            while (i < released.size() && released[i] / SegmentSize == s) ++i;
            if (segments_[s].links.occupied == i - begin) total += bytes(s);
        }
        return total;
    }

    [[nodiscard]] std::uint32_t first_missing() const noexcept
        requires(TrackOccupancy)
    {
        return missing_ == UINT32_MAX ? size_ : missing_ * SegmentSize;
    }

    [[nodiscard]] std::uint32_t next_missing(std::uint32_t first) const noexcept
        requires(TrackOccupancy)
    {
        const auto next = segments_[first / SegmentSize].links.next;
        return next == UINT32_MAX ? size_ : next * SegmentSize;
    }

    void release_segment(std::uint32_t first_index) noexcept {
        if (first_index % SegmentSize || first_index >= size_) std::terminate();
        auto& segment = segments_[first_index / SegmentSize];
        if (!segment.data) return;
        const auto count = std::min(SegmentSize, limit_ - first_index);
        const auto bytes = std::size_t(count) * sizeof(T);
        if constexpr (TrackOccupancy) {
            if (!idle(first_index)) std::terminate();
            unlink(first_index / SegmentSize, empty_);
            link(first_index / SegmentSize, missing_);
        }
        segment.data.reset();
        payload_bytes_ -= bytes;
        budget_->release(bytes);
    }

    [[nodiscard]] T& operator[](std::uint32_t index) noexcept {
        return segments_[index / SegmentSize].data[index % SegmentSize];
    }

    [[nodiscard]] const T& operator[](std::uint32_t index) const noexcept {
        return segments_[index / SegmentSize].data[index % SegmentSize];
    }

private:
    std::uint32_t limit_;
    std::uint32_t size_        = 0;
    std::size_t payload_bytes_ = 0;
    MetadataBudget* budget_;
    std::vector<Segment> segments_;
    std::uint32_t empty_ = UINT32_MAX, missing_ = UINT32_MAX;

    void link(std::uint32_t index, std::uint32_t& head) noexcept
        requires(TrackOccupancy)
    {
        auto& node = segments_[index].links;
        node.prev  = UINT32_MAX;
        node.next  = head;
        if (head != UINT32_MAX) segments_[head].links.prev = index;
        head = index;
    }

    void unlink(std::uint32_t index, std::uint32_t& head) noexcept
        requires(TrackOccupancy)
    {
        const auto& node = segments_[index].links;
        if (node.prev == UINT32_MAX)
            head = node.next;
        else
            segments_[node.prev].links.next = node.next;
        if (node.next != UINT32_MAX) segments_[node.next].links.prev = node.prev;
    }
};

template <class T>
using ReclaimableSlots = DirectorySlots<T, 256, true>;

} // namespace ninfer::models::context
