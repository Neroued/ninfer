#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <span>
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
    explicit MetadataBudget(std::size_t limit) : limit_(limit) {}

    [[nodiscard]] std::size_t limit() const noexcept { return limit_; }

    [[nodiscard]] std::size_t allocated() const noexcept { return allocated_; }

    [[nodiscard]] std::size_t reserved() const noexcept { return reserved_; }

    [[nodiscard]] std::size_t available() const noexcept { return limit_ - allocated_ - reserved_; }

    void acquire(std::size_t bytes) {
        if (bytes > available()) throw MetadataExhausted(bytes);
        allocated_ += bytes;
    }

    void release(std::size_t bytes) noexcept {
        if (bytes > allocated_) std::terminate();
        allocated_ -= bytes;
    }

    void reserve(std::size_t bytes) {
        if (bytes > available()) throw MetadataExhausted(bytes);
        reserved_ += bytes;
    }

    void consume(std::size_t bytes) noexcept {
        if (bytes > reserved_) std::terminate();
        reserved_ -= bytes;
        allocated_ += bytes;
    }

    void cancel(std::size_t bytes) noexcept {
        if (bytes > reserved_) std::terminate();
        reserved_ -= bytes;
    }

private:
    std::size_t limit_;
    std::size_t allocated_ = 0;
    std::size_t reserved_  = 0;
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

// Stable indices and addresses. Payload segments are materialized during admission, within the
// shared metadata budget; publication never grows this store.
template <class T, std::uint32_t SegmentSize = 256>
class DirectorySlots {
public:
    DirectorySlots(std::uint32_t limit, MetadataBudget& budget) : limit_(limit), budget_(&budget) {}

    ~DirectorySlots() { budget_->release(storage_bytes()); }

    DirectorySlots(const DirectorySlots&)            = delete;
    DirectorySlots& operator=(const DirectorySlots&) = delete;

    [[nodiscard]] std::uint32_t limit() const noexcept { return limit_; }

    [[nodiscard]] std::uint32_t size() const noexcept { return size_; }

    [[nodiscard]] std::size_t storage_bytes() const noexcept {
        return payload_bytes_ + segments_.capacity() * sizeof(std::unique_ptr<T[]>);
    }

    [[nodiscard]] std::size_t growth_bytes(std::uint32_t minimum_size) const {
        if (minimum_size > limit_) throw std::runtime_error("Context descriptor limit exhausted");
        if (minimum_size <= size_) return 0;
        const auto segments = (std::uint64_t(minimum_size) + SegmentSize - 1) / SegmentSize;
        const auto target   = std::min<std::uint64_t>(limit_, segments * SegmentSize);
        return (target - size_) * sizeof(T) +
               (segments > segments_.capacity() ? segments : 0) * sizeof(std::unique_ptr<T[]>);
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
            const auto before = segments_.capacity() * sizeof(std::unique_ptr<T[]>);
            const auto bytes  = segments * sizeof(std::unique_ptr<T[]>);
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
                segments_.push_back(std::make_unique<T[]>(count));
            } catch (...) {
                budget_->release(bytes);
                throw;
            }
            payload_bytes_ += bytes;
            size_ += count;
        }
    }

    [[nodiscard]] bool is_allocated(std::uint32_t index) const noexcept {
        return index < size_ && segments_[index / SegmentSize] != nullptr;
    }

    void ensure(std::uint32_t index) {
        if (index >= size_) {
            grow(index + 1);
            return;
        }
        auto& segment = segments_[index / SegmentSize];
        if (segment) return;
        const auto count = std::min(SegmentSize, limit_ - index / SegmentSize * SegmentSize);
        const auto bytes = std::size_t(count) * sizeof(T);
        budget_->acquire(bytes);
        try {
            segment = std::make_unique<T[]>(count);
        } catch (...) {
            budget_->release(bytes);
            throw;
        }
        payload_bytes_ += bytes;
    }

    void release_segment(std::uint32_t first_index) noexcept {
        if (first_index % SegmentSize || first_index >= size_) std::terminate();
        auto& segment = segments_[first_index / SegmentSize];
        if (!segment) return;
        const auto count = std::min(SegmentSize, limit_ - first_index);
        const auto bytes = std::size_t(count) * sizeof(T);
        segment.reset();
        payload_bytes_ -= bytes;
        budget_->release(bytes);
    }

    [[nodiscard]] T& operator[](std::uint32_t index) noexcept {
        return segments_[index / SegmentSize][index % SegmentSize];
    }

    [[nodiscard]] const T& operator[](std::uint32_t index) const noexcept {
        return segments_[index / SegmentSize][index % SegmentSize];
    }

private:
    std::uint32_t limit_;
    std::uint32_t size_        = 0;
    std::size_t payload_bytes_ = 0;
    MetadataBudget* budget_;
    std::vector<std::unique_ptr<T[]>> segments_;
};

} // namespace ninfer::models::context
