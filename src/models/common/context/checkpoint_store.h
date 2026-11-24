#pragma once

#include "models/common/context/directory/metadata.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <exception>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::context {
struct CheckpointHandle {
    const void* owner        = nullptr;
    std::uint32_t index      = 0;
    std::uint64_t generation = 0;

    friend bool operator==(CheckpointHandle, CheckpointHandle) noexcept = default;
};

// Records express retention purposes; leases keep their complete snapshot readable through
// binding or migration. Slot publication and retirement own the record references.
template <class Record>
class CheckpointStore : public MetadataCatalog {
public:
    struct Slot {
        std::optional<Record> value;
        std::uint64_t generation = 0;
        std::uint32_t pins       = 0;
        bool reserved            = false;
    };

    class Lease {
    public:
        Lease() noexcept = default;

        ~Lease() { reset(); }

        Lease(Lease&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), handle_(other.handle_) {}

        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                reset();
                owner_  = std::exchange(other.owner_, nullptr);
                handle_ = other.handle_;
            }
            return *this;
        }

        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;

        [[nodiscard]] CheckpointHandle handle() const noexcept { return handle_; }

        [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

        void reset() noexcept {
            if (owner_) {
                if (!owner_->valid(handle_) || !owner_->slot(handle_.index).pins) std::terminate();
                --owner_->slot(handle_.index).pins;
                owner_ = nullptr;
            }
        }
    private:
        Lease(CheckpointStore& owner, CheckpointHandle handle) : owner_(&owner), handle_(handle) {}

        CheckpointStore* owner_ = nullptr;
        CheckpointHandle handle_;
        friend class CheckpointStore;
    };

    class Reservation {
    public:
        Reservation() noexcept = default;

        ~Reservation() {
            if (owner_) owner_->cancel(handle_);
        }

        Reservation(Reservation&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), handle_(other.handle_) {}

        Reservation& operator=(Reservation&& other) noexcept {
            if (this != &other) {
                if (owner_) owner_->cancel(handle_);
                owner_  = std::exchange(other.owner_, nullptr);
                handle_ = other.handle_;
            }
            return *this;
        }

        Reservation(const Reservation&)            = delete;
        Reservation& operator=(const Reservation&) = delete;

        [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

        CheckpointHandle publish(Record record) {
            if (!owner_) throw std::logic_error("checkpoint reservation is empty");
            owner_->publish(handle_, std::move(record));
            owner_ = nullptr;
            return handle_;
        }
    private:
        Reservation(CheckpointStore& owner, CheckpointHandle handle)
            : owner_(&owner), handle_(handle) {}

        CheckpointStore* owner_ = nullptr;
        CheckpointHandle handle_;
        friend class CheckpointStore;
    };

    CheckpointStore() = default;

    CheckpointStore(std::size_t capacity, MetadataBudget& budget,
                    MetadataCapacity* index = nullptr) {
        configure(capacity, budget, index);
    }

    [[nodiscard]] static std::size_t metadata_bytes(std::size_t capacity) {
        return sizeof(CheckpointStore) + (capacity + 255) * sizeof(Slot) +
               (capacity + 510) / 256 * sizeof(std::unique_ptr<Slot[]>);
    }

    [[nodiscard]] static std::uint32_t capacity_for_metadata(std::size_t bytes) {
        return static_cast<std::uint32_t>(
            std::min<std::size_t>(UINT32_MAX - 1, bytes / sizeof(Slot)));
    }

    CheckpointStore(const CheckpointStore&)            = delete;
    CheckpointStore& operator=(const CheckpointStore&) = delete;

    void configure(std::size_t capacity, MetadataBudget& budget,
                   MetadataCapacity* index = nullptr) {
        if (slots_) throw std::logic_error("checkpoint metadata is already configured");
        slots_  = std::make_unique<DirectorySlots<Slot>>(capacity, budget);
        budget_ = &budget;
        index_  = index;
    }

    class Iterator {
    public:
        const Slot& operator*() const { return owner_->slot(index_); }

        Iterator& operator++() {
            ++index_;
            return *this;
        }

        bool operator==(const Iterator&) const = default;
    private:
        const CheckpointStore* owner_;
        std::size_t index_;

        Iterator(const CheckpointStore* owner, std::size_t index) : owner_(owner), index_(index) {}
        friend class CheckpointStore;
    };

    [[nodiscard]] Iterator begin() const noexcept { return {this, 0}; }

    [[nodiscard]] Iterator end() const noexcept { return {this, size()}; }

    [[nodiscard]] std::size_t size() const noexcept { return slots_ ? slots_->size() : 0; }

    [[nodiscard]] std::size_t capacity() const noexcept { return slots_ ? slots_->limit() : 0; }

    [[nodiscard]] std::size_t allocated_metadata_bytes() const noexcept {
        return sizeof(*this) + (slots_ ? slots_->storage_bytes() : 0);
    }

    // Semantic index lifetime; residency and leases do not change immutable checkpoint identity.
    [[nodiscard]] const std::uint64_t& revision() const noexcept { return revision_; }

    [[nodiscard]] const Slot& operator[](std::size_t index) const { return slot(index); }

    [[nodiscard]] CheckpointHandle handle(std::uint32_t index) const {
        return {this, index, slot(index).generation};
    }

    [[nodiscard]] bool valid(CheckpointHandle h) const noexcept {
        return h.owner == this && h.index < size() && slot(h.index).generation == h.generation &&
               slot(h.index).value.has_value();
    }

    [[nodiscard]] std::optional<CheckpointHandle> reserve() {
        for (std::uint32_t i = 0; i < size(); ++i) {
            if (!slots_->is_allocated(i)) continue;
            auto& candidate = slot(i);
            if (!candidate.value && !candidate.reserved) return reserve_slot(i);
        }
        auto index = static_cast<std::uint32_t>(size());
        for (std::uint32_t first = 0; first < size(); first += 256)
            if (!slots_->is_allocated(first)) {
                index = first;
                break;
            }
        const auto target =
            std::max<std::size_t>(size(), std::min(capacity(), std::size_t(index / 256 + 1) * 256));
        const auto index_bytes = index_ ? index_->metadata_requirement(target) : 0;
        if (index == capacity() || slots_->ensure_bytes(index) + index_bytes > budget_->available())
            return std::nullopt;
        if (index_) index_->reserve_metadata(target);
        slots_->ensure(index);
        return reserve_slot(index);
    }

    [[nodiscard]] Reservation reserve_owned() {
        const auto handle = reserve();
        return handle ? Reservation(*this, *handle) : Reservation{};
    }

    void cancel(CheckpointHandle h) noexcept {
        if (h.owner != this || h.index >= size()) std::terminate();
        auto& slot = this->slot(h.index);
        if (slot.generation != h.generation || slot.value || slot.pins) std::terminate();
        slot.reserved = false;
        reclaim_segment(h.index);
    }

    void publish(CheckpointHandle h, Record record) {
        if (h.owner != this || h.index >= size())
            throw std::logic_error("invalid checkpoint reservation");
        auto& slot = this->slot(h.index);
        if (slot.generation != h.generation || !slot.reserved || slot.value)
            throw std::logic_error("checkpoint reservation is stale");
        slot.value.emplace(std::move(record));
        slot.reserved = false;
        ++revision_;
    }

    [[nodiscard]] Record& at(CheckpointHandle h) {
        if (!valid(h)) throw std::logic_error("checkpoint handle is stale");
        return *slot(h.index).value;
    }

    [[nodiscard]] const Record& at(CheckpointHandle h) const {
        if (!valid(h)) throw std::logic_error("checkpoint handle is stale");
        return *slot(h.index).value;
    }

    [[nodiscard]] Lease lease(CheckpointHandle h) {
        (void)at(h);
        ++slot(h.index).pins;
        return Lease(*this, h);
    }

    [[nodiscard]] bool erasable(CheckpointHandle h) const noexcept {
        return valid(h) && !slot(h.index).pins;
    }

    [[nodiscard]] MetadataRelease retirement(CheckpointHandle h) const {
        if (!erasable(h)) throw std::logic_error("checkpoint retirement has no release authority");
        return {this, h.index, h.generation};
    }

    [[nodiscard]] std::size_t
    released_metadata(std::span<const MetadataRelease> releases) const override {
        std::size_t bytes = 0;
        for (std::uint32_t first = 0; first < size(); first += 256) {
            if (!slots_->is_allocated(first)) continue;
            const auto end = std::min<std::size_t>(size(), std::size_t(first) + 256);
            bool empty     = true;
            for (auto i = first; i < end; ++i) {
                const auto& value = slot(i);
                if (value.reserved || value.pins ||
                    (value.value &&
                     std::find(releases.begin(), releases.end(),
                               MetadataRelease{this, i, value.generation}) == releases.end())) {
                    empty = false;
                    break;
                }
            }
            if (empty) bytes += (end - first) * sizeof(Slot);
        }
        return bytes;
    }

    bool erase(CheckpointHandle h) noexcept {
        if (!erasable(h)) return false;
        auto& slot = this->slot(h.index);
        slot.value.reset();
        reclaim_segment(h.index);
        ++revision_;
        return true;
    }

    void clear() noexcept {
        ++revision_;
        for (std::uint32_t first = 0; first < size(); first += 256) {
            if (!slots_->is_allocated(first)) continue;
            const auto end = std::min<std::size_t>(size(), std::size_t(first) + 256);
            for (auto i = first; i < end; ++i)
                if (slot(i).pins) std::terminate();
            slots_->release_segment(first);
        }
    }
private:
    CheckpointHandle reserve_slot(std::uint32_t index) {
        if (next_generation_ == UINT64_MAX)
            throw std::overflow_error("checkpoint generation exhausted");
        auto& value      = slot(index);
        value.reserved   = true;
        value.generation = ++next_generation_;
        return handle(index);
    }

    void reclaim_segment(std::uint32_t index) noexcept {
        const auto first = index / 256 * 256;
        const auto end   = std::min<std::size_t>(size(), std::size_t(first) + 256);
        for (auto i = first; i < end; ++i)
            if (slot(i).value || slot(i).reserved || slot(i).pins) return;
        slots_->release_segment(first);
    }

    Slot& slot(std::size_t index) {
        if (index >= size())
            throw std::out_of_range("checkpoint descriptor is outside materialized metadata");
        return (*slots_)[index];
    }

    const Slot& slot(std::size_t index) const {
        if (index >= size())
            throw std::out_of_range("checkpoint descriptor is outside materialized metadata");
        static const Slot empty;
        return slots_->is_allocated(index) ? (*slots_)[index] : empty;
    }

    std::unique_ptr<DirectorySlots<Slot>> slots_;
    MetadataBudget* budget_        = nullptr;
    MetadataCapacity* index_       = nullptr;
    std::uint64_t next_generation_ = 0;
    std::uint64_t revision_        = 1;
};
} // namespace ninfer::models::context
