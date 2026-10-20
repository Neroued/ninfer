#pragma once

#include <cstdint>
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
class CheckpointStore {
public:
    struct Slot {
        std::optional<Record> value;
        std::uint64_t generation = 1;
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
                if (!owner_->valid(handle_) || !owner_->slots_[handle_.index].pins)
                    std::terminate();
                --owner_->slots_[handle_.index].pins;
                owner_ = nullptr;
            }
        }
    private:
        Lease(CheckpointStore& owner, CheckpointHandle handle) : owner_(&owner), handle_(handle) {}

        CheckpointStore* owner_ = nullptr;
        CheckpointHandle handle_;
        friend class CheckpointStore;
    };

    explicit CheckpointStore(std::size_t capacity = 0) : slots_(capacity) {}

    [[nodiscard]] static std::size_t metadata_bytes(std::size_t capacity) {
        return sizeof(CheckpointStore) + capacity * sizeof(Slot);
    }

    CheckpointStore(const CheckpointStore&)            = delete;
    CheckpointStore& operator=(const CheckpointStore&) = delete;

    void resize(std::size_t capacity) {
        if (!slots_.empty()) throw std::logic_error("checkpoint capacity is fixed at startup");
        slots_.resize(capacity);
    }

    [[nodiscard]] auto begin() const noexcept { return slots_.begin(); }

    [[nodiscard]] auto end() const noexcept { return slots_.end(); }

    [[nodiscard]] std::size_t size() const noexcept { return slots_.size(); }

    // Semantic index lifetime; residency and leases do not change immutable checkpoint identity.
    [[nodiscard]] const std::uint64_t& revision() const noexcept { return revision_; }

    [[nodiscard]] const Slot& operator[](std::size_t index) const { return slots_.at(index); }

    [[nodiscard]] CheckpointHandle handle(std::uint32_t index) const {
        return {this, index, slots_.at(index).generation};
    }

    [[nodiscard]] bool valid(CheckpointHandle h) const noexcept {
        return h.owner == this && h.index < slots_.size() &&
               slots_[h.index].generation == h.generation && slots_[h.index].value.has_value();
    }

    [[nodiscard]] std::optional<CheckpointHandle> reserve() {
        for (std::uint32_t i = 0; i < slots_.size(); ++i) {
            auto& slot = slots_[i];
            if (!slot.value && !slot.reserved) {
                slot.reserved = true;
                return handle(i);
            }
        }
        return std::nullopt;
    }

    void cancel(CheckpointHandle h) noexcept {
        if (h.owner != this || h.index >= slots_.size()) std::terminate();
        auto& slot = slots_[h.index];
        if (slot.generation != h.generation || slot.value || slot.pins) std::terminate();
        slot.reserved = false;
        if (++slot.generation == 0) ++slot.generation;
    }

    void publish(CheckpointHandle h, Record record) {
        if (h.owner != this || h.index >= slots_.size())
            throw std::logic_error("invalid checkpoint reservation");
        auto& slot = slots_[h.index];
        if (slot.generation != h.generation || !slot.reserved || slot.value)
            throw std::logic_error("checkpoint reservation is stale");
        slot.value.emplace(std::move(record));
        slot.reserved = false;
        ++revision_;
    }

    [[nodiscard]] Record& at(CheckpointHandle h) {
        if (!valid(h)) throw std::logic_error("checkpoint handle is stale");
        return *slots_[h.index].value;
    }

    [[nodiscard]] const Record& at(CheckpointHandle h) const {
        if (!valid(h)) throw std::logic_error("checkpoint handle is stale");
        return *slots_[h.index].value;
    }

    [[nodiscard]] Lease lease(CheckpointHandle h) {
        (void)at(h);
        ++slots_[h.index].pins;
        return Lease(*this, h);
    }

    [[nodiscard]] bool erasable(CheckpointHandle h) const noexcept {
        return valid(h) && !slots_[h.index].pins;
    }

    bool erase(CheckpointHandle h) noexcept {
        if (!erasable(h)) return false;
        auto& slot = slots_[h.index];
        slot.value.reset();
        if (++slot.generation == 0) ++slot.generation;
        ++revision_;
        return true;
    }

    void clear() noexcept {
        ++revision_;
        for (std::uint32_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].pins) std::terminate();
            slots_[i].value.reset();
            slots_[i].reserved = false;
            if (++slots_[i].generation == 0) ++slots_[i].generation;
        }
    }
private:
    std::vector<Slot> slots_;
    std::uint64_t revision_ = 1;
};
} // namespace ninfer::models::context
