#pragma once

#include "models/common/context/checkpoint_store.h"
#include "models/common/context/directory/records.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::context {

// One synchronous inventory of physical objects. An object contributes only when all its
// checkpoint holders are removed and Native has proved that no external holder blocks release.
// It owns neither checkpoint content nor store leases.
template <class Resources, class Handle = CheckpointHandle>
class ReleaseInventory {
public:
    struct Object {
        std::vector<Handle> holders;
        Resources released;
    };

    struct Work {
        std::uint64_t probes = 0, owner_visits = 0;
    };

    explicit ReleaseInventory(std::size_t capacity)
        : handles_(capacity), objects_by_handle_(capacity) {}

    void include(Handle handle) { handles_.at(handle.index) = handle; }

    bool contains(Handle handle) const {
        return handle.index < handles_.size() && handles_[handle.index] == handle;
    }

    std::size_t capacity() const { return handles_.size(); }

    std::uint32_t add(std::vector<Handle> holders, Resources released) {
        std::sort(holders.begin(), holders.end(), [](auto a, auto b) { return a.index < b.index; });
        holders.erase(std::unique(holders.begin(), holders.end()), holders.end());
        if (holders.empty()) throw std::logic_error("physical release has no checkpoint holders");
        const auto index = static_cast<std::uint32_t>(objects_.size());
        for (const auto handle : holders) {
            if (!contains(handle)) throw std::logic_error("physical release has a foreign holder");
            objects_by_handle_[handle.index].push_back(index);
        }
        objects_.push_back({std::move(holders), std::move(released)});
        return index;
    }

    const std::vector<Object>& objects() const { return objects_; }

    const Work& work() const { return work_; }

    class Selection {
    public:
        explicit Selection(const ReleaseInventory& inventory)
            : inventory_(&inventory), selected_(inventory.capacity()),
              remaining_(inventory.objects_.size()) {
            for (std::size_t i = 0; i < remaining_.size(); ++i)
                remaining_[i] = inventory.objects_[i].holders.size();
        }

        class Probe {
        public:
            Probe(Selection& selection, std::span<const Handle> additions)
                : selection_(&selection), size_(selection.handles_.size()),
                  freed_(selection.freed_.size()) {
                ++selection.inventory_->work_.probes;
                for (const auto handle : additions) selection.add(handle);
            }

            Probe(const Probe&)            = delete;
            Probe& operator=(const Probe&) = delete;

            ~Probe() {
                if (selection_) selection_->rollback(size_, freed_);
            }

            void commit() { selection_ = nullptr; }
        private:
            Selection* selection_;
            std::size_t size_, freed_;
        };

        [[nodiscard]] Probe probe(std::span<const Handle> additions) {
            return Probe(*this, additions);
        }

        const Resources& released() const { return released_; }

        std::span<const Handle> handles() const { return handles_; }

        std::span<const std::uint32_t> freed() const { return freed_; }

        bool contains(Handle handle) const {
            return inventory_->contains(handle) && selected_[handle.index];
        }

    private:
        static void add_resources(Resources& a, const Resources& b) { a.merge(b); }

        static void subtract_resources(Resources& a, const Resources& b) { a.subtract(b); }

        void add(Handle handle) {
            if (!inventory_->contains(handle))
                throw std::logic_error("release query has a stale checkpoint");
            if (selected_[handle.index]) return;
            selected_[handle.index] = true;
            handles_.push_back(handle);
            for (const auto object : inventory_->objects_by_handle_[handle.index]) {
                ++inventory_->work_.owner_visits;
                if (!--remaining_[object]) {
                    add_resources(released_, inventory_->objects_[object].released);
                    freed_.push_back(object);
                }
            }
        }

        void rollback(std::size_t size, std::size_t freed) {
            while (handles_.size() > size) {
                const auto handle = handles_.back();
                handles_.pop_back();
                selected_[handle.index] = false;
                for (const auto object : inventory_->objects_by_handle_[handle.index]) {
                    if (!remaining_[object]++ && size != 0)
                        subtract_resources(released_, inventory_->objects_[object].released);
                }
            }
            freed_.resize(freed);
            if (size == 0) released_ = {};
        }

        const ReleaseInventory* inventory_;
        std::vector<bool> selected_;
        std::vector<std::uint32_t> remaining_, freed_;
        std::vector<Handle> handles_;
        Resources released_;
    };

    Selection select() const { return Selection(*this); }

private:
    std::vector<std::optional<Handle>> handles_;
    std::vector<std::vector<std::uint32_t>> objects_by_handle_;
    std::vector<Object> objects_;
    mutable Work work_;
};

// Quotes and their inventory describe one synchronous store view. Rebuild after mutation or
// yielding; retaining this metadata does not keep any underlying resource alive.
template <class Release, class Resources, class Handle = CheckpointHandle>
struct ReleasePlan {
    std::vector<Release> releases;
    std::shared_ptr<const ReleaseInventory<Resources, Handle>> inventory;
};

} // namespace ninfer::models::context
