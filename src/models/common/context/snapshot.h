#pragma once

#include "models/common/context/directory/history.h"
#include "models/common/context/directory/state.h"
#include "models/common/context/directory.h"
#include "models/common/context/work.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::context {

// A complete immutable recovery point. Metadata and identity are native types; the common
// owner only retains the exact StateImage and the supplied logical KV prefixes.
template <class Metadata, class Identity>
class ContextSnapshot {
public:
    ContextSnapshot(StateDirectory& states, Continuation<Metadata> continuation,
                    std::vector<HistoryPrefix> prefixes, std::shared_ptr<const Identity> identity)
        : states_(&states), continuation_(std::move(continuation)), prefixes_(std::move(prefixes)),
          identity_(std::move(identity)) {
        if (!identity_) throw std::invalid_argument("snapshot has no input identity");
        for (const auto& prefix : prefixes_)
            if (!prefix.valid())
                throw std::invalid_argument("snapshot has an empty prefix reference");
        states_->retain_checkpoint_reference(continuation_.image);
        for (auto& prefix : prefixes_) prefix.set_use(ContextUse::Cache);
    }

    ~ContextSnapshot() { release(); }

    ContextSnapshot(const ContextSnapshot& other)
        : states_(other.states_), continuation_(other.continuation_), prefixes_(other.prefixes_),
          identity_(other.identity_) {
        if (states_) states_->retain_checkpoint_reference(continuation_.image);
        set_use(other.use_, other.memory_);
    }

    ContextSnapshot(ContextSnapshot&& other) noexcept
        : states_(std::exchange(other.states_, nullptr)),
          continuation_(std::move(other.continuation_)), prefixes_(std::move(other.prefixes_)),
          identity_(std::move(other.identity_)), use_(other.use_), memory_(other.memory_) {}

    ContextSnapshot& operator=(const ContextSnapshot& other) {
        if (this != &other) {
            ContextSnapshot copy(other);
            *this = std::move(copy);
        }
        return *this;
    }

    ContextSnapshot& operator=(ContextSnapshot&& other) noexcept {
        if (this != &other) {
            release();
            states_       = std::exchange(other.states_, nullptr);
            continuation_ = std::move(other.continuation_);
            prefixes_     = std::move(other.prefixes_);
            identity_     = std::move(other.identity_);
            use_          = other.use_;
            memory_       = other.memory_;
        }
        return *this;
    }

    [[nodiscard]] const Continuation<Metadata>& continuation() const noexcept {
        return continuation_;
    }

    [[nodiscard]] const Identity& identity() const noexcept { return *identity_; }

    [[nodiscard]] const std::shared_ptr<const Identity>& identity_owner() const noexcept {
        return identity_;
    }

    [[nodiscard]] const HistoryPrefix& prefix(std::size_t index) const {
        return prefixes_.at(index);
    }

    [[nodiscard]] std::span<const HistoryPrefix> prefixes() const noexcept { return prefixes_; }

    void set_use(ContextUse use, ContextMemory memory = ContextMemory::Device) {
        if (use == use_ && memory == memory_) return;
        if (states_ && use == ContextUse::Request)
            states_->retain_request_reference(continuation_.image, memory);
        if (states_ && use_ == ContextUse::Request)
            states_->release_request_reference(continuation_.image, memory_);
        for (auto& prefix : prefixes_) prefix.set_use(use, memory);
        use_    = use;
        memory_ = memory;
    }

    [[nodiscard]] std::vector<runtime::ContextTransferRequirement>
    restore_requirements(const ContextDirectory& directory) const {
        std::vector<runtime::ContextTransferRequirement> result;
        const auto state_records = states_->allocations(continuation_.image);
        for (const auto rank : states_->ranks(states_->variant(continuation_.image))) {
            if (std::any_of(state_records.begin(), state_records.end(), [&](const auto& record) {
                    return record.domain.rank == rank &&
                           record.domain.memory == ContextMemory::Device;
                }))
                continue;
            const auto source = states_->host_binding(continuation_.image, rank);
            auto requirement  = state_transfer_requirement(
                {source.bytes, 1}, runtime::ContextTransferDirection::HostToDevice);
            requirement.rank = rank;
            result.push_back(requirement);
        }
        for (const auto& prefix : prefixes_) {
            const auto pool     = prefix.history().pool_id;
            const auto& storage = directory.pool(pool);
            for (const auto& part : storage.pages().layout().parts) {
                std::uint32_t missing = 0;
                TransferWork work;
                for (std::uint32_t i = 0; i < kv_pages_for_frontier(prefix.frontier()); ++i) {
                    const auto page  = storage.histories().logical_page(prefix.history(), i);
                    const auto ranks = storage.pages().ranks(page);
                    if (std::find(ranks.begin(), ranks.end(), part.rank) == ranks.end()) continue;
                    const auto columns = std::min<std::uint32_t>(
                        kPagedKVPageSize, prefix.frontier() - i * kPagedKVPageSize);
                    if (storage.pages().device_covers(page, part.rank, columns)) continue;
                    const auto source = storage.pages().host_binding(page, part.rank);
                    work.payload_bytes += source.bytes;
                    ++work.copy_operations;
                    ++missing;
                }
                if (missing) {
                    auto requirement = kv_transfer_requirement(
                        {runtime::KvPoolId{static_cast<std::uint16_t>(pool)}},
                        runtime::ContextTransferDirection::HostToDevice, missing, work);
                    requirement.rank = part.rank;
                    result.push_back(requirement);
                }
            }
        }
        return result;
    }

    [[nodiscard]] StateId take_mutable_image() {
        if (!states_) throw std::logic_error("snapshot image already consumed");
        states_->consume_checkpoint(continuation_.image);
        if (use_ == ContextUse::Request)
            states_->release_request_reference(continuation_.image, memory_);
        states_ = nullptr;
        return continuation_.image;
    }

private:
    void release() noexcept {
        if (states_ && use_ == ContextUse::Request)
            states_->release_request_reference(continuation_.image, memory_);
        if (states_ && !states_->release_checkpoint_owner(continuation_.image)) std::terminate();
        states_ = nullptr;
    }

    StateDirectory* states_;
    Continuation<Metadata> continuation_;
    std::vector<HistoryPrefix> prefixes_;
    std::shared_ptr<const Identity> identity_;
    ContextUse use_       = ContextUse::Cache;
    ContextMemory memory_ = ContextMemory::Device;
};

} // namespace ninfer::models::context
