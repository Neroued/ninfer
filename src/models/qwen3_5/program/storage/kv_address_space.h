#pragma once

#include "models/qwen3_5/program/storage/logical_kv_store.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

class KVAddressSpaceStore;

class KVAddressSpaceHandle {
public:
    KVAddressSpaceHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(KVAddressSpaceHandle,
                                         KVAddressSpaceHandle) noexcept = default;

private:
    KVAddressSpaceHandle(const KVAddressSpaceStore* owner, std::uint32_t index,
                         std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const KVAddressSpaceStore* owner_ = nullptr;
    std::uint32_t index_              = 0;
    std::uint32_t generation_         = 0;

    friend class KVAddressSpaceStore;
};

class KVActivationReservation {
public:
    KVActivationReservation() noexcept = default;

    KVActivationReservation(KVActivationReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), address_(other.address_),
          growth_pages_(other.growth_pages_), activation_frontier_(other.activation_frontier_),
          page_reservation_(std::move(other.page_reservation_)), row_(std::move(other.row_)) {}

    KVActivationReservation& operator=(KVActivationReservation&&)      = delete;
    KVActivationReservation(const KVActivationReservation&)            = delete;
    KVActivationReservation& operator=(const KVActivationReservation&) = delete;

private:
    KVActivationReservation(KVAddressSpaceStore& owner, KVAddressSpaceHandle address,
                            std::uint32_t growth_pages,
                            std::optional<std::uint32_t> activation_frontier,
                            DeviceKVPageReservation&& page_reservation,
                            KVExecutionRowLease&& row) noexcept
        : owner_(&owner), address_(address), growth_pages_(growth_pages),
          activation_frontier_(activation_frontier), page_reservation_(std::move(page_reservation)),
          row_(std::move(row)) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle address_;
    std::uint32_t growth_pages_ = 0;
    std::optional<std::uint32_t> activation_frontier_;
    DeviceKVPageReservation page_reservation_;
    std::optional<KVExecutionRowLease> row_;

    friend class KVAddressSpaceStore;
};

class KVPrefixForkReservation {
public:
    KVPrefixForkReservation() noexcept = default;
    ~KVPrefixForkReservation();

    KVPrefixForkReservation(KVPrefixForkReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), source_(other.source_),
          destination_(other.destination_), frontier_(other.frontier_),
          full_pages_(other.full_pages_), tail_columns_(other.tail_columns_),
          growth_pages_(other.growth_pages_), page_reservation_(std::move(other.page_reservation_)),
          row_(std::move(other.row_)), tail_destination_(other.tail_destination_) {
        other.tail_destination_.reset();
    }

    KVPrefixForkReservation& operator=(KVPrefixForkReservation&&)      = delete;
    KVPrefixForkReservation(const KVPrefixForkReservation&)            = delete;
    KVPrefixForkReservation& operator=(const KVPrefixForkReservation&) = delete;

    [[nodiscard]] bool needs_tail_copy() const noexcept { return tail_columns_ != 0; }

private:
    KVPrefixForkReservation(KVAddressSpaceStore& owner, KVAddressSpaceHandle source,
                            KVAddressSpaceHandle destination, std::uint32_t frontier,
                            std::uint32_t full_pages, std::uint32_t tail_columns,
                            std::uint32_t growth_pages, DeviceKVPageReservation&& page_reservation,
                            KVExecutionRowLease&& row,
                            std::optional<LogicalKVPageHandle> tail_destination) noexcept
        : owner_(&owner), source_(source), destination_(destination), frontier_(frontier),
          full_pages_(full_pages), tail_columns_(tail_columns), growth_pages_(growth_pages),
          page_reservation_(std::move(page_reservation)), row_(std::move(row)),
          tail_destination_(tail_destination) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle source_;
    KVAddressSpaceHandle destination_;
    std::uint32_t frontier_     = 0;
    std::uint32_t full_pages_   = 0;
    std::uint32_t tail_columns_ = 0;
    std::uint32_t growth_pages_ = 0;
    DeviceKVPageReservation page_reservation_;
    std::optional<KVExecutionRowLease> row_;
    std::optional<LogicalKVPageHandle> tail_destination_;

    friend class KVAddressSpaceStore;
};

// An immutable prefix view exported from a stopped active history. The history keeps its
// execution row, growth reservation and suffix; only the view's partial tail needs a copy.
class KVActivePrefixViewReservation {
public:
    KVActivePrefixViewReservation() noexcept = default;
    ~KVActivePrefixViewReservation();

    KVActivePrefixViewReservation(KVActivePrefixViewReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), source_(other.source_),
          destination_(other.destination_), frontier_(other.frontier_),
          sources_(std::move(other.sources_)),
          tail_reservation_(std::move(other.tail_reservation_)),
          tail_destination_(std::exchange(other.tail_destination_, std::nullopt)) {}

    KVActivePrefixViewReservation& operator=(KVActivePrefixViewReservation&&)      = delete;
    KVActivePrefixViewReservation(const KVActivePrefixViewReservation&)            = delete;
    KVActivePrefixViewReservation& operator=(const KVActivePrefixViewReservation&) = delete;

    [[nodiscard]] bool needs_tail_copy() const noexcept {
        return frontier_ % static_cast<std::uint32_t>(kPagedKVPageSize) != 0;
    }

private:
    KVActivePrefixViewReservation(KVAddressSpaceStore& owner, KVAddressSpaceHandle source,
                                  KVAddressSpaceHandle destination, std::uint32_t frontier,
                                  std::vector<LogicalKVPageHandle>&& sources,
                                  DeviceKVPageReservation&& tail_reservation,
                                  std::optional<LogicalKVPageHandle> tail_destination) noexcept
        : owner_(&owner), source_(source), destination_(destination), frontier_(frontier),
          sources_(std::move(sources)), tail_reservation_(std::move(tail_reservation)),
          tail_destination_(tail_destination) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle source_;
    KVAddressSpaceHandle destination_;
    std::uint32_t frontier_ = 0;
    std::vector<LogicalKVPageHandle> sources_;
    DeviceKVPageReservation tail_reservation_;
    std::optional<LogicalKVPageHandle> tail_destination_;

    friend class KVAddressSpaceStore;
};

// A logical page occurs once in an address and keeps its position in prefix aliases.
// Growth/COW introduce fresh handles; truncation removes a suffix without moving pages.
class KVAddressSpaceStore {
public:
    KVAddressSpaceStore(LogicalKVPageStore& pages, KVExecutionTablePool& tables,
                        std::uint32_t address_capacity, std::uint32_t page_capacity)
        : pages_(&pages), tables_(&tables), page_capacity_(page_capacity),
          addresses_(address_capacity), free_(address_capacity), free_count_(address_capacity) {
        if (address_capacity == 0 || page_capacity == 0 ||
            page_capacity != tables.logical_page_capacity()) {
            throw std::invalid_argument("KV address-space geometry is invalid");
        }
        for (std::uint32_t index = 0; index < address_capacity; ++index) {
            free_[index] = address_capacity - 1U - index;
        }
        publish_scratch_.reserve(page_capacity_);
        materialization_scratch_.reserve(page_capacity_);
        while (directory_capacity_ < page_capacity_) { directory_capacity_ *= 2; }
    }

    KVAddressSpaceStore(const KVAddressSpaceStore&)            = delete;
    KVAddressSpaceStore& operator=(const KVAddressSpaceStore&) = delete;
    KVAddressSpaceStore(KVAddressSpaceStore&&)                 = delete;
    KVAddressSpaceStore& operator=(KVAddressSpaceStore&&)      = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(addresses_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    [[nodiscard]] std::optional<KVAddressSpaceHandle>
    create_active(std::uint32_t growth_pages, std::int32_t execution_row, cudaStream_t stream) {
        if (growth_pages > page_capacity_) { return std::nullopt; }
        std::optional<KVAddressSpaceHandle> handle = create_inactive();
        if (!handle) { return std::nullopt; }
        try {
            activate(*handle, growth_pages, execution_row, stream);
            return handle;
        } catch (...) {
            (void)release(*handle);
            throw;
        }
    }

    [[nodiscard]] std::optional<KVAddressSpaceHandle> create_inactive() noexcept {
        if (free_count_ == 0) { return std::nullopt; }
        const std::uint32_t index = free_[--free_count_];
        Address& address          = addresses_[index];
        if (address.occupied) {
            free_[free_count_++] = index;
            return std::nullopt;
        }
        address.occupied = true;
        return KVAddressSpaceHandle(this, index, address.generation);
    }

    [[nodiscard]] bool valid(KVAddressSpaceHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < addresses_.size() &&
               addresses_[handle.index_].occupied &&
               addresses_[handle.index_].generation == handle.generation_;
    }

    void activate(KVAddressSpaceHandle handle, std::uint32_t growth_pages,
                  std::int32_t execution_row, cudaStream_t stream) {
        auto reservation = prepare_activation(handle, growth_pages, execution_row);
        commit_activation(std::move(reservation), stream);
    }

    [[nodiscard]] KVActivationReservation
    prepare_activation(KVAddressSpaceHandle handle, std::uint32_t growth_pages,
                       std::int32_t execution_row,
                       std::optional<std::uint32_t> activation_frontier = std::nullopt) {
        Address& address = require(handle);
        if (address.active || address.row || address.reservation.valid() ||
            growth_pages > page_capacity_ ||
            (activation_frontier && *activation_frontier > address.committed_frontier)) {
            throw std::logic_error("KV address space is not reservable for activation");
        }
        const std::uint32_t required_pages =
            activation_frontier ? pages_for_tokens(*activation_frontier) : address.page_count;
        DeviceKVPageReservation reservation = pages_->physical_pool().make_empty_reservation();
        std::uint32_t missing_replicas      = 0;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            if (!pages_->device_resident(membership(address, page))) { ++missing_replicas; }
        }
        if (growth_pages > page_capacity_ - required_pages) {
            throw std::invalid_argument("KV activation growth exceeds address capacity");
        }
        const std::uint64_t required = static_cast<std::uint64_t>(missing_replicas) + growth_pages;
        if (required > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("KV activation reservation overflow");
        }
        pages_->physical_pool().resize_reservation(reservation,
                                                   static_cast<std::uint32_t>(required));
        KVExecutionRowLease row = tables_->acquire(execution_row);
        return KVActivationReservation(*this, handle, growth_pages, activation_frontier,
                                       std::move(reservation), std::move(row));
    }

    [[nodiscard]] DeviceKVPageReservation& page_reservation(KVActivationReservation& activation) {
        if (activation.owner_ != this || !valid(activation.address_)) {
            throw std::logic_error("KV activation reservation is stale");
        }
        return activation.page_reservation_;
    }

    void commit_activation(KVActivationReservation&& activation, cudaStream_t stream) {
        if (activation.owner_ != this) {
            throw std::logic_error("KV activation reservation belongs to another store");
        }
        if (!activation.row_) {
            throw std::logic_error("KV activation reservation has no execution row");
        }
        if (!valid(activation.address_)) {
            throw std::logic_error("KV activation reservation address is stale");
        }
        Address& address = require(activation.address_);
        if (address.active || address.row || address.reservation.valid() ||
            activation.growth_pages_ > page_capacity_ - address.page_count ||
            (activation.activation_frontier_ &&
             address.committed_frontier != *activation.activation_frontier_)) {
            throw std::logic_error("KV activation destination changed after reservation");
        }
        const std::uint32_t expected = activation.growth_pages_;
        if (!activation.page_reservation_.belongs_to(pages_->physical_pool()) ||
            activation.page_reservation_.pages() != expected) {
            throw std::logic_error("KV activation capacity reservation changed");
        }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (!pages_->device_resident(logical) || (pages_->address_references(logical) == 1 &&
                                                      !pages_->can_set_writer(logical, true))) {
                throw std::logic_error("KV activation has an unavailable Device page");
            }
        }
        publish_scratch_.clear();
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            publish_scratch_.push_back(pages_->physical(membership(address, page)));
        }
        tables_->publish(activation.row_->handle(), 0, publish_scratch_, stream);
        address.reservation = std::move(activation.page_reservation_);
        address.row.emplace(std::move(*activation.row_));
        activation.row_.reset();
        address.active    = true;
        activation.owner_ = nullptr;
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (pages_->address_references(logical) == 1) { pages_->set_writer(logical, true); }
            pages_->retain_active_reference(logical);
        }
    }

    void deactivate(KVAddressSpaceHandle handle) {
        Address& address = require_active(handle);
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (pages_->writer_references(logical) != 0 &&
                !pages_->can_set_writer(logical, false)) {
                throw std::logic_error("KV deactivation has an invalid writer reference");
            }
        }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (pages_->writer_references(logical) != 0) { pages_->set_writer(logical, false); }
            pages_->release_active_reference(logical);
        }
        address.row.reset();
        address.reservation.release();
        address.active = false;
    }

    [[nodiscard]] KVPrefixForkReservation
    prepare_prefix_fork(KVAddressSpaceHandle source_handle, KVAddressSpaceHandle destination_handle,
                        std::uint32_t frontier, std::uint32_t growth_pages,
                        std::int32_t execution_row) {
        Address& source      = require(source_handle);
        Address& destination = require(destination_handle);
        if (source.active || source.row || source.reservation.valid()) {
            throw std::logic_error("KV retained-prefix source is still active");
        }
        if (destination.active || destination.row || destination.reservation.valid() ||
            destination.page_count != 0 || destination.committed_frontier != 0) {
            throw std::logic_error("KV retained-prefix destination is not empty");
        }
        if (frontier == 0 || frontier > source.committed_frontier) {
            throw std::logic_error("KV retained-prefix frontier is unavailable");
        }
        const std::uint32_t required_pages = pages_for_tokens(frontier);
        if (growth_pages > page_capacity_ - required_pages) {
            throw std::invalid_argument("KV retained-prefix growth exceeds address capacity");
        }
        const std::uint32_t page_size    = static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t full_pages   = frontier / page_size;
        const std::uint32_t tail_columns = frontier % page_size;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            const std::uint32_t required      = page < full_pages ? page_size : tail_columns;
            if (!pages_->device_resident(logical) ||
                pages_->committed_columns(logical) < required || !pages_->can_pin_source(logical) ||
                (page < full_pages && !pages_->can_retain_reference(logical, false))) {
                throw std::logic_error("KV retained-prefix source is not stable");
            }
        }

        DeviceKVPageReservation reservation = pages_->physical_pool().make_empty_reservation();
        pages_->physical_pool().resize_reservation(reservation,
                                                   growth_pages + (tail_columns != 0 ? 1U : 0U));
        KVExecutionRowLease row = tables_->acquire(execution_row);
        std::optional<LogicalKVPageHandle> tail_destination;
        if (tail_columns != 0) {
            tail_destination = pages_->materialize_transfer_destination(reservation, tail_columns);
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            pages_->pin_source(membership(source, page));
        }
        return KVPrefixForkReservation(*this, source_handle, destination_handle, frontier,
                                       full_pages, tail_columns, growth_pages,
                                       std::move(reservation), std::move(row), tail_destination);
    }

    [[nodiscard]] DeviceKVPageHandle
    prefix_fork_tail_source(const KVPrefixForkReservation& fork) const {
        require_prefix_fork(fork);
        if (fork.tail_columns_ == 0) {
            throw std::logic_error("page-aligned KV prefix fork has no tail source");
        }
        const Address& source = require(fork.source_);
        return pages_->physical(membership(source, fork.full_pages_));
    }

    [[nodiscard]] DeviceKVPageHandle
    prefix_fork_tail_destination(const KVPrefixForkReservation& fork) const {
        require_prefix_fork(fork);
        if (!fork.tail_destination_) {
            throw std::logic_error("page-aligned KV prefix fork has no tail destination");
        }
        return pages_->physical(*fork.tail_destination_);
    }

    void commit_prefix_fork(KVPrefixForkReservation&& fork, cudaStream_t stream) {
        require_prefix_fork(fork);
        Address& source                    = require(fork.source_);
        Address& destination               = require(fork.destination_);
        const std::uint32_t required_pages = pages_for_tokens(fork.frontier_);
        const std::uint32_t growth         = fork.growth_pages_;
        if (!fork.row_ || destination.active || destination.row ||
            destination.reservation.valid() || destination.page_count != 0 ||
            fork.page_reservation_.pages() != growth) {
            throw std::logic_error("KV prefix-fork destination changed before publication");
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            if (pages_->source_pins(logical) == 0 || !pages_->device_resident(logical) ||
                (page < fork.full_pages_ && !pages_->can_retain_reference(logical, false))) {
                throw std::logic_error("KV prefix-fork source changed before publication");
            }
        }
        if (fork.tail_columns_ != 0 &&
            (!fork.tail_destination_ || !pages_->valid(*fork.tail_destination_))) {
            throw std::logic_error("KV prefix-fork tail destination is stale");
        }

        publish_scratch_.clear();
        for (std::uint32_t page = 0; page < fork.full_pages_; ++page) {
            publish_scratch_.push_back(pages_->physical(membership(source, page)));
        }
        if (fork.tail_destination_) {
            publish_scratch_.push_back(pages_->physical(*fork.tail_destination_));
        }
        auto directory = prefix_directory(source, fork.full_pages_);
        if (fork.tail_destination_) {
            directory_slot(directory, fork.full_pages_) = *fork.tail_destination_;
        }
        tables_->publish(fork.row_->handle(), 0, publish_scratch_, stream);

        for (std::uint32_t page = 0; page < fork.full_pages_; ++page) {
            const LogicalKVPageHandle logical = membership(source, page);
            pages_->retain_reference(logical, false);
            pages_->protect_coverage(logical, static_cast<std::uint32_t>(kPagedKVPageSize));
        }
        if (fork.tail_destination_) {
            pages_->publish_transfer_destination(*fork.tail_destination_, true);
            fork.tail_destination_.reset();
        }
        destination.directory          = std::move(directory);
        destination.page_count         = required_pages;
        destination.committed_frontier = fork.frontier_;
        destination.reservation        = std::move(fork.page_reservation_);
        destination.row.emplace(std::move(*fork.row_));
        fork.row_.reset();
        destination.active = true;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            pages_->retain_active_reference(membership(destination, page));
        }
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            pages_->unpin_source(membership(source, page));
        }
        fork.owner_ = nullptr;
    }

    void abort_prefix_fork(KVPrefixForkReservation& fork) noexcept {
        if (fork.owner_ != this) { return; }
        if (valid(fork.source_)) {
            Address& source                    = addresses_[fork.source_.index_];
            const std::uint32_t required_pages = pages_for_tokens(fork.frontier_);
            for (std::uint32_t page = 0; page < required_pages; ++page) {
                const LogicalKVPageHandle logical = membership(source, page);
                if (pages_->valid(logical) && pages_->source_pins(logical) != 0) {
                    pages_->unpin_source(logical);
                }
            }
        }
        if (fork.tail_destination_) {
            pages_->abort_transfer_destination(*fork.tail_destination_, fork.page_reservation_);
            fork.tail_destination_.reset();
        }
        fork.page_reservation_.release();
        fork.row_.reset();
        fork.owner_ = nullptr;
    }

    [[nodiscard]] KVActivePrefixViewReservation
    prepare_active_prefix_view(KVAddressSpaceHandle source_handle,
                               KVAddressSpaceHandle destination_handle, std::uint32_t frontier) {
        const Address& source      = require_active(source_handle);
        const Address& destination = require(destination_handle);
        if (destination.active || destination.row || destination.reservation.valid() ||
            destination.page_count != 0 || destination.committed_frontier != 0 || frontier == 0 ||
            frontier > source.checkpoint_frontier || frontier > source.committed_frontier) {
            throw std::logic_error("active KV prefix-view endpoints are invalid");
        }
        const auto page_size    = static_cast<std::uint32_t>(kPagedKVPageSize);
        const auto full_pages   = frontier / page_size;
        const auto tail_columns = frontier % page_size;
        const auto count        = pages_for_tokens(frontier);
        std::vector<LogicalKVPageHandle> sources;
        sources.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto page     = membership(source, i);
            const auto required = i < full_pages ? page_size : tail_columns;
            if (!pages_->device_resident(page) || pages_->committed_columns(page) < required ||
                (!pages_->can_pin_source(page) && !pages_->can_pin_active_source(page)) ||
                (i < full_pages && !pages_->can_retain_reference(page, false))) {
                throw std::logic_error("active KV prefix-view source is not stable");
            }
            sources.push_back(page);
        }
        auto reservation = pages_->physical_pool().make_empty_reservation();
        std::optional<LogicalKVPageHandle> tail;
        if (tail_columns != 0) {
            pages_->physical_pool().resize_reservation(reservation, 1);
            tail = pages_->materialize_transfer_destination(reservation, tail_columns);
        }
        for (const auto page : sources) { pages_->pin_source(page); }
        return KVActivePrefixViewReservation(*this, source_handle, destination_handle, frontier,
                                             std::move(sources), std::move(reservation), tail);
    }

    [[nodiscard]] DeviceKVPageHandle
    active_prefix_view_tail_source(const KVActivePrefixViewReservation& view) const {
        require_active_prefix_view(view);
        if (!view.needs_tail_copy()) {
            throw std::logic_error("page-aligned KV prefix view has no tail source");
        }
        return pages_->physical(
            membership(require(view.source_), view.frontier_ / kPagedKVPageSize));
    }

    [[nodiscard]] DeviceKVPageHandle
    active_prefix_view_tail_destination(const KVActivePrefixViewReservation& view) const {
        require_active_prefix_view(view);
        if (!view.tail_destination_) {
            throw std::logic_error("page-aligned KV prefix view has no tail destination");
        }
        return pages_->physical(*view.tail_destination_);
    }

    // The caller retires the tail copy before commit or abort, and keeps the source execution
    // stopped while the reservation is alive. No execution row is allocated or republished.
    void commit_active_prefix_view(KVActivePrefixViewReservation&& view) {
        require_active_prefix_view(view);
        const Address& source = require_active(view.source_);
        Address& destination  = require(view.destination_);
        if (destination.active || destination.row || destination.reservation.valid() ||
            destination.page_count != 0 || destination.committed_frontier != 0 ||
            source.committed_frontier < view.frontier_ || view.tail_reservation_.pages() != 0) {
            throw std::logic_error("active KV prefix-view endpoints changed before publication");
        }
        const auto page_size  = static_cast<std::uint32_t>(kPagedKVPageSize);
        const auto full_pages = view.frontier_ / page_size;
        const auto count      = pages_for_tokens(view.frontier_);
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto page     = membership(source, i);
            const auto required = i < full_pages ? page_size : view.frontier_ % page_size;
            if (page != view.sources_[i] || !pages_->device_resident(page) ||
                pages_->source_pins(page) == 0 || pages_->committed_columns(page) < required ||
                (i < full_pages && !pages_->can_retain_reference(page, false))) {
                throw std::logic_error("active KV prefix-view source changed before publication");
            }
        }
        if (view.needs_tail_copy() &&
            (!view.tail_destination_ || !pages_->valid(*view.tail_destination_))) {
            throw std::logic_error("active KV prefix-view tail destination is stale");
        }

        // Prepare all directory allocations before transferring any physical ownership.
        auto directory = prefix_directory(source, full_pages);
        if (view.tail_destination_) {
            directory_slot(directory, full_pages) = *view.tail_destination_;
        }
        for (std::uint32_t i = 0; i < full_pages; ++i) {
            const auto page = membership(source, i);
            if (pages_->writer_references(page) != 0) { pages_->set_writer(page, false); }
            pages_->retain_reference(page, false);
            pages_->protect_coverage(page, page_size);
        }
        if (view.tail_destination_) {
            pages_->publish_transfer_destination(*view.tail_destination_, false);
            pages_->protect_coverage(*view.tail_destination_, view.frontier_ % page_size);
            view.tail_destination_.reset();
        }
        destination.directory           = std::move(directory);
        destination.page_count          = count;
        destination.committed_frontier  = view.frontier_;
        destination.checkpoint_frontier = view.frontier_;
        for (std::uint32_t i = 0; i < count; ++i) { pages_->unpin_source(membership(source, i)); }
        view.owner_ = nullptr;
    }

    void abort_active_prefix_view(KVActivePrefixViewReservation& view) noexcept {
        if (view.owner_ != this) { return; }
        for (const auto page : view.sources_) {
            if (pages_->valid(page) && pages_->source_pins(page) != 0) {
                pages_->unpin_source(page);
            }
        }
        if (view.tail_destination_) {
            pages_->abort_transfer_destination(*view.tail_destination_, view.tail_reservation_);
            view.tail_destination_.reset();
        }
        view.tail_reservation_.release();
        view.owner_ = nullptr;
    }

    // A reservation covers a finite execution or recovery interval. Resizing is atomic at the pool;
    // callers combining several addresses retain their old counts until the group is acquired.
    void reserve_growth(KVAddressSpaceHandle handle, std::uint32_t growth_pages) {
        Address& address = require_active(handle);
        if (growth_pages > page_capacity_ - address.page_count) {
            throw std::invalid_argument("KV growth exceeds address capacity");
        }
        pages_->physical_pool().resize_reservation(address.reservation, growth_pages);
    }

    void release_growth(KVAddressSpaceHandle handle) {
        Address& address = require_active(handle);
        pages_->physical_pool().resize_reservation(address.reservation, 0);
    }

    [[nodiscard]] std::uint32_t growth_pages_for_tokens(KVAddressSpaceHandle handle,
                                                        std::uint32_t tokens) const {
        const Address& address     = require(handle);
        const std::uint32_t target = pages_for_tokens(tokens);
        if (target > page_capacity_) {
            throw std::invalid_argument("KV unit exceeds address capacity");
        }
        return target > address.page_count ? target - address.page_count : 0U;
    }

    void settle_growth(KVAddressSpaceHandle handle, std::uint32_t frontier,
                       std::uint32_t retain_through = 0) {
        if (frontier >= committed_frontier(handle)) { commit_frontier(handle, frontier); }
        destructive_truncate(handle, frontier);
        reserve_growth(handle, growth_pages_for_tokens(handle, retain_through));
    }

    // Coverage is a lower bound. A speculative mapping may already extend beyond this stage's
    // needs; only an explicit truncate releases it, and commit_frontier publishes valid tokens.
    void ensure_mapped_to_tokens(KVAddressSpaceHandle handle, std::uint32_t tokens,
                                 cudaStream_t stream) {
        Address& address           = require_active(handle);
        const std::uint32_t target = pages_for_tokens(tokens);
        if (target > address.page_count + address.reservation.pages()) {
            throw std::invalid_argument(
                "KV coverage exceeds reserved unit growth: tokens=" + std::to_string(tokens) +
                " required_pages=" + std::to_string(target) +
                " mapped_pages=" + std::to_string(address.page_count) +
                " reserved_pages=" + std::to_string(address.reservation.pages()));
        }
        if (target <= address.page_count) { return; }
        const std::uint32_t begin = address.page_count;
        const std::uint32_t count = target - begin;
        // Prepare only the append paths. Existing shared prefix nodes remain immutable.
        for (std::uint32_t page = begin; page < target; ++page) {
            (void)directory_slot(address.directory, page);
        }
        materialization_scratch_.assign(count, LogicalKVPageHandle{});
        std::span<LogicalKVPageHandle> added(materialization_scratch_);
        const std::optional<LogicalKVPageHandle> predecessor =
            begin == 0 ? std::nullopt
                       : std::optional<LogicalKVPageHandle>(membership(address, begin - 1U));
        pages_->materialize(address.reservation, added, predecessor);
        try {
            publish_scratch_.clear();
            for (const LogicalKVPageHandle page : added) {
                publish_scratch_.push_back(pages_->physical(page));
            }
            tables_->publish(address.row->handle(), begin, publish_scratch_, stream);
        } catch (...) {
            for (LogicalKVPageHandle& page : added) {
                pages_->dematerialize(page, address.reservation);
                page = {};
            }
            throw;
        }
        for (std::uint32_t offset = 0; offset < count; ++offset) {
            pages_->retain_active_reference(added[offset]);
            directory_slot(address.directory, begin + offset) = added[offset];
        }
        address.page_count = target;
    }

    void commit_frontier(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        Address& address = require_active(handle);
        if (frontier < address.committed_frontier ||
            pages_for_tokens(frontier) > address.page_count) {
            throw std::invalid_argument("KV committed frontier is invalid");
        }
        if (frontier == address.committed_frontier) { return; }
        const std::uint32_t page_size          = static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t first_changed_page = address.committed_frontier / page_size;
        const std::uint32_t final_changed_page = (frontier - 1U) / page_size;
        for (std::uint32_t page = first_changed_page; page <= final_changed_page; ++page) {
            const std::uint32_t begin   = page * static_cast<std::uint32_t>(kPagedKVPageSize);
            const std::uint32_t columns = std::min(page_size, frontier - begin);
            if (columns > pages_->committed_columns(membership(address, page))) {
                pages_->commit_coverage(membership(address, page), columns);
            }
        }
        address.committed_frontier = frontier;
    }

    void destructive_truncate(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        Address& address = require_active(handle);
        if (frontier > address.committed_frontier) {
            throw std::invalid_argument("KV destructive truncate extends the frontier");
        }
        if (frontier < address.checkpoint_frontier) {
            throw std::logic_error("KV truncate would remove protected checkpoint coverage");
        }
        const std::uint32_t target = pages_for_tokens(frontier);
        if (frontier == address.committed_frontier && target == address.page_count) { return; }
        for (std::uint32_t page = target; page < address.page_count; ++page) {
            if (!pages_->can_dematerialize(membership(address, page))) {
                throw std::logic_error("KV truncate would partially release a protected page");
            }
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            if (columns != pages_->committed_columns(membership(address, target - 1U)) &&
                !pages_->can_destructive_truncate(membership(address, target - 1U), columns)) {
                throw std::logic_error("KV truncate would overwrite protected coverage");
            }
        }
        while (address.page_count > target) {
            const std::uint32_t index = --address.page_count;
            LogicalKVPageHandle page  = membership(address, index);
            pages_->release_active_reference(page);
            pages_->dematerialize(page, address.reservation);
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            if (columns != pages_->committed_columns(membership(address, target - 1U))) {
                pages_->destructive_truncate(membership(address, target - 1U), columns);
            }
        }
        trim_unique_suffix(address.directory, directory_capacity_, target);
        address.committed_frontier = frontier;
    }

    [[nodiscard]] bool can_destructive_truncate_inactive(
        KVAddressSpaceHandle handle, std::uint32_t frontier,
        bool tail_host_replica_will_be_released = false) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (address.active || address.row || address.reservation.valid() ||
            frontier > address.committed_frontier) {
            return false;
        }
        const std::uint32_t target = pages_for_tokens(frontier);
        for (std::uint32_t page = target; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (!pages_->can_release_reference(logical, false)) { return false; }
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            const LogicalKVPageHandle tail = membership(address, target - 1U);
            if (columns != pages_->committed_columns(tail) &&
                !(tail_host_replica_will_be_released
                      ? pages_->can_destructive_truncate_inactive_after_host_release(tail, columns)
                      : pages_->can_destructive_truncate_inactive(tail, columns))) {
                return false;
            }
        }
        return true;
    }

    void destructive_truncate_inactive(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        if (!can_destructive_truncate_inactive(handle, frontier)) {
            throw std::logic_error("inactive KV address space is not destructively truncatable");
        }
        Address& address           = require(handle);
        const std::uint32_t target = pages_for_tokens(frontier);
        // Full pages before the selected frontier can remain shared. Only removed suffix
        // references and a changed private partial tail need mutation; protection is rebuilt from
        // the surviving address-space requirements after publication.
        auto directory = prefix_directory(address, target);
        while (address.page_count > target) {
            const std::uint32_t index         = --address.page_count;
            const LogicalKVPageHandle logical = membership(address, index);
            if (!pages_->release_reference(logical, false)) { std::terminate(); }
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            const LogicalKVPageHandle tail = membership(address, target - 1U);
            if (columns != pages_->committed_columns(tail)) {
                pages_->destructive_truncate_inactive(tail, columns);
            }
        }
        address.directory           = std::move(directory);
        address.committed_frontier  = frontier;
        address.checkpoint_frontier = 0;
        rebuild_checkpoint_protection();
    }

    void set_checkpoint_requirement(KVAddressSpaceHandle handle, std::uint32_t protected_frontier) {
        Address& address = require(handle);
        if (protected_frontier > address.committed_frontier) {
            throw std::invalid_argument("KV checkpoint requirement exceeds committed frontier");
        }
        address.checkpoint_frontier = protected_frontier;
        rebuild_checkpoint_protection();
    }

    // This checks the address lifecycle and frontier. Each page still needs its own release
    // check; callers may cache those facts only within a read-only evaluation.
    [[nodiscard]] std::optional<std::pair<std::uint32_t, std::uint32_t>>
    inactive_suffix_range(KVAddressSpaceHandle handle, std::uint32_t frontier) const noexcept {
        if (!valid(handle)) { return std::nullopt; }
        const Address& address = addresses_[handle.index_];
        if (address.active || address.row || address.reservation.valid() ||
            frontier > address.committed_frontier) {
            return std::nullopt;
        }
        return std::pair{pages_for_tokens(frontier), address.page_count};
    }

    // Inspect the suffix while checking the same per-reference release conditions as truncation.
    // The visitor may collect read-only facts; it must not change references or residency.
    template <typename Visitor>
    [[nodiscard]] bool visit_releasable_inactive_suffix(KVAddressSpaceHandle handle,
                                                        std::uint32_t frontier,
                                                        Visitor&& visitor) const {
        const auto range = inactive_suffix_range(handle, frontier);
        if (!range) { return false; }
        const Address& address = addresses_[handle.index_];
        for (auto page = range->first; page < range->second; ++page) {
            const auto logical = membership(address, page);
            if (!pages_->can_release_reference(logical, false)) { return false; }
            visitor(logical);
        }
        return true;
    }

    [[nodiscard]] bool can_truncate_inactive_prefix(KVAddressSpaceHandle handle,
                                                    std::uint32_t frontier) const noexcept {
        return visit_releasable_inactive_suffix(handle, frontier, [](auto) {});
    }

    void truncate_inactive_prefix(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        if (!can_truncate_inactive_prefix(handle, frontier)) {
            throw std::logic_error("inactive KV checkpoint prefix is not truncatable");
        }
        Address& address           = require(handle);
        const std::uint32_t target = pages_for_tokens(frontier);
        auto directory             = prefix_directory(address, target);
        while (address.page_count > target) {
            const std::uint32_t index         = --address.page_count;
            const LogicalKVPageHandle logical = membership(address, index);
            if (!pages_->release_reference(logical, false)) { std::terminate(); }
        }
        address.directory           = std::move(directory);
        address.committed_frontier  = frontier;
        address.checkpoint_frontier = std::min(address.checkpoint_frontier, frontier);
        rebuild_checkpoint_protection();
    }

    [[nodiscard]] std::uint32_t mapped_pages(KVAddressSpaceHandle handle) const {
        return require(handle).page_count;
    }

    [[nodiscard]] std::uint32_t reserved_growth_pages(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        return address.reservation.valid() ? address.reservation.pages() : 0U;
    }

    [[nodiscard]] std::uint32_t committed_frontier(KVAddressSpaceHandle handle) const {
        return require(handle).committed_frontier;
    }

    [[nodiscard]] std::int32_t bound_row(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle) || !addresses_[handle.index_].row) { return -1; }
        return addresses_[handle.index_].row->row_index();
    }

    [[nodiscard]] bool active(KVAddressSpaceHandle handle) const noexcept {
        return valid(handle) && addresses_[handle.index_].active;
    }

    [[nodiscard]] const KVExecutionRowLease& execution_row(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        if (!address.active || !address.row) {
            throw std::logic_error("KV address space has no execution row");
        }
        return *address.row;
    }

    [[nodiscard]] DeviceKVPageHandle physical_page(KVAddressSpaceHandle handle,
                                                   std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return pages_->physical(membership(address, logical_page));
    }

    [[nodiscard]] std::uint64_t content_epoch(KVAddressSpaceHandle handle,
                                              std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return pages_->content_epoch(membership(address, logical_page));
    }

    [[nodiscard]] LogicalKVPageHandle logical_page(KVAddressSpaceHandle handle,
                                                   std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return membership(address, logical_page);
    }

    [[nodiscard]] bool has_active_reference(LogicalKVPageHandle page) const noexcept {
        return pages_->valid(page) && pages_->active_address_references(page) != 0;
    }

    [[nodiscard]] bool can_release(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (address.active || address.row || address.reservation.valid()) { return false; }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->can_release_reference(membership(address, page), false)) { return false; }
        }
        return true;
    }

    [[nodiscard]] bool can_release_after_deactivate(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (!address.active) { return can_release(handle); }
        if (!address.row) { return false; }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->can_release_reference_after_active_reference(membership(address, page))) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool release_after_deactivate(KVAddressSpaceHandle handle) noexcept {
        if (!can_release_after_deactivate(handle)) { return false; }
        try {
            if (addresses_[handle.index_].active) { deactivate(handle); }
        } catch (...) { std::terminate(); }
        if (!release(handle)) { std::terminate(); }
        return true;
    }

    [[nodiscard]] bool release(KVAddressSpaceHandle handle) noexcept {
        if (!can_release(handle)) { return false; }
        Address& address = addresses_[handle.index_];
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->release_reference(membership(address, page), false)) { std::terminate(); }
        }
        const std::uint32_t index      = handle.index_;
        const std::uint32_t generation = next_generation(address.generation);
        address                        = Address{};
        address.generation             = generation;
        free_[free_count_++]           = index;
        rebuild_checkpoint_protection();
        return true;
    }

private:
    // Address roots share immutable chunks. Nodes hold non-owning logical capabilities; the
    // address store, rather than shared_ptr destruction, owns every physical page reference.
    // Thus dropping an ancestor cannot invalidate descendants, and Host replicas keep the same
    // logical identity. Only paths containing changed membership are copied.
    static constexpr std::uint32_t kDirectoryChunkPages = 32;

    struct DirectoryNode {
        std::shared_ptr<DirectoryNode> left;
        std::shared_ptr<DirectoryNode> right;
        std::array<LogicalKVPageHandle, kDirectoryChunkPages> pages{};
    };

    using Directory = std::shared_ptr<DirectoryNode>;

    struct Address {
        Directory directory;
        std::uint32_t generation          = 1;
        std::uint32_t page_count          = 0;
        std::uint32_t committed_frontier  = 0;
        std::uint32_t checkpoint_frontier = 0;
        DeviceKVPageReservation reservation;
        std::optional<KVExecutionRowLease> row;
        bool occupied = false;
        bool active   = false;
    };

    [[nodiscard]] static std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
        return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
    }

    [[nodiscard]] static std::uint32_t next_generation(std::uint32_t generation) noexcept {
        ++generation;
        return generation == 0 ? 1 : generation;
    }

    void rebuild_checkpoint_protection() noexcept {
        try {
            for (const Address& address : addresses_) {
                if (!address.occupied) { continue; }
                for (std::uint32_t page = 0; page < address.page_count; ++page) {
                    pages_->set_protected_coverage(membership(address, page), 0);
                }
            }
            for (const Address& address : addresses_) {
                if (!address.occupied || address.checkpoint_frontier == 0) { continue; }
                if (address.checkpoint_frontier > address.committed_frontier ||
                    pages_for_tokens(address.checkpoint_frontier) > address.page_count) {
                    std::terminate();
                }
                for (std::uint32_t page = 0; page < pages_for_tokens(address.checkpoint_frontier);
                     ++page) {
                    const std::uint32_t begin = page * static_cast<std::uint32_t>(kPagedKVPageSize);
                    pages_->protect_coverage(membership(address, page),
                                             std::min(static_cast<std::uint32_t>(kPagedKVPageSize),
                                                      address.checkpoint_frontier - begin));
                }
            }
        } catch (...) { std::terminate(); }
    }

    [[nodiscard]] Address& require(KVAddressSpaceHandle handle) {
        if (!valid(handle)) { throw std::invalid_argument("KV address-space handle is stale"); }
        return addresses_[handle.index_];
    }

    [[nodiscard]] const Address& require(KVAddressSpaceHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("KV address-space handle is stale"); }
        return addresses_[handle.index_];
    }

    void require_prefix_fork(const KVPrefixForkReservation& fork) const {
        if (fork.owner_ != this || !valid(fork.source_) || !valid(fork.destination_)) {
            throw std::logic_error("KV prefix-fork reservation is stale");
        }
    }

    void require_active_prefix_view(const KVActivePrefixViewReservation& view) const {
        if (view.owner_ != this || !valid(view.source_) || !valid(view.destination_)) {
            throw std::logic_error("active KV prefix-view reservation is stale");
        }
    }

    [[nodiscard]] Address& require_active(KVAddressSpaceHandle handle) {
        Address& address = require(handle);
        if (!address.active || !address.row || !address.reservation.valid()) {
            throw std::logic_error("KV address space is not active");
        }
        return address;
    }

    [[nodiscard]] const Address& require_active(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        if (!address.active || !address.row || !address.reservation.valid()) {
            throw std::logic_error("KV address space is not active");
        }
        return address;
    }

    [[nodiscard]] LogicalKVPageHandle membership(const Address& address,
                                                 std::uint32_t page) const noexcept {
        const DirectoryNode* node = address.directory.get();
        std::uint64_t width       = directory_capacity_;
        while (width > kDirectoryChunkPages) {
            width /= 2;
            if (page < width) {
                node = node->left.get();
            } else {
                page -= static_cast<std::uint32_t>(width);
                node = node->right.get();
            }
        }
        return node->pages[page];
    }

    [[nodiscard]] LogicalKVPageHandle& directory_slot(Directory& root, std::uint32_t page) {
        Directory* node     = &root;
        std::uint64_t width = directory_capacity_;
        for (;;) {
            if (!*node) {
                *node = std::make_shared<DirectoryNode>();
            } else if (node->use_count() != 1) {
                *node = std::make_shared<DirectoryNode>(**node);
            }
            if (width == kDirectoryChunkPages) { return (*node)->pages[page]; }
            width /= 2;
            if (page < width) {
                node = &(*node)->left;
            } else {
                page -= static_cast<std::uint32_t>(width);
                node = &(*node)->right;
            }
        }
    }

    [[nodiscard]] static Directory directory_prefix(const Directory& node, std::uint64_t width,
                                                    std::uint32_t count) {
        if (count == 0 || !node) { return {}; }
        if (count >= width) { return node; }
        auto result = std::make_shared<DirectoryNode>(*node);
        if (width == kDirectoryChunkPages) {
            std::fill(result->pages.begin() + count, result->pages.end(), LogicalKVPageHandle{});
        } else {
            width /= 2;
            if (count <= width) {
                result->left = directory_prefix(node->left, width, count);
                result->right.reset();
            } else {
                result->right =
                    directory_prefix(node->right, width, count - static_cast<std::uint32_t>(width));
            }
        }
        return result;
    }

    [[nodiscard]] Directory prefix_directory(const Address& address, std::uint32_t count) const {
        if (count == address.page_count) { return address.directory; }
        return directory_prefix(address.directory, directory_capacity_, count);
    }

    static void trim_unique_suffix(Directory& node, std::uint64_t width,
                                   std::uint32_t count) noexcept {
        if (!node || count >= width) { return; }
        if (count == 0) {
            node.reset();
            return;
        }
        // A shared node contains only retained prefix pages: appending a suffix made its
        // changed path private. It needs no pruning when all removed pages lived elsewhere.
        // shared_ptr::unique() was removed in C++20 (MSVC rejects it); use_count() == 1 is
        // the same advisory last-owner check.
        if (node.use_count() != 1) { return; }
        if (width == kDirectoryChunkPages) {
            std::fill(node->pages.begin() + count, node->pages.end(), LogicalKVPageHandle{});
        } else {
            width /= 2;
            if (count <= width) {
                trim_unique_suffix(node->left, width, count);
                node->right.reset();
            } else {
                trim_unique_suffix(node->right, width, count - static_cast<std::uint32_t>(width));
            }
        }
    }

    LogicalKVPageStore* pages_    = nullptr;
    KVExecutionTablePool* tables_ = nullptr;
    std::uint32_t page_capacity_  = 0;
    std::vector<Address> addresses_;
    std::vector<std::uint32_t> free_;
    std::uint64_t directory_capacity_ = kDirectoryChunkPages;
    std::vector<LogicalKVPageHandle> materialization_scratch_;
    std::vector<DeviceKVPageHandle> publish_scratch_;
    std::uint32_t free_count_ = 0;
};

inline KVPrefixForkReservation::~KVPrefixForkReservation() {
    if (owner_ != nullptr) { owner_->abort_prefix_fork(*this); }
}

inline KVActivePrefixViewReservation::~KVActivePrefixViewReservation() {
    if (owner_ != nullptr) { owner_->abort_active_prefix_view(*this); }
}
} // namespace ninfer::models::qwen3_5::detail
