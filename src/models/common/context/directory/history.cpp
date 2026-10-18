#include "models/common/context/directory/history.h"
#include "models/common/context/directory/host.h"

#include "models/common/context/directory.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::models::context {
namespace {
constexpr auto kPageTokens = static_cast<std::uint32_t>(kPagedKVPageSize);

std::uint32_t page_count(std::uint32_t tokens) {
    return tokens == 0 ? 0 : 1 + (tokens - 1) / kPageTokens;
}

std::uint32_t coverage(std::uint32_t frontier, std::uint32_t page) {
    const auto begin = static_cast<std::uint64_t>(page) * kPageTokens;
    return begin >= frontier ? 0 : std::min<std::uint64_t>(kPageTokens, frontier - begin);
}
} // namespace

RowLease::~RowLease() { reset(); }

RowLease::RowLease(RowLease&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), row_(std::exchange(other.row_, {})) {}

RowLease& RowLease::operator=(RowLease&& other) noexcept {
    if (this != &other) {
        reset();
        owner_ = std::exchange(other.owner_, nullptr);
        row_   = std::exchange(other.row_, {});
    }
    return *this;
}

void RowLease::reset() noexcept {
    if (owner_) {
        owner_->release_row(row_);
        owner_ = nullptr;
        row_   = {};
    }
}

HistoryOwner::~HistoryOwner() { reset(); }

HistoryOwner::HistoryOwner(HistoryOwner&& other) noexcept
    : store_(std::exchange(other.store_, nullptr)), handle_(std::exchange(other.handle_, {})) {}

HistoryOwner& HistoryOwner::operator=(HistoryOwner&& other) noexcept {
    if (this != &other) {
        reset();
        store_  = std::exchange(other.store_, nullptr);
        handle_ = std::exchange(other.handle_, {});
    }
    return *this;
}

void HistoryOwner::reset() noexcept {
    if (store_) {
        store_->release_owner(handle_);
        store_  = nullptr;
        handle_ = {};
    }
}

HistoryPrefix::~HistoryPrefix() { reset(); }

HistoryPrefix::HistoryPrefix(const HistoryPrefix& other)
    : store_(other.store_), history_(other.history_), frontier_(other.frontier_) {
    if (valid()) store_->add_prefix(history_, frontier_);
}

HistoryPrefix& HistoryPrefix::operator=(const HistoryPrefix& other) {
    if (this != &other) {
        HistoryPrefix copy(other);
        *this = std::move(copy);
    }
    return *this;
}

HistoryPrefix::HistoryPrefix(HistoryPrefix&& other) noexcept
    : store_(std::exchange(other.store_, nullptr)), history_(std::exchange(other.history_, {})),
      frontier_(std::exchange(other.frontier_, 0)) {}

HistoryPrefix& HistoryPrefix::operator=(HistoryPrefix&& other) noexcept {
    if (this != &other) {
        reset();
        store_    = std::exchange(other.store_, nullptr);
        history_  = std::exchange(other.history_, {});
        frontier_ = std::exchange(other.frontier_, 0);
    }
    return *this;
}

void HistoryPrefix::reset() noexcept {
    if (valid()) {
        store_->remove_prefix(history_, frontier_);
        store_    = nullptr;
        history_  = {};
        frontier_ = 0;
    }
}

HistoryBinding::~HistoryBinding() {
    if (store_) store_->abort(*this);
}

HistoryBinding::HistoryBinding(HistoryBinding&& other) noexcept
    : store_(std::exchange(other.store_, nullptr)), history_(std::move(other.history_)),
      source_(std::move(other.source_)), reservation_(std::move(other.reservation_)),
      row_(std::move(other.row_)), restores_(std::move(other.restores_)),
      tail_destination_(std::exchange(other.tail_destination_, std::nullopt)),
      tail_source_(other.tail_source_), tail_pinned_(std::exchange(other.tail_pinned_, false)),
      restores_published_(other.restores_published_), write_tail_(other.write_tail_),
      frontier_(other.frontier_), mapping_prepared_(other.mapping_prepared_) {}

bool HistoryBinding::blocks_writer(HistoryId history) const noexcept {
    return store_ && tail_destination_ && store_->valid(history) &&
           store_->pages_->writer_identity(tail_source_) == store_->writer(history);
}

PageSlot HistoryBinding::tail_source() const {
    if (!store_ || !tail_pinned_) throw std::logic_error("KV tail source is not ready");
    for (const auto& restore : restores_) {
        if (restore.page() == tail_source_) return restore.device();
    }
    return store_->pages().physical(tail_source_);
}

PageSlot HistoryBinding::tail_destination() const {
    if (!store_ || !tail_destination_) throw std::logic_error("KV binding has no tail copy");
    return store_->pages().physical(*tail_destination_);
}

void HistoryBinding::publish_restores() {
    if (!store_ || restores_published_) throw std::logic_error("KV restores already published");
    for (auto& restore : restores_) restore.publish();
    restores_.clear();
    restores_published_ = true;
}

void HistoryBinding::prepare_tail_copy() {
    if (!store_) throw std::logic_error("KV binding already consumed");
    if (tail_destination_ && !tail_pinned_) {
        store_->pages().pin_source(tail_source_);
        tail_pinned_ = true;
    }
}

PageDirectory& HistoryBinding::pages() const noexcept { return store_->pages(); }

void HistoryBinding::prepare_mapping(ContextTransaction& transaction) {
    if (!store_ || mapping_prepared_) throw std::logic_error("KV mapping is not reservable");
    auto& handles = store_->publish_;
    handles.clear();
    const auto& history = store_->require(this->history());
    auto restore        = restores_.begin();
    for (std::uint32_t i = 0; i < page_count(frontier_); ++i) {
        const auto page = store_->membership(history, i);
        // Restores are reserved in history order, including a restored COW source tail.
        PageSlot physical;
        if (restore != restores_.end() && restore->page() == page) {
            physical = restore->device();
            ++restore;
        } else {
            physical = store_->pages().physical(page);
        }
        if (tail_destination_ && i + 1 == page_count(frontier_)) {
            handles.push_back(tail_destination());
        } else {
            handles.push_back(physical);
        }
    }
    transaction.update_mapping(row_->handle(), 0, handles);
    mapping_prepared_ = true;
}

HistoryOwner HistoryBinding::commit() {
    if (!store_) throw std::logic_error("KV binding already consumed");
    return store_->commit(*this);
}

HistoryDirectory::HistoryDirectory(PageDirectory& pages, std::uint32_t row_count,
                                   std::uint32_t history_capacity, std::uint32_t page_capacity,
                                   HostDirectory* host)
    : pages_(&pages), host_(host), page_capacity_(page_capacity), row_generations_(row_count, 1),
      row_active_(row_count, false), histories_(history_capacity), free_(history_capacity),
      free_count_(history_capacity) {
    if (!history_capacity || !page_capacity || !row_count ||
        row_count > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("KV history capacity is inconsistent");
    for (std::uint32_t i = 0; i < history_capacity; ++i) free_[i] = history_capacity - 1 - i;
    while (directory_capacity_ < page_capacity) directory_capacity_ *= 2;
    publish_.reserve(page_capacity);
    materialize_.reserve(page_capacity);
}

HistoryDirectory::~HistoryDirectory() = default;

bool HistoryDirectory::valid(HistoryId handle) const noexcept {
    return handle.pool_id == pages_->allocator().pool_id() && handle.index < histories_.size() &&
           histories_[handle.index].occupied &&
           histories_[handle.index].generation == handle.generation;
}

bool HistoryDirectory::valid(TableRow row) const noexcept {
    return row.pool_id == pages_->allocator().pool_id() && row.index < row_active_.size() &&
           row_active_[row.index] && row_generations_[row.index] == row.generation;
}

RowLease HistoryDirectory::acquire_row(std::int32_t index) {
    if (index < 0 || static_cast<std::size_t>(index) >= row_active_.size())
        throw std::out_of_range("KV execution row is outside directory capacity");
    if (row_active_[index]) throw std::logic_error("KV execution row is already occupied");
    row_active_[index] = true;
    return RowLease(*this, {pages_->allocator().pool_id(), static_cast<std::uint32_t>(index),
                            row_generations_[index]});
}

void HistoryDirectory::release_row(TableRow row) noexcept {
    if (!valid(row)) std::terminate();
    row_active_[row.index] = false;
    auto& generation       = row_generations_[row.index];
    if (++generation == 0) generation = 1;
}

std::uint32_t HistoryDirectory::capacity() const noexcept { return histories_.size(); }

std::uint32_t HistoryDirectory::occupied() const noexcept { return capacity() - free_count_; }

HistoryDirectory::History& HistoryDirectory::require(HistoryId handle) {
    if (!valid(handle)) throw std::invalid_argument("KV history handle is stale");
    return histories_[handle.index];
}

const HistoryDirectory::History& HistoryDirectory::require(HistoryId handle) const {
    if (!valid(handle)) throw std::invalid_argument("KV history handle is stale");
    return histories_[handle.index];
}

HistoryDirectory::History& HistoryDirectory::require_owner(const HistoryOwner& owner) {
    if (owner.store_ != this)
        throw std::invalid_argument("KV history belongs to another directory");
    return require(owner.handle());
}

std::uint64_t HistoryDirectory::writer(HistoryId handle) const noexcept {
    return (static_cast<std::uint64_t>(handle.generation) << 32) | (handle.index + 1ULL);
}

HistoryId HistoryDirectory::allocate() {
    if (!free_count_) throw std::runtime_error("KV history descriptors exhausted");
    HistoryId handle;
    handle.pool_id    = pages_->allocator().pool_id();
    handle.index      = free_[--free_count_];
    auto& history     = histories_[handle.index];
    handle.generation = history.generation;
    history.occupied = history.owned = true;
    return handle;
}

bool HistoryDirectory::owned(HistoryId h) const { return require(h).owned; }

bool HistoryDirectory::active(HistoryId h) const { return require(h).row.has_value(); }

std::uint32_t HistoryDirectory::committed_frontier(HistoryId h) const {
    return require(h).frontier;
}

std::uint32_t HistoryDirectory::mapped_pages(HistoryId h) const { return require(h).page_count; }

std::int32_t HistoryDirectory::bound_row(HistoryId h) const {
    const auto& row = require(h).row;
    return row ? row->row_index() : -1;
}

PageId HistoryDirectory::logical_page(HistoryId h, std::uint32_t index) const {
    const auto& history = require(h);
    if (index >= history.page_count) throw std::out_of_range("KV block is outside history");
    return membership(history, index);
}

PageSlot HistoryDirectory::physical_page(HistoryId h, std::uint32_t index) const {
    return pages_->physical(logical_page(h, index));
}

void HistoryDirectory::update_ranges(History& history, std::uint32_t old_visible,
                                     std::uint32_t new_visible, std::uint32_t old_protected,
                                     std::uint32_t new_protected) {
    const auto update = [&](auto before, auto after, auto method) {
        if (before == after) return;
        const auto first = std::min(before, after) / kPageTokens;
        const auto last  = page_count(std::max(before, after));
        for (auto i = first; i < last; ++i)
            (pages_->*method)(membership(history, i), coverage(before, i), coverage(after, i));
    };
    update(old_visible, new_visible, &PageDirectory::update_required);
    update(old_protected, new_protected, &PageDirectory::update_protection);
}

HistoryPrefix HistoryDirectory::retain_prefix(HistoryId h, std::uint32_t frontier) {
    add_prefix(h, frontier);
    return HistoryPrefix(*this, h, frontier);
}

void HistoryDirectory::add_prefix(HistoryId h, std::uint32_t frontier) {
    auto& history = require(h);
    if (frontier > history.frontier)
        throw std::invalid_argument("KV retained prefix is outside committed history");
    const auto old = history.retained;
    ++history.prefixes[frontier];
    ++history.prefix_count;
    history.retained = std::max(old, frontier);
    update_ranges(history, std::max(history.owned ? history.frontier : 0, old),
                  std::max(history.owned ? history.frontier : 0, history.retained), old,
                  history.retained);
}

void HistoryDirectory::remove_prefix(HistoryId h, std::uint32_t frontier) noexcept {
    try {
        auto& history = require(h);
        auto it       = history.prefixes.find(frontier);
        if (it == history.prefixes.end() || !it->second) std::terminate();
        const auto old = history.retained;
        if (--it->second == 0) history.prefixes.erase(it);
        --history.prefix_count;
        history.retained = history.prefixes.empty() ? 0 : history.prefixes.rbegin()->first;
        update_ranges(history, std::max(history.owned ? history.frontier : 0, old),
                      std::max(history.owned ? history.frontier : 0, history.retained), old,
                      history.retained);
        collect(h);
    } catch (...) { std::terminate(); }
}

std::uint32_t
HistoryDirectory::retained_after(HistoryId h, std::span<const HistoryPrefix* const> removed) const {
    const auto& history = require(h);
    for (auto it = history.prefixes.rbegin(); it != history.prefixes.rend(); ++it) {
        const auto erased = std::count_if(removed.begin(), removed.end(), [&](auto* prefix) {
            return prefix && prefix->history() == h && prefix->frontier() == it->first;
        });
        if (erased > it->second)
            throw std::logic_error("KV release counts a prefix more than once");
        if (erased < it->second) return it->first;
    }
    return 0;
}

std::vector<PageId>
HistoryDirectory::released_pages(std::span<const HistoryPrefix* const> removed) const {
    std::map<std::uint32_t, HistoryId> touched;
    for (const auto* prefix : removed) {
        if (!prefix || prefix->store_ != this)
            throw std::invalid_argument("foreign KV prefix release");
        (void)require(prefix->history());
        touched.emplace(prefix->history().index, prefix->history());
    }

    struct RemovedPage {
        PageId page;
        std::uint32_t references = 0;
    };

    std::map<std::uint32_t, RemovedPage> candidates;
    for (const auto& [index, handle] : touched) {
        const auto& history = require(handle);
        if (history.owned) continue;
        const auto retained = retained_after(handle, removed);
        for (auto i = page_count(retained); i < history.page_count; ++i) {
            auto page   = membership(history, i);
            auto& entry = candidates[pages_->descriptor_index(page)];
            entry.page  = page;
            ++entry.references;
        }
    }
    std::vector<PageId> result;
    for (const auto& [index, entry] : candidates)
        if (entry.references == pages_->history_references(entry.page) &&
            pages_->active_history_references(entry.page) == 0 &&
            pages_->source_pins(entry.page) == 0)
            result.push_back(entry.page);
    return result;
}

void HistoryDirectory::truncate_membership(History& history, std::uint32_t count) {
    if (count == history.page_count) return;
    auto directory = prefix_directory(history.directory, directory_capacity_, count);
    std::vector<PageId> host_releases;
    host_releases.reserve(history.page_count - count);
    for (std::uint32_t i = count; i < history.page_count; ++i)
        if (!pages_->can_release_reference(membership(history, i)))
            throw std::logic_error("KV history still has an unowned reader");
    for (std::uint32_t i = count; i < history.page_count; ++i) {
        auto page       = membership(history, i);
        const bool host = pages_->host_resident(page);
        if (!pages_->release_reference(page))
            throw std::logic_error("KV history still has an unowned reader");
        if (host && pages_->history_references(page) == 0 && pages_->source_pins(page) == 0)
            host_releases.push_back(page);
    }
    history.directory  = std::move(directory);
    history.page_count = count;
    if (host_ && !host_releases.empty() && !host_->release_page_replicas(*pages_, host_releases))
        throw std::logic_error("unowned Host KV pages could not be retired");
}

void HistoryDirectory::collect(HistoryId h) {
    auto& history = require(h);
    if (history.owned) return;
    truncate_membership(history, page_count(history.retained));
    history.frontier = history.retained;
    if (history.prefix_count) return;
    auto generation      = history.generation + 1;
    history              = History{};
    history.generation   = generation ? generation : 1;
    free_[free_count_++] = h.index;
}

void HistoryDirectory::release_owner(HistoryId h) noexcept {
    try {
        auto& history = require(h);
        if (!history.owned) std::terminate();
        if (history.row) deactivate_retired(h);
        const auto before = std::max(history.frontier, history.retained);
        history.reservation.release();
        history.owned = false;
        update_ranges(history, before, history.retained, history.retained, history.retained);
        collect(h);
    } catch (...) { std::terminate(); }
}

HistoryOwner HistoryDirectory::fork_history(const HistoryPrefix& source, std::uint32_t frontier) {
    if (!source.valid() || source.store_ != this || frontier > source.frontier())
        throw std::invalid_argument("KV binding source belongs to another history");
    auto& from = require(source.history());
    HistoryOwner result(*this, allocate());
    auto& to     = require(result.handle());
    to.directory = prefix_directory(from.directory, directory_capacity_, page_count(frontier));
    to.frontier  = frontier;
    for (std::uint32_t i = 0; i < page_count(frontier); ++i) {
        auto page = membership(to, i);
        pages_->retain_reference(page);
        pages_->update_required(page, 0, coverage(frontier, i));
        ++to.page_count;
    }
    return result;
}

HistoryBindingDemand
HistoryDirectory::binding_demand(const HistoryPrefix* source, std::uint32_t frontier,
                                 std::uint32_t peak,
                                 std::span<const HistoryPrefix* const> retired) const {
    if (frontier > peak || page_count(peak) > page_capacity_ || (!source && frontier))
        throw std::invalid_argument("KV binding frontier is invalid");
    HistoryBindingDemand result{.growth_pages = page_count(peak) - page_count(frontier)};
    if (!source) return result;
    if (source->store_ != this || frontier > source->frontier())
        throw std::invalid_argument("KV prefix does not cover binding");
    const auto& history  = require(source->history());
    result.reuse_history = !history.owned && retained_after(source->history(), retired) <= frontier;
    for (std::uint32_t i = 0; i < page_count(frontier); ++i) {
        auto page         = membership(history, i);
        const auto needed = coverage(frontier, i);
        if (!pages_->device_covers(page, needed)) {
            if (!pages_->host_covers(page, needed))
                throw std::logic_error("KV prefix has no covering replica");
            ++result.restore_pages;
        }
    }
    if (frontier % kPageTokens) {
        const auto tail = membership(history, page_count(frontier) - 1);
        const auto id   = result.reuse_history ? writer(source->history()) : UINT64_MAX;
        const bool writable =
            pages_->writer_identity(tail) == 0 || pages_->writer_identity(tail) == id;
        auto required = pages_->required_references_above(tail, frontier % kPageTokens);
        std::vector<HistoryId> touched;
        for (const auto* prefix : retired) {
            if (!prefix || prefix->store_ != this)
                throw std::invalid_argument("foreign retired KV prefix");
            const auto h = prefix->history();
            if (std::find(touched.begin(), touched.end(), h) != touched.end()) continue;
            touched.push_back(h);
            const auto& other = require(h);
            const auto index  = page_count(frontier) - 1;
            if (other.page_count <= index || membership(other, index) != tail) continue;
            const auto before = std::max(other.owned ? other.frontier : 0, other.retained);
            const auto after =
                std::max(other.owned ? other.frontier : 0, retained_after(h, retired));
            required -= coverage(before, index) > frontier % kPageTokens;
            required += coverage(after, index) > frontier % kPageTokens;
        }
        result.tail_copies = !writable || pages_->source_pins(tail) != 0 || required != 0;
        if (result.tail_copies) result.reuse_history = false;
    }
    return result;
}

HistoryBinding HistoryDirectory::prepare_binding(const HistoryPrefix* source,
                                                 std::uint32_t frontier, std::uint32_t peak,
                                                 std::int32_t execution_row, bool consume_source) {
    const std::array<const HistoryPrefix*, 1> consumed{source};
    const auto demand =
        binding_demand(source, frontier, peak,
                       consume_source && source ? std::span<const HistoryPrefix* const>(consumed)
                                                : std::span<const HistoryPrefix* const>{});
    HistoryBinding binding;
    binding.store_      = this;
    binding.frontier_   = frontier;
    binding.write_tail_ = frontier % kPageTokens;
    if (source) binding.source_ = *source;
    if (source && !demand.reuse_history) {
        binding.history_ = fork_history(*source, frontier);
    } else {
        if (source) {
            auto& original   = require(source->history());
            original.owned   = true;
            binding.history_ = HistoryOwner(*this, source->history());
        } else {
            binding.history_ = HistoryOwner(*this, allocate());
        }
    }
    binding.reservation_ = pages_->allocator().make_empty_reservation();
    pages_->allocator().resize_reservation(binding.reservation_, demand.device_pages());
    binding.row_.emplace(acquire_row(execution_row));
    const auto& history = require(binding.history());
    binding.restores_.reserve(demand.restore_pages);
    for (std::uint32_t i = 0; i < page_count(frontier); ++i) {
        auto page = membership(history, i);
        if (pages_->device_covers(page, coverage(frontier, i))) continue;
        binding.restores_.push_back(pages_->reserve_device_replica(page, binding.reservation_));
    }
    if (demand.tail_copies) {
        binding.tail_source_ = membership(history, page_count(frontier) - 1);
        binding.tail_destination_ =
            pages_->materialize_transfer_destination(binding.reservation_, frontier % kPageTokens);
    }
    return binding;
}

void HistoryDirectory::abort(HistoryBinding& binding) noexcept {
    try {
        if (binding.tail_pinned_) pages_->unpin_source(binding.tail_source_);
        if (binding.tail_destination_)
            pages_->abort_transfer_destination(*binding.tail_destination_, binding.reservation_);
        binding.restores_.clear();
        binding.store_ = nullptr;
        binding.row_.reset();
        binding.reservation_.release();
        binding.history_.reset();
        binding.source_.reset();
    } catch (...) { std::terminate(); }
}

HistoryOwner HistoryDirectory::commit(HistoryBinding& binding) {
    if (!binding.restores_published_ || !binding.row_ || !binding.mapping_prepared_)
        throw std::logic_error("KV binding has unfinished replicas");
    auto& history = require(binding.history());
    binding.source_.reset();
    if (history.retained > binding.frontier_)
        throw std::logic_error("KV binding still retains a prefix beyond its adopted frontier");
    update_ranges(history, history.frontier, binding.frontier_, history.retained, history.retained);
    truncate_membership(history, page_count(binding.frontier_));
    history.frontier = binding.frontier_;
    if (binding.tail_destination_) {
        if (!binding.tail_pinned_) throw std::logic_error("KV tail copy was not prepared");
        const auto index = history.page_count - 1;
        auto old         = membership(history, index);
        const auto count = coverage(history.frontier, index);
        // A reused history must not replace a mapping retained by one of its prefixes.
        if (history.retained > index * kPageTokens)
            throw std::logic_error("COW binding must branch the retained history");
        auto& slot = directory_slot(history.directory, index);
        pages_->publish_transfer_destination(*binding.tail_destination_);
        pages_->update_required(old, count, 0);
        pages_->update_required(*binding.tail_destination_, 0, count);
        slot = *binding.tail_destination_;
        if (!pages_->release_reference(old))
            throw std::logic_error("KV COW source is still borrowed");
        pages_->unpin_source(old);
        binding.tail_pinned_ = false;
        binding.tail_destination_.reset();
    }
    if (binding.write_tail_) {
        const auto tail = membership(history, history.page_count - 1);
        pages_->prepare_write(tail, writer(binding.history()), history.frontier % kPageTokens,
                              history.frontier % kPageTokens);
    }
    history.row.emplace(std::move(*binding.row_));
    binding.row_.reset();
    history.reservation = std::move(binding.reservation_);
    for (std::uint32_t i = 0; i < history.page_count; ++i)
        pages_->retain_active_reference(membership(history, i));
    binding.store_ = nullptr;
    return std::move(binding.history_);
}

void HistoryDirectory::deactivate(HistoryId handle, ContextTransaction& transaction) {
    auto& history = require(handle);
    if (!history.row) return;
    for (std::uint32_t i = 0; i < history.page_count; ++i)
        transaction.retain(*pages_, membership(history, i));
    transaction.retire(std::move(*history.row));
    history.row.reset();
    for (std::uint32_t i = 0; i < history.page_count; ++i) {
        const auto page = membership(history, i);
        pages_->release_writer(page, writer(handle));
        pages_->release_active_reference(page);
    }
    history.reservation.release();
}

void HistoryDirectory::deactivate_retired(HistoryId handle) {
    auto& history = require(handle);
    if (!history.row) return;
    for (std::uint32_t i = 0; i < history.page_count; ++i) {
        auto page = membership(history, i);
        pages_->release_writer(page, writer(handle));
        pages_->release_active_reference(page);
    }
    history.row.reset();
    history.reservation.release();
}

const RowLease& HistoryDirectory::execution_row(HistoryId h) const {
    const auto& row = require(h).row;
    if (!row) throw std::logic_error("KV history has no execution row");
    return *row;
}

void HistoryDirectory::settle_growth(HistoryOwner& owner, std::uint32_t frontier,
                                     std::uint32_t retain_through,
                                     ContextTransaction& transaction) {
    (void)require_owner(owner);
    if (frontier > committed_frontier(owner.handle())) commit_frontier(owner.handle(), frontier);
    trim(owner, frontier, transaction);
    const auto pages =
        retain_through > frontier ? write_pages(owner.handle(), frontier, retain_through) : 0;
    transaction.reserve_after_retirement(*this, owner.handle(), pages);
}

void HistoryDirectory::prepare_commit(HistoryOwner& owner, std::uint32_t frontier,
                                      std::uint32_t retain_through,
                                      ContextTransaction& transaction) {
    const auto handle = owner.handle();
    auto& history     = require_owner(owner);
    if (!history.row || frontier < history.frontier || page_count(frontier) > history.page_count)
        throw std::invalid_argument("KV commit is outside the prepared range");
    if (frontier > history.frontier)
        for (auto i = history.frontier / kPageTokens; i < page_count(frontier); ++i)
            if (pages_->writer_identity(membership(history, i)) != writer(handle))
                throw std::logic_error("KV commit has no writer");
    transaction.commit_history(*this, handle, frontier, retain_through);
    const auto retained = page_count(std::max(frontier, history.retained));
    for (auto i = retained; i < history.page_count; ++i)
        transaction.retain(*pages_, membership(history, i));
    for (auto i = retained; i < history.page_count; ++i) {
        const auto page = membership(history, i);
        pages_->release_writer(page, writer(handle));
        pages_->release_active_reference(page);
    }
    truncate_membership(history, retained);
}

void HistoryDirectory::reserve_growth(HistoryId handle, std::uint32_t count) {
    auto& history = require(handle);
    if (!history.owned || !history.row || count > page_capacity_)
        throw std::invalid_argument("KV growth does not fit the active history");
    pages_->allocator().resize_reservation(history.reservation, count);
}

void HistoryDirectory::release_growth(HistoryId handle) { reserve_growth(handle, 0); }

std::uint32_t HistoryDirectory::reserved_growth_pages(HistoryId h) const {
    return require(h).reservation.pages();
}

std::uint32_t HistoryDirectory::growth_pages_for_tokens(HistoryId h, std::uint32_t frontier) const {
    const auto& history = require(h);
    if (page_count(frontier) > page_capacity_)
        throw std::out_of_range("KV growth exceeds history capacity");
    return page_count(frontier) > history.page_count ? page_count(frontier) - history.page_count
                                                     : 0;
}

std::uint32_t HistoryDirectory::write_pages(HistoryId handle, std::uint32_t begin,
                                            std::uint32_t end) const {
    const auto& history = require(handle);
    if (begin > history.frontier || begin > end || page_count(end) > page_capacity_)
        throw std::invalid_argument("KV write interval is invalid");
    auto count = growth_pages_for_tokens(handle, end);
    if (begin == end) return count;
    for (auto i = begin / kPageTokens; i < std::min(page_count(end), history.page_count); ++i)
        count += !pages_->can_write(membership(history, i), writer(handle),
                                    i == begin / kPageTokens ? begin % kPageTokens : 0,
                                    coverage(history.frontier, i));
    return count;
}

void HistoryDirectory::prepare_write(HistoryOwner& owner, std::uint32_t begin, std::uint32_t end,
                                     ContextTransaction& transaction) {
    (void)require_owner(owner);
    auto handle = owner.handle();
    if (begin < require(handle).frontier) trim(owner, begin, transaction);
    handle         = owner.handle();
    auto& original = require(handle);
    if (!original.row || !original.owned || begin != original.frontier || begin > end)
        throw std::invalid_argument("KV write must start at its actual frontier");
    const auto copies = write_pages(handle, begin, end);
    if (copies > original.reservation.pages())
        throw std::logic_error("KV write exceeds its page reservation");
    if (begin == end) return;
    const auto first = begin / kPageTokens;
    if (first < original.page_count && original.retained > first * kPageTokens &&
        !pages_->can_write(membership(original, first), writer(handle), begin % kPageTokens,
                           coverage(original.frontier, first))) {
        branch_owner(owner, begin, transaction);
        handle = owner.handle();
    }
    auto& history        = require(handle);
    const auto old_count = history.page_count;
    const auto target    = page_count(end);
    auto changed         = target;
    for (auto i = old_count; i < target; ++i) (void)directory_slot(history.directory, i);
    if (target > old_count) {
        materialize_.assign(target - old_count, {});
        pages_->materialize(history.reservation, materialize_,
                            old_count ? std::optional(membership(history, old_count - 1))
                                      : std::nullopt);
        for (auto page : materialize_) {
            directory_slot(history.directory, history.page_count++) = page;
            pages_->retain_active_reference(page);
        }
        changed = old_count;
    }
    for (auto i = first; i < target; ++i) {
        auto page          = membership(history, i);
        const auto offset  = i == first ? begin % kPageTokens : 0;
        const auto visible = coverage(history.frontier, i);
        if (!pages_->can_write(page, writer(handle), offset, visible)) {
            if (pages_->history_references(page) == 1)
                throw std::logic_error("KV write is waiting on its sole replica reader");
            auto& slot = directory_slot(history.directory, i);
            auto replacement =
                pages_->materialize_transfer_destination(history.reservation, std::max(1U, offset));
            try {
                transaction.copy_page(*pages_, page, replacement);
            } catch (...) {
                pages_->abort_transfer_destination(replacement, history.reservation);
                throw;
            }
            pages_->publish_transfer_destination(replacement);
            pages_->update_required(page, visible, 0);
            pages_->update_required(replacement, 0, visible);
            pages_->release_active_reference(page);
            pages_->release_writer(page, writer(handle));
            slot = replacement;
            pages_->retain_active_reference(replacement);
            if (!pages_->release_reference(page))
                throw std::logic_error("KV copied source cannot retire");
            page    = replacement;
            changed = std::min(changed, i);
        }
        pages_->prepare_write(page, writer(handle), offset, visible);
    }
    if (changed < target) {
        publish_.clear();
        for (auto i = changed; i < target; ++i)
            publish_.push_back(pages_->physical(membership(history, i)));
        transaction.update_mapping(history.row->handle(), changed, publish_);
    }
}

void HistoryDirectory::commit_frontier(HistoryId handle, std::uint32_t frontier) {
    auto& history = require(handle);
    if (!history.row || frontier < history.frontier || page_count(frontier) > history.page_count)
        throw std::invalid_argument("KV commit is outside the prepared range");
    if (frontier == history.frontier) return;
    for (auto i = history.frontier / kPageTokens; i < page_count(frontier); ++i) {
        auto page = membership(history, i);
        if (pages_->writer_identity(page) != writer(handle))
            throw std::logic_error("KV commit has no writer");
        pages_->commit_coverage(page, coverage(frontier, i));
    }
    update_ranges(history, std::max(history.frontier, history.retained),
                  std::max(frontier, history.retained), history.retained, history.retained);
    history.frontier = frontier;
}

void HistoryDirectory::branch_owner(HistoryOwner& owner, std::uint32_t frontier,
                                    ContextTransaction& transaction) {
    const auto handle   = owner.handle();
    auto source         = retain_prefix(handle, frontier);
    HistoryOwner branch = fork_history(source, frontier);
    auto& from          = require(handle);
    auto& to            = require(branch.handle());
    if (from.row) {
        for (auto i = to.page_count; i < from.page_count; ++i)
            transaction.retain(*pages_, membership(from, i));
        for (std::uint32_t i = 0; i < from.page_count; ++i) {
            auto page = membership(from, i);
            pages_->release_writer(page, writer(handle));
            pages_->release_active_reference(page);
        }
        to.row = std::move(from.row);
        from.row.reset();
        to.reservation = std::move(from.reservation);
        for (std::uint32_t i = 0; i < to.page_count; ++i)
            pages_->retain_active_reference(membership(to, i));
        publish_.clear();
        for (std::uint32_t i = 0; i < to.page_count; ++i)
            publish_.push_back(pages_->physical(membership(to, i)));
        transaction.update_mapping(to.row->handle(), 0, publish_);
    }
    owner = std::move(branch);
}

void HistoryDirectory::trim(HistoryOwner& owner, std::uint32_t frontier,
                            ContextTransaction& transaction) {
    const auto handle = owner.handle();
    auto& history     = require_owner(owner);
    if (frontier > history.frontier) throw std::invalid_argument("KV trim cannot extend history");
    if (frontier < history.retained) {
        branch_owner(owner, frontier, transaction);
        return;
    }
    if (history.row) {
        for (auto i = page_count(frontier); i < history.page_count; ++i)
            transaction.retain(*pages_, membership(history, i));
    }
    update_ranges(history, std::max(history.frontier, history.retained),
                  std::max(frontier, history.retained), history.retained, history.retained);
    for (auto i = page_count(frontier); i < history.page_count; ++i) {
        auto page = membership(history, i);
        pages_->release_writer(page, writer(handle));
        if (history.row) pages_->release_active_reference(page);
    }
    truncate_membership(history, page_count(frontier));
    history.frontier = frontier;
}

PageId HistoryDirectory::membership(const History& history, std::uint32_t index) const noexcept {
    const auto* node = history.directory.get();
    auto width       = directory_capacity_;
    while (width > kChunkPages) {
        width /= 2;
        if (index < width)
            node = node->left.get();
        else {
            index -= width;
            node = node->right.get();
        }
    }
    return node->pages[index];
}

PageId& HistoryDirectory::directory_slot(Directory& root, std::uint32_t index) {
    auto* node = &root;
    auto width = directory_capacity_;
    for (;;) {
        if (!*node)
            *node = std::make_shared<DirectoryNode>();
        else if (!node->unique())
            *node = std::make_shared<DirectoryNode>(**node);
        if (width == kChunkPages) return (*node)->pages[index];
        width /= 2;
        if (index < width)
            node = &(*node)->left;
        else {
            index -= width;
            node = &(*node)->right;
        }
    }
}

HistoryDirectory::Directory HistoryDirectory::prefix_directory(const Directory& node,
                                                               std::uint64_t width,
                                                               std::uint32_t count) {
    if (!node || !count) return {};
    if (count >= width) return node;
    auto result = std::make_shared<DirectoryNode>(*node);
    if (width == kChunkPages)
        std::fill(result->pages.begin() + count, result->pages.end(), PageId{});
    else {
        width /= 2;
        if (count <= width) {
            result->left = prefix_directory(node->left, width, count);
            result->right.reset();
        } else
            result->right = prefix_directory(node->right, width, count - width);
    }
    return result;
}
} // namespace ninfer::models::context
