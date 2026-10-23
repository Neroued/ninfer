#include "models/common/context/directory/history.h"

#include "models/common/context/directory.h"

#include <algorithm>
#include <limits>
#include <map>
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

std::uint32_t directory_depth(std::uint32_t page_capacity) {
    std::uint32_t result = 1;
    for (std::uint64_t width = 32; width < page_capacity; width *= 2) ++result;
    return result;
}

std::uint32_t node_limit(std::uint32_t pages, std::uint32_t histories,
                         std::uint32_t page_capacity) {
    const auto limit =
        std::uint64_t(pages) * 2 + std::uint64_t(histories) * directory_depth(page_capacity) * 4;
    if (limit > UINT32_MAX) throw std::overflow_error("KV directory metadata limit exceeds uint32");
    return static_cast<std::uint32_t>(limit);
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
      frontier_(other.frontier_), mapping_prepared_(other.mapping_prepared_),
      conversion_pages_(std::move(other.conversion_pages_)),
      conversion_readers_(std::move(other.conversion_readers_)),
      conversion_layout_(std::move(other.conversion_layout_)),
      conversion_prepared_(other.conversion_prepared_) {}

bool HistoryBinding::blocks_writer(HistoryId history) const noexcept {
    return store_ && tail_destination_ && store_->valid(history) &&
           store_->pages_->writer_identity(tail_source_) == store_->writer(history);
}

ContextRecordBinding HistoryBinding::tail_source(std::uint32_t rank) const {
    if (!store_ || !tail_pinned_) throw std::logic_error("KV tail source is not ready");
    for (const auto& restore : restores_) {
        if (restore.page() == tail_source_ && restore.rank() == rank) return restore.device();
    }
    return store_->pages().physical(tail_source_, rank);
}

ContextRecordBinding HistoryBinding::tail_destination(std::uint32_t rank) const {
    if (!store_ || !tail_destination_) throw std::logic_error("KV binding has no tail copy");
    return store_->pages().physical(*tail_destination_, rank);
}

void HistoryBinding::publish_restores() {
    if (!store_ || restores_published_) throw std::logic_error("KV restores already published");
    for (auto& restore : restores_) restore.publish();
    restores_.clear();
    auto& history = store_->require(this->history());
    for (const auto& page : conversion_pages_) {
        store_->pages().publish_transfer_destination(page.destination);
        ++history.page_count;
    }
    conversion_pages_.clear();
    conversion_readers_.clear();
    restores_published_ = true;
}

void HistoryBinding::prepare_conversion(ContextTransaction& transaction) {
    if (!store_ || conversion_prepared_)
        throw std::logic_error("KV conversion actions are already prepared");
    for (const auto& page : conversion_pages_) {
        const auto& ranges = page.columns == kPageTokens ? conversion_layout_.full_page
                                                         : conversion_layout_.partial_page;
        for (const auto& field : ranges)
            transaction.copy_range(
                store_->pages().physical(page.source, field.source_rank),
                store_->pages().prepared_physical(page.destination, field.destination_rank), field,
                {runtime::KvPoolId{static_cast<std::uint16_t>(history().pool_id)}});
    }
    conversion_prepared_ = true;
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
    for (std::uint32_t i = 0; i < page_count(frontier_); ++i)
        handles.push_back(!conversion_pages_.empty() ? conversion_pages_[i].destination
                          : tail_destination_ && i + 1 == page_count(frontier_)
                              ? *tail_destination_
                              : store_->membership(history, i));
    store_->update_mapping(row_->handle(), 0, handles, transaction);
    mapping_prepared_ = true;
}

std::span<const std::uint32_t> HistoryBinding::ranks() const noexcept {
    return store_->require(history()).placement;
}

HistoryOwner HistoryBinding::commit() {
    if (!store_) throw std::logic_error("KV binding already consumed");
    return store_->commit(*this);
}

HistoryDirectory::HistoryDirectory(PageDirectory& pages, MetadataBudget& metadata,
                                   std::uint32_t row_count, std::uint32_t history_capacity,
                                   std::uint32_t page_capacity)
    : metadata_(metadata), pages_(&pages), page_capacity_(page_capacity),
      row_generations_(row_count, 1), row_active_(row_count, false),
      nodes_(node_limit(pages.capacity(), history_capacity, page_capacity), metadata),
      histories_(history_capacity, metadata), free_count_(0), row_pages_(row_count) {
    if (!history_capacity || !page_capacity || !row_count || row_count > INT32_MAX)
        throw std::invalid_argument("KV history capacity is inconsistent");
    while (directory_capacity_ < page_capacity) directory_capacity_ *= 2;
    tree_depth_ = directory_depth(page_capacity);
    metadata_.acquire(row_generations_.capacity() * sizeof(std::uint32_t) +
                      (row_active_.capacity() + 7) / 8 +
                      row_pages_.capacity() * sizeof(std::vector<PageId>));
    metadata_.reserve(publish_, page_capacity);
    metadata_.reserve(materialize_, page_capacity);
    for (auto& row : row_pages_) metadata_.reserve(row, page_capacity);
    pages.reserve_table_capacity(row_count);
}

std::size_t HistoryDirectory::metadata_bytes(std::uint32_t histories, std::uint32_t rows,
                                             std::uint32_t page_capacity, std::uint32_t ranks) {
    const auto nodes = std::size_t(histories) * directory_depth(page_capacity) * 4;
    return sizeof(HistoryDirectory) +
           std::size_t(histories) *
               (sizeof(History) + (std::size_t(ranks) + 1) * sizeof(std::uint32_t)) +
           nodes * (sizeof(DirectoryNode) + 2 * sizeof(DirectoryNode*)) +
           std::size_t(rows) * (sizeof(std::uint32_t) + sizeof(std::vector<PageId>) +
                                std::size_t(page_capacity) * sizeof(PageId)) +
           ((std::size_t(rows) + 63) / 64) * 8 + std::size_t(page_capacity) * sizeof(PageId) * 2;
}

std::size_t HistoryDirectory::page_metadata_bytes() noexcept {
    return 2 * (sizeof(DirectoryNode) + 2 * sizeof(DirectoryNode*));
}

void HistoryDirectory::reserve_metadata(std::uint32_t count) {
    if (count > free_count_) {
        const auto needed = std::uint64_t(initialized_) + count - free_count_;
        if (needed > histories_.limit()) throw std::runtime_error("KV history metadata exhausted");
        const auto target = std::min<std::uint64_t>(histories_.limit(), (needed + 255) / 256 * 256);
        metadata_.reserve(free_, std::max<std::uint64_t>(histories_.size(), target));
        free_.resize(std::max<std::uint64_t>(histories_.size(), target));
        histories_.grow(needed);
        while (initialized_ < histories_.size()) free_[free_count_++] = initialized_++;
    }
    for (auto i = free_count_ - count; i < free_count_; ++i) {
        histories_.ensure(free_[i]);
        metadata_.reserve(histories_[free_[i]].placement, pages_->layout().parts.size());
    }
}

MetadataReservation HistoryDirectory::reserve_descriptors(std::uint32_t count) {
    reserve_metadata(count);
    std::vector<std::uint32_t> ids;
    ids.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto index = free_[--free_count_];
        auto& history    = histories_[index];
        if (history.occupied || history.reserved) std::terminate();
        history.reserved = true;
        ids.push_back(index);
    }
    return MetadataReservation(*this, std::move(ids));
}

void HistoryDirectory::release_metadata(std::span<const std::uint32_t> ids) noexcept {
    for (auto index : ids) {
        if (!histories_.is_allocated(index)) std::terminate();
        auto& history = histories_[index];
        if (!history.reserved || history.occupied) std::terminate();
        release_spares(history);
        history.branch_descriptor.reset();
        history.reserved     = false;
        free_[free_count_++] = index;
    }
}

HistoryDirectory::MetadataRetirement
HistoryDirectory::metadata_retirement(std::span<const HistoryId> released) const {
    MetadataRetirement result;
    std::vector<const DirectoryNode*> pending;
    for (const auto handle : released) {
        const auto& history = require(handle);
        if (history.owned)
            throw std::logic_error("active KV writer cannot be retired by a cache quote");
        if (std::find(result.histories.begin(), result.histories.end(), handle.index) !=
            result.histories.end())
            continue;
        if (history.branch_descriptor.remaining()) {
            const auto branch = history.branch_descriptor.last();
            result.histories.push_back(branch);
            for (const auto* node : histories_[branch].spare_nodes)
                result.nodes.push_back(node->index);
        }
        result.histories.push_back(handle.index);
        for (const auto* node : history.spare_nodes) result.nodes.push_back(node->index);
        if (history.directory) pending.push_back(history.directory.get());
    }
    std::map<std::uint32_t, std::uint32_t> dropped;
    while (!pending.empty()) {
        const auto* node = pending.back();
        pending.pop_back();
        if (++dropped[node->index] != node->references) continue;
        result.nodes.push_back(node->index);
        if (node->left) pending.push_back(node->left.get());
        if (node->right) pending.push_back(node->right.get());
    }
    return result;
}

std::size_t HistoryDirectory::reclaimable_metadata(std::span<const HistoryId> released) const {
    const auto retired = metadata_retirement(released);
    std::size_t bytes  = 0;
    for (std::uint32_t first = 0; first < histories_.size(); first += 256) {
        if (!histories_.is_allocated(first)) continue;
        const auto end      = first + std::min(256U, histories_.size() - first);
        std::size_t segment = 0;
        for (auto i = first; i < end; ++i) {
            const auto& history = histories_[i];
            if ((history.occupied || history.reserved) &&
                std::find(retired.histories.begin(), retired.histories.end(), i) ==
                    retired.histories.end()) {
                segment = 0;
                break;
            }
            segment +=
                sizeof(History) + history.placement.capacity() * sizeof(std::uint32_t) +
                history.spare_nodes.capacity() * sizeof(DirectoryNode*) +
                history.prefixes.capacity() * sizeof(std::pair<std::uint32_t, std::uint32_t>);
        }
        bytes += segment;
    }
    for (std::uint32_t first = 0; first < nodes_.size(); first += 256) {
        if (!nodes_.is_allocated(first)) continue;
        const auto end = first + std::min(256U, nodes_.size() - first);
        bool empty     = true;
        for (auto i = first; i < end; ++i)
            empty &=
                (!nodes_[i].references && !nodes_[i].reserved) ||
                std::find(retired.nodes.begin(), retired.nodes.end(), i) != retired.nodes.end();
        if (empty) bytes += std::size_t(end - first) * sizeof(DirectoryNode);
    }
    return bytes;
}

void HistoryDirectory::reclaim_metadata() noexcept {
    for (std::uint32_t first = 0; first < histories_.size(); first += 256) {
        if (!histories_.is_allocated(first)) continue;
        const auto end = std::min<std::uint64_t>(histories_.size(), std::uint64_t(first) + 256);
        bool empty     = true;
        for (auto i = first; i < end; ++i)
            empty &= !histories_[i].occupied && !histories_[i].reserved;
        if (!empty) continue;
        for (auto i = first; i < end; ++i) {
            auto& history = histories_[i];
            metadata_.release(history.placement.capacity() * sizeof(std::uint32_t) +
                              history.spare_nodes.capacity() * sizeof(DirectoryNode*) +
                              history.prefixes.capacity() *
                                  sizeof(std::pair<std::uint32_t, std::uint32_t>));
        }
        histories_.release_segment(first);
    }
    for (std::uint32_t first = 0; first < nodes_.size(); first += 256) {
        if (!nodes_.is_allocated(first)) continue;
        const auto end = std::min<std::uint64_t>(nodes_.size(), std::uint64_t(first) + 256);
        bool empty     = true;
        for (auto i = first; i < end; ++i) empty &= !nodes_[i].references && !nodes_[i].reserved;
        if (!empty) continue;
        free_nodes_.erase(std::remove_if(free_nodes_.begin(), free_nodes_.end(),
                                         [first, end](const DirectoryNode* node) {
                                             return node->index >= first && node->index < end;
                                         }),
                          free_nodes_.end());
        nodes_.release_segment(first);
    }
}

std::optional<std::size_t>
HistoryDirectory::metadata_requirement(std::span<const HistoryMetadataDemand> demands,
                                       std::span<const HistoryId> released) const {
    if (demands.empty()) return 0;
    const auto retired = metadata_retirement(released);
    auto free_count    = free_count_;
    auto retired_count = retired.histories.size();
    auto materialized  = histories_.size();
    auto new_begin = materialized, new_end = materialized;
    std::vector<std::uint32_t> segments;
    std::size_t bytes = 0, extra_nodes = 0;
    const auto vector_bytes = [](std::size_t capacity, std::size_t required, std::size_t element) {
        return required > capacity ? (required - capacity) * element : 0;
    };
    const auto descriptor = [&]() -> std::optional<std::uint32_t> {
        std::uint32_t index;
        if (retired_count) {
            index = retired.histories[--retired_count];
        } else if (free_count) {
            index = free_[--free_count];
        } else {
            if (new_begin == new_end) {
                if (materialized == histories_.limit()) return std::nullopt;
                const auto end =
                    std::min<std::uint64_t>(histories_.limit(), std::uint64_t(materialized) + 256);
                new_begin    = materialized;
                new_end      = end;
                materialized = end;
            }
            index = --new_end;
        }
        if (!histories_.is_allocated(index) &&
            std::find(segments.begin(), segments.end(), index / 256) == segments.end()) {
            segments.push_back(index / 256);
            bytes += std::size_t(std::min(256U, histories_.limit() - index / 256 * 256)) *
                     sizeof(History);
        }
        const auto capacity =
            histories_.is_allocated(index) ? histories_[index].placement.capacity() : 0;
        bytes += vector_bytes(capacity, pages_->layout().parts.size(), sizeof(std::uint32_t));
        return index;
    };
    const auto prepare = [&](std::uint32_t index, std::size_t nodes, std::size_t prefixes) {
        const auto* history = histories_.is_allocated(index) ? &histories_[index] : nullptr;
        const auto held = history && std::find(retired.histories.begin(), retired.histories.end(),
                                               index) == retired.histories.end()
                              ? history->spare_nodes.size()
                              : 0;
        extra_nodes += nodes > held ? nodes - held : 0;
        bytes += vector_bytes(history ? history->spare_nodes.capacity() : 0, nodes,
                              sizeof(DirectoryNode*));
        bytes += vector_bytes(history ? history->prefixes.capacity() : 0, prefixes,
                              sizeof(std::pair<std::uint32_t, std::uint32_t>));
    };
    std::vector<HistoryId> writers;
    for (const auto& demand : demands) {
        std::uint32_t index;
        if (demand.history.valid()) {
            (void)require(demand.history);
            if (std::find(writers.begin(), writers.end(), demand.history) != writers.end())
                throw std::invalid_argument("duplicate KV writer metadata demand");
            writers.push_back(demand.history);
            index = demand.history.index;
        } else {
            const auto next = descriptor();
            if (!next) return std::nullopt;
            index = *next;
        }
        const auto* history = demand.history.valid() ? &histories_[index] : nullptr;
        const auto prefixes = history ? history->prefixes.size() : 0;
        prepare(index,
                (std::size_t(demand.growth_pages) + prefixes + (history ? 4 : 6)) * tree_depth_,
                prefixes + 1);
        if (history && history->branch_descriptor.remaining()) {
            prepare(history->branch_descriptor.last(), (prefixes + 4) * tree_depth_, 0);
        } else {
            const auto branch = descriptor();
            if (!branch) return std::nullopt;
            prepare(*branch, (prefixes + 4) * tree_depth_, 0);
        }
    }
    bytes += vector_bytes(free_.capacity(), materialized, sizeof(std::uint32_t));
    // growth_bytes includes newly allocated segments and their pointer table.
    const auto new_payload = std::size_t(materialized - histories_.size()) * sizeof(History);
    bytes += histories_.growth_bytes(materialized) - new_payload;
    auto available_nodes = free_nodes_.size() + retired.nodes.size();
    auto node_size       = nodes_.size();
    for (std::uint32_t first = 0; available_nodes < extra_nodes && first < nodes_.limit();
         first += 256) {
        if (nodes_.is_allocated(first)) continue;
        const auto count = std::min(256U, nodes_.limit() - first);
        available_nodes += count;
        bytes += std::size_t(count) * sizeof(DirectoryNode);
        node_size = std::max(node_size, first + count);
    }
    if (available_nodes < extra_nodes) return std::nullopt;
    bytes += vector_bytes(free_nodes_.capacity(), node_size, sizeof(DirectoryNode*));
    bytes += nodes_.growth_bytes(node_size) -
             std::size_t(node_size - nodes_.size()) * sizeof(DirectoryNode);
    return bytes;
}

MetadataReservation HistoryDirectory::reserve_control(const HistoryMetadataDemand& demand) {
    MetadataReservation descriptor;
    History* history;
    if (demand.history.valid()) {
        history = &require(demand.history);
    } else {
        descriptor = reserve_descriptors(1);
        history    = &histories_[descriptor.last()];
    }
    reserve_nodes(*history, (std::size_t(demand.growth_pages) + history->prefixes.size() +
                             (demand.history.valid() ? 4 : 6)) *
                                tree_depth_);
    metadata_.reserve(history->prefixes, history->prefixes.size() + 1);
    reserve_branch(*history);
    return descriptor;
}

void HistoryDirectory::reserve_nodes(History& history, std::size_t count) {
    if (history.spare_nodes.size() >= count) return;
    const auto extra = count - history.spare_nodes.size();
    metadata_.reserve(history.spare_nodes, count);
    while (free_nodes_.size() < extra) {
        auto first = nodes_.size();
        for (std::uint32_t i = 0; i < nodes_.size(); i += 256)
            if (!nodes_.is_allocated(i)) {
                first = i;
                break;
            }
        if (first == nodes_.limit()) throw std::runtime_error("KV tree metadata exhausted");
        const auto end = std::min<std::uint64_t>(nodes_.limit(), std::uint64_t(first) + 256);
        metadata_.reserve(free_nodes_, std::max<std::uint64_t>(nodes_.size(), end));
        nodes_.ensure(first);
        for (auto i = first; i < end; ++i) {
            nodes_[i].owner = this;
            nodes_[i].index = i;
            free_nodes_.push_back(&nodes_[i]);
        }
    }
    for (std::size_t i = 0; i < extra; ++i) {
        auto* node = free_nodes_.back();
        free_nodes_.pop_back();
        node->reserved = true;
        history.spare_nodes.push_back(node);
    }
}

void HistoryDirectory::reserve_branch(History& history) {
    if (!history.branch_descriptor.remaining()) history.branch_descriptor = reserve_descriptors(1);
    const auto index = history.branch_descriptor.take(*this);
    history.branch_descriptor.restore(*this, index);
    reserve_nodes(histories_[index], (history.prefixes.size() + 4) * tree_depth_);
}

void HistoryDirectory::release_spares(History& history) noexcept {
    for (auto* node : history.spare_nodes) {
        node->reserved = false;
        free_nodes_.push_back(node);
    }
    history.spare_nodes.clear();
}

void HistoryDirectory::trim_prepared_metadata(History& history, std::uint32_t records) noexcept {
    const auto keep = (std::size_t(records) + history.prefixes.size() + 4) * tree_depth_;
    while (history.spare_nodes.size() > keep) {
        auto* node = history.spare_nodes.back();
        history.spare_nodes.pop_back();
        node->reserved = false;
        free_nodes_.push_back(node);
    }
    if (!records) history.branch_descriptor.reset();
}

HistoryDirectory::Directory HistoryDirectory::allocate_node(History& reservation,
                                                            const DirectoryNode* source) {
    if (reservation.spare_nodes.empty())
        throw std::logic_error("KV tree mutation exceeds its metadata reservation");
    auto* node = reservation.spare_nodes.back();
    reservation.spare_nodes.pop_back();
    if (source) {
        node->left  = source->left;
        node->right = source->right;
        node->pages = source->pages;
    }
    node->references = 1;
    node->reserved   = false;
    return Directory(node);
}

void HistoryDirectory::release_node(DirectoryNode* node) noexcept {
    if (!node->references) std::terminate();
    if (--node->references) return;
    node->left.reset();
    node->right.reset();
    node->pages = {};
    free_nodes_.push_back(node);
}

HistoryDirectory::Directory::~Directory() { reset(); }

HistoryDirectory::Directory::Directory(const Directory& other) noexcept : node_(other.node_) {
    if (node_) ++node_->references;
}

HistoryDirectory::Directory&
HistoryDirectory::Directory::operator=(const Directory& other) noexcept {
    if (this != &other) {
        Directory copy(other);
        *this = std::move(copy);
    }
    return *this;
}

HistoryDirectory::Directory::Directory(Directory&& other) noexcept
    : node_(std::exchange(other.node_, nullptr)) {}

HistoryDirectory::Directory& HistoryDirectory::Directory::operator=(Directory&& other) noexcept {
    if (this != &other) {
        reset();
        node_ = std::exchange(other.node_, nullptr);
    }
    return *this;
}

bool HistoryDirectory::Directory::unique() const noexcept {
    return node_ && node_->references == 1;
}

void HistoryDirectory::Directory::reset() noexcept {
    if (auto* node = std::exchange(node_, nullptr)) node->owner->release_node(node);
}

HistoryDirectory::~HistoryDirectory() = default;

bool HistoryDirectory::valid(HistoryId handle) const noexcept {
    return handle.pool_id == pages_->pool_id() && histories_.is_allocated(handle.index) &&
           histories_[handle.index].occupied &&
           histories_[handle.index].generation == handle.generation;
}

bool HistoryDirectory::valid(TableRow row) const noexcept {
    return row.pool_id == pages_->pool_id() && row.index < row_active_.size() &&
           row_active_[row.index] && row_generations_[row.index] == row.generation;
}

RowLease HistoryDirectory::acquire_row(std::int32_t index) {
    if (index < 0 || static_cast<std::size_t>(index) >= row_active_.size())
        throw std::out_of_range("KV execution row is outside directory capacity");
    if (row_active_[index]) throw std::logic_error("KV execution row is already occupied");
    row_active_[index] = true;
    return RowLease(
        *this, {pages_->pool_id(), static_cast<std::uint32_t>(index), row_generations_[index]});
}

void HistoryDirectory::release_row(TableRow row) noexcept {
    if (!valid(row)) std::terminate();
    trim_mapping(row, 0);
    row_active_[row.index] = false;
    auto& generation       = row_generations_[row.index];
    if (++generation == 0) generation = 1;
}

std::uint32_t HistoryDirectory::capacity() const noexcept { return histories_.limit(); }

std::uint32_t HistoryDirectory::occupied() const noexcept { return occupied_; }

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

HistoryId HistoryDirectory::allocate(std::span<const std::uint32_t> placement,
                                     MetadataReservation& reservation) {
    if (next_generation_ == UINT32_MAX)
        throw std::overflow_error("KV history generation exhausted");
    const auto index = reservation.take(*this);
    auto& history    = histories_[index];
    if (!history.reserved || history.occupied || history.placement.capacity() < placement.size())
        throw std::logic_error("KV history creation has no owned metadata descriptor");
    history.placement.assign(placement.begin(), placement.end());
    history.generation = next_generation_++;
    history.reserved   = false;
    history.occupied = history.owned = true;
    ++occupied_;
    return {pages_->pool_id(), index, history.generation};
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

ContextRecordBinding HistoryDirectory::physical_page(HistoryId h, std::uint32_t index,
                                                     std::uint32_t rank) const {
    return pages_->physical(logical_page(h, index), rank);
}

std::span<const std::uint32_t> HistoryDirectory::placement(HistoryId h) const {
    return require(h).placement;
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
    reserve_nodes(history, (std::size_t(history.prefixes.size()) + 3) * tree_depth_);
    auto entry = std::lower_bound(history.prefixes.begin(), history.prefixes.end(), frontier,
                                  [](const auto& value, auto key) { return value.first < key; });
    if (entry == history.prefixes.end() || entry->first != frontier) {
        const auto index = entry - history.prefixes.begin();
        metadata_.reserve(history.prefixes, history.prefixes.size() + 1);
        entry = history.prefixes.insert(history.prefixes.begin() + index, {frontier, 0});
    }
    ++entry->second;
    ++history.prefix_count;
    history.retained = std::max(old, frontier);
    update_ranges(history, std::max(history.owned ? history.frontier : 0, old),
                  std::max(history.owned ? history.frontier : 0, history.retained), old,
                  history.retained);
}

void HistoryDirectory::remove_prefix(HistoryId h, std::uint32_t frontier) noexcept {
    try {
        auto& history = require(h);
        auto it       = std::lower_bound(history.prefixes.begin(), history.prefixes.end(), frontier,
                                         [](const auto& value, auto key) { return value.first < key; });
        if (it == history.prefixes.end() || it->first != frontier || !it->second) std::terminate();
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

std::vector<HistoryId>
HistoryDirectory::released_histories(std::span<const HistoryPrefix* const> removed) const {
    std::vector<HistoryId> result;
    for (const auto* prefix : removed) {
        if (!prefix || prefix->store_ != this)
            throw std::invalid_argument("foreign history retirement proof");
        const auto handle   = prefix->history();
        const auto& history = require(handle);
        if (history.owned || std::find(result.begin(), result.end(), handle) != result.end())
            continue;
        const auto count = std::count_if(removed.begin(), removed.end(),
                                         [&](const auto* p) { return p->history() == handle; });
        if (count == history.prefix_count) result.push_back(handle);
    }
    return result;
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

    std::vector<PageId> candidates;
    for (const auto& [index, handle] : touched) {
        const auto& history = require(handle);
        if (history.owned) continue;
        const auto retained = retained_after(handle, removed);
        for (auto i = page_count(retained); i < history.page_count; ++i) {
            const auto page = membership(history, i);
            (void)pages_->descriptor_index(page);
            candidates.push_back(page);
        }
    }
    std::vector<PageId> result;
    std::sort(candidates.begin(), candidates.end(),
              [](const auto a, const auto b) { return a.index < b.index; });
    for (std::size_t begin = 0; begin < candidates.size();) {
        auto end = begin + 1;
        while (end < candidates.size() && candidates[end] == candidates[begin]) ++end;
        const auto page = candidates[begin];
        if (end - begin == pages_->history_references(page) &&
            pages_->active_history_references(page) == 0 && pages_->source_pins(page) == 0)
            result.push_back(page);
        begin = end;
    }
    return result;
}

std::vector<PageId> HistoryDirectory::released_pages(HistoryId owner) const {
    const auto& history = require(owner);
    if (!history.owned) throw std::invalid_argument("KV release query has no active owner");
    std::vector<PageId> result;
    for (auto i = page_count(history.retained); i < history.page_count; ++i) {
        const auto page = membership(history, i);
        if (pages_->history_references(page) == 1 && pages_->source_pins(page) == 0 &&
            pages_->active_history_references(page) ==
                static_cast<unsigned>(history.row.has_value()))
            result.push_back(page);
    }
    return result;
}

std::vector<HistoryId> HistoryDirectory::released_histories(HistoryId owner) const {
    const auto& history = require(owner);
    if (!history.owned) throw std::invalid_argument("KV release query has no active owner");
    return history.prefix_count ? std::vector<HistoryId>{} : std::vector{owner};
}

void HistoryDirectory::truncate_membership(History& history, std::uint32_t count) {
    if (count == history.page_count) return;
    auto directory = prefix_directory(history, history.directory, directory_capacity_, count);
    for (std::uint32_t i = count; i < history.page_count; ++i)
        if (!pages_->can_release_reference(membership(history, i)))
            throw std::logic_error("KV history still has an unowned reader");
    for (std::uint32_t i = count; i < history.page_count; ++i)
        if (!pages_->release_reference(membership(history, i)))
            throw std::logic_error("KV history still has an unowned reader");
    history.directory  = std::move(directory);
    history.page_count = count;
}

void HistoryDirectory::collect(HistoryId h) {
    auto& history = require(h);
    if (history.owned) return;
    truncate_membership(history, page_count(history.retained));
    history.frontier = history.retained;
    if (history.prefix_count) return;
    release_spares(history);
    auto placement   = std::move(history.placement);
    auto spare_nodes = std::move(history.spare_nodes);
    auto prefixes    = std::move(history.prefixes);
    placement.clear();
    prefixes.clear();
    history             = History{};
    history.placement   = std::move(placement);
    history.spare_nodes = std::move(spare_nodes);
    history.prefixes    = std::move(prefixes);
    --occupied_;
    free_[free_count_++] = h.index;
}

void HistoryDirectory::release_owner(HistoryId h) noexcept {
    try {
        auto& history = require(h);
        if (!history.owned) std::terminate();
        if (history.row) deactivate_retired(h);
        const auto before = std::max(history.frontier, history.retained);
        history.reservation.clear();
        trim_prepared_metadata(history, 0);
        history.owned = false;
        update_ranges(history, before, history.retained, history.retained, history.retained);
        collect(h);
    } catch (...) { std::terminate(); }
}

HistoryOwner HistoryDirectory::fork_history(const HistoryPrefix& source, std::uint32_t frontier,
                                            MetadataReservation& metadata) {
    if (!source.valid() || source.store_ != this || frontier > source.frontier())
        throw std::invalid_argument("KV binding source belongs to another history");
    auto& from = require(source.history());
    HistoryOwner result(*this, allocate(from.placement, metadata));
    auto& to     = require(result.handle());
    to.directory = prefix_directory(to, from.directory, directory_capacity_, page_count(frontier));
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
                                 std::uint32_t peak, std::span<const HistoryPrefix* const> retired,
                                 std::span<const std::uint32_t> placement) const {
    if (frontier > peak || page_count(peak) > page_capacity_ || (!source && frontier))
        throw std::invalid_argument("KV binding frontier is invalid");
    std::vector<std::uint32_t> ranks;
    if (source) {
        if (source->store_ != this || frontier > source->frontier())
            throw std::invalid_argument("KV prefix does not cover binding");
        ranks = require(source->history()).placement;
        if (!placement.empty() &&
            !std::equal(placement.begin(), placement.end(), ranks.begin(), ranks.end()))
            throw std::invalid_argument("KV prefix placement differs from the binding");
    } else if (placement.empty()) {
        for (const auto& local : pages_->layout().parts) ranks.push_back(local.rank);
    } else {
        ranks.assign(placement.begin(), placement.end());
    }
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        const auto rank = ranks[i];
        if (std::find(ranks.begin(), ranks.begin() + i, rank) != ranks.begin() + i ||
            std::none_of(pages_->layout().parts.begin(), pages_->layout().parts.end(),
                         [rank](const RecordPartSpec& part) { return part.rank == rank; }))
            throw std::invalid_argument("KV history rank placement is invalid");
    }
    HistoryBindingDemand result{.growth_pages = page_count(peak) - page_count(frontier)};
    result.logical_pages = result.growth_pages;
    for (auto rank : ranks)
        result.records.push_back(
            {pages_->layout_id(), ContextMemory::Device, result.growth_pages, {rank}});
    if (!source) {
        result.metadata.growth_pages = result.growth_pages;
        return result;
    }
    const auto& history  = require(source->history());
    result.reuse_history = !history.owned && retained_after(source->history(), retired) <= frontier;
    for (std::uint32_t i = 0; i < page_count(frontier); ++i) {
        const auto page   = membership(history, i);
        const auto needed = coverage(frontier, i);
        bool restoring    = false;
        for (std::size_t r = 0; r < ranks.size(); ++r) {
            if (pages_->device_covers(page, ranks[r], needed)) continue;
            if (!pages_->host_covers(page, ranks[r], needed))
                throw std::logic_error("KV prefix has no covering replica on a required rank");
            ++result.records[r].count;
            restoring = true;
        }
        result.restore_pages += restoring;
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
    for (auto& demand : result.records) demand.count += result.tail_copies;
    result.logical_pages += result.tail_copies;
    result.metadata = {result.reuse_history ? source->history() : HistoryId{},
                       result.growth_pages + result.tail_copies};
    return result;
}

HistoryBinding HistoryDirectory::prepare_binding(const HistoryPrefix* source,
                                                 std::uint32_t frontier, std::uint32_t peak,
                                                 std::int32_t execution_row,
                                                 RecordReservation reservation, bool consume_source,
                                                 std::span<const std::uint32_t> placement) {
    const std::array<const HistoryPrefix*, 1> consumed{source};
    const auto demand =
        binding_demand(source, frontier, peak,
                       consume_source && source ? std::span<const HistoryPrefix* const>(consumed)
                                                : std::span<const HistoryPrefix* const>{},
                       placement);
    auto history_metadata =
        demand.reuse_history ? MetadataReservation{} : reservation.take_metadata_reservation(*this);
    HistoryBinding binding;
    binding.store_      = this;
    binding.frontier_   = frontier;
    binding.write_tail_ = frontier % kPageTokens;
    if (source) binding.source_ = *source;
    if (source && !demand.reuse_history) {
        binding.history_ = fork_history(*source, frontier, history_metadata);
    } else {
        if (source) {
            auto& original   = require(source->history());
            original.owned   = true;
            binding.history_ = HistoryOwner(*this, source->history());
        } else {
            std::vector<std::uint32_t> ranks;
            if (placement.empty()) {
                for (const auto& local : pages_->layout().parts) ranks.push_back(local.rank);
                placement = ranks;
            }
            binding.history_ = HistoryOwner(*this, allocate(placement, history_metadata));
        }
    }
    binding.reservation_ = std::move(reservation);
    binding.reservation_.reserve_merge_capacity(
        (std::size_t(demand.growth_pages) + demand.tail_copies + 2) *
            require(binding.history()).placement.size(),
        demand.growth_pages + demand.tail_copies + 2);
    binding.row_.emplace(acquire_row(execution_row));
    auto& history = require(binding.history());
    pages_->prepare_returns(binding.reservation_, history.placement, page_count(peak));
    std::size_t restores = 0;
    for (const auto& record : demand.records)
        restores += record.count - demand.growth_pages - demand.tail_copies;
    binding.restores_.reserve(restores);
    for (std::uint32_t i = 0; i < page_count(frontier); ++i) {
        auto page = membership(history, i);
        for (auto rank : history.placement) {
            if (pages_->device_covers(page, rank, coverage(frontier, i))) continue;
            binding.restores_.push_back(
                pages_->reserve_device_replica(page, rank, binding.reservation_));
        }
    }
    if (demand.tail_copies) {
        binding.tail_source_      = membership(history, page_count(frontier) - 1);
        binding.tail_destination_ = pages_->materialize_transfer_destination(
            binding.reservation_, frontier % kPageTokens, history.placement);
    }
    return binding;
}

HistoryBindingDemand
HistoryDirectory::conversion_demand(const HistoryPrefix& source, std::uint32_t frontier,
                                    std::uint32_t peak,
                                    std::span<const std::uint32_t> placement) const {
    if (source.store_ != this || !source.valid() || frontier > source.frontier() || frontier > peak)
        throw std::invalid_argument("KV conversion source does not cover the requested frontier");
    auto demand          = binding_demand(nullptr, 0, peak, {}, placement);
    demand.restore_pages = page_count(frontier);
    demand.growth_pages  = page_count(peak) - demand.restore_pages;
    return demand;
}

HistoryBinding HistoryDirectory::prepare_conversion(const HistoryPrefix& source,
                                                    std::uint32_t frontier, std::uint32_t peak,
                                                    std::int32_t execution_row,
                                                    RecordReservation reservation,
                                                    std::span<const std::uint32_t> placement,
                                                    HistoryCopyLayout layout) {
    const auto demand = conversion_demand(source, frontier, peak, placement);
    if ((frontier >= kPageTokens && layout.full_page.empty()) ||
        (frontier % kPageTokens &&
         (layout.partial_columns != frontier % kPageTokens || layout.partial_page.empty())))
        throw std::invalid_argument("KV conversion layout does not describe its valid coverage");
    std::vector<std::uint32_t> target_ranks;
    for (const auto& part : demand.records) target_ranks.push_back(part.ranks.front());
    const auto source_ranks = this->placement(source.history());
    const auto validate     = [&](std::span<const RecordRangeCopy> fields) {
        for (const auto& field : fields) {
            if (!field.bytes ||
                std::find(source_ranks.begin(), source_ranks.end(), field.source_rank) ==
                    source_ranks.end() ||
                std::find(target_ranks.begin(), target_ranks.end(), field.destination_rank) ==
                    target_ranks.end())
                throw std::invalid_argument("KV conversion range names an invalid rank part");
            const auto& parts = pages_->layout().parts;
            const auto bytes  = [&](auto rank) {
                return std::find_if(parts.begin(), parts.end(),
                                         [&](const auto& part) { return part.rank == rank; })
                    ->record.bytes;
            };
            const auto source_bytes      = bytes(field.source_rank);
            const auto destination_bytes = bytes(field.destination_rank);
            if (field.source_offset > source_bytes ||
                field.bytes > source_bytes - field.source_offset ||
                field.destination_offset > destination_bytes ||
                field.bytes > destination_bytes - field.destination_offset)
                throw std::invalid_argument("KV conversion range exceeds its record layout");
        }
    };
    if (frontier >= kPageTokens) validate(layout.full_page);
    if (frontier % kPageTokens) validate(layout.partial_page);

    auto history_metadata = reservation.take_metadata_reservation(*this);
    HistoryBinding binding;
    binding.store_       = this;
    binding.frontier_    = frontier;
    binding.write_tail_  = frontier % kPageTokens;
    binding.source_      = source;
    binding.history_     = HistoryOwner(*this, allocate(target_ranks, history_metadata));
    binding.reservation_ = std::move(reservation);
    binding.reservation_.reserve_merge_capacity(
        (std::size_t(page_count(peak)) + 2) * target_ranks.size(), page_count(peak) + 2);
    binding.row_.emplace(acquire_row(execution_row));
    binding.conversion_layout_ = std::move(layout);
    binding.conversion_pages_.reserve(page_count(frontier));
    binding.conversion_readers_.reserve(std::size_t(page_count(frontier)) * source_ranks.size());
    pages_->prepare_returns(binding.reservation_, target_ranks, page_count(peak));
    auto& history = require(binding.history());
    for (std::uint32_t i = 0; i < page_count(frontier); ++i) {
        const auto page    = logical_page(source.history(), i);
        const auto columns = coverage(frontier, i);
        const auto& fields = columns == kPageTokens ? binding.conversion_layout_.full_page
                                                    : binding.conversion_layout_.partial_page;
        for (auto rank : source_ranks)
            if (std::any_of(fields.begin(), fields.end(),
                            [rank](const auto& field) { return field.source_rank == rank; }))
                binding.conversion_readers_.push_back(
                    pages_->read_device_prefix(page, rank, columns));
        auto& slot             = directory_slot(history, i);
        const auto destination = pages_->materialize_conversion_destination(
            binding.reservation_, page, columns, target_ranks);
        binding.conversion_pages_.push_back({page, destination, columns});
        slot = destination;
    }
    return binding;
}

void HistoryDirectory::abort(HistoryBinding& binding) noexcept {
    try {
        binding.row_.reset();
        binding.conversion_readers_.clear();
        for (const auto& page : binding.conversion_pages_)
            pages_->abort_transfer_destination(page.destination, binding.reservation_);
        binding.conversion_pages_.clear();
        if (binding.tail_pinned_) pages_->unpin_source(binding.tail_source_);
        if (binding.tail_destination_)
            pages_->abort_transfer_destination(*binding.tail_destination_, binding.reservation_);
        binding.restores_.clear();
        binding.store_ = nullptr;
        binding.reservation_.clear();
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
        auto& slot = directory_slot(history, index);
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
    history.reservation.clear();
    trim_prepared_metadata(history, 0);
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
    history.reservation.clear();
    trim_prepared_metadata(history, 0);
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
    transaction.commit_history(*this, owner.handle(), frontier, retain_through);
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
    trim_mapping(history.row->handle(), retained);
    for (auto i = retained; i < history.page_count; ++i) {
        const auto page = membership(history, i);
        if (!history.row || !pages_->can_reuse_after_retirement(page)) continue;
        pages_->reuse_after_retirement(page, history.reservation);
    }
    truncate_membership(history, retained);
}

void HistoryDirectory::prepare_growth(HistoryId handle, const RecordReservation& reservation) {
    auto& history = require(handle);
    if (!history.owned || !history.row)
        throw std::invalid_argument("KV growth requires an active history");
    std::uint32_t count = 0;
    for (auto rank : history.placement) {
        if (reservation.remaining(rank) > page_capacity_)
            throw std::invalid_argument("KV growth exceeds history capacity");
        count = std::max(count, reservation.remaining(rank) + history.reservation.remaining(rank));
    }
    history.reservation.reserve_merge(reservation);
    pages_->prepare_returns(history.reservation, history.placement, history.page_count + count);
    history.reservation.reserve_merge_capacity((std::size_t(count) + 2) * history.placement.size(),
                                               count + 2);
}

void HistoryDirectory::adopt_growth(HistoryId handle, RecordReservation reservation) {
    auto& history = require(handle);
    if (!history.owned || !history.row)
        throw std::invalid_argument("KV growth requires an active history");
    history.reservation.merge(std::move(reservation));
}

HistoryMetadataDemand HistoryDirectory::growth_metadata_demand(HistoryId handle,
                                                               std::uint32_t count) const {
    const auto& history = require(handle);
    std::uint32_t held  = 0;
    for (auto rank : history.placement) held = std::max(held, history.reservation.remaining(rank));
    return {handle, count + held};
}

RecordDemand HistoryDirectory::growth_demand(HistoryId handle, std::uint32_t count) const {
    const auto& history = require(handle);
    if (count > page_capacity_) throw std::invalid_argument("KV growth exceeds history capacity");
    return {pages_->layout_id(), ContextMemory::Device, count, history.placement};
}

void HistoryDirectory::release_growth(HistoryId handle) {
    auto& history = require(handle);
    history.reservation.clear();
    trim_prepared_metadata(history, 0);
}

bool HistoryDirectory::growth_contributes(HistoryId handle, const RecordAssessment& pressure,
                                          bool metadata_pressure) const {
    const auto& history = require(handle);
    if (history.reservation.contributes_to(pressure, metadata_pressure)) return true;
    if (!metadata_pressure) return false;
    const auto retained_nodes = (history.prefixes.size() + 4) * tree_depth_;
    return history.branch_descriptor.remaining() != 0 ||
           history.spare_nodes.size() > retained_nodes;
}

void HistoryDirectory::shrink_growth(HistoryId handle, std::uint32_t count) {
    auto& history = require(handle);
    for (auto rank : history.placement)
        if (history.reservation.remaining(rank) < count)
            throw std::logic_error("KV growth shrink cannot increase its reservation");
    history.reservation.shrink(count);
    trim_prepared_metadata(history, count);
}

void HistoryDirectory::settle_reservation(HistoryId handle, std::uint32_t retain_through) {
    const auto& history = require(handle);
    const auto count    = retain_through > history.frontier
                              ? write_pages(handle, history.frontier, retain_through)
                              : 0;
    shrink_growth(handle, count);
}

std::uint32_t HistoryDirectory::reserved_growth_pages(HistoryId handle) const {
    const auto& history = require(handle);
    auto result         = UINT32_MAX;
    for (auto rank : history.placement)
        result = std::min(result, history.reservation.remaining(rank));
    return result == UINT32_MAX ? 0 : result;
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
    if (copies > reserved_growth_pages(handle))
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
    for (auto i = old_count; i < target; ++i) (void)directory_slot(history, i);
    if (target > old_count) {
        materialize_.assign(target - old_count, {});
        pages_->materialize(history.reservation, materialize_, history.placement);
        for (auto page : materialize_) {
            directory_slot(history, history.page_count++) = page;
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
            auto& slot       = directory_slot(history, i);
            auto replacement = pages_->materialize_transfer_destination(
                history.reservation, std::max(1U, offset), history.placement);
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
        for (auto i = changed; i < target; ++i) publish_.push_back(membership(history, i));
        update_mapping(history.row->handle(), changed, publish_, transaction);
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
    HistoryOwner branch = fork_history(source, frontier, require(handle).branch_descriptor);
    auto& from          = require(handle);
    auto& to            = require(branch.handle());
    to.spare_nodes.swap(from.spare_nodes);
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
        for (std::uint32_t i = 0; i < to.page_count; ++i) publish_.push_back(membership(to, i));
        update_mapping(to.row->handle(), 0, publish_, transaction);
        trim_mapping(to.row->handle(), to.page_count);
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
    if (history.row) trim_mapping(history.row->handle(), page_count(frontier));
    for (auto i = page_count(frontier); i < history.page_count; ++i) {
        const auto page = membership(history, i);
        if (!history.row || !pages_->can_reuse_after_retirement(page)) continue;
        pages_->reuse_after_retirement(page, history.reservation);
    }
    truncate_membership(history, page_count(frontier));
    history.frontier = frontier;
}

void HistoryDirectory::update_mapping(TableRow row, std::uint32_t begin,
                                      std::span<const PageId> pages,
                                      ContextTransaction& transaction) {
    if (!valid(row) || begin > row_pages_[row.index].size() ||
        pages.size() > page_capacity_ - begin)
        throw std::invalid_argument("KV execution mapping exceeds its row");
    auto& mapped = row_pages_[row.index];
    mapped.resize(std::max(mapped.size(), std::size_t(begin) + pages.size()));
    for (std::uint32_t i = 0; i < pages.size(); ++i) {
        const auto column = begin + i;
        if (mapped[column] == pages[i]) continue;
        const PageTableLocation location{row.index, row.generation, column};
        pages_->bind_table(pages[i], location);
        if (mapped[column].valid()) pages_->unbind_table(mapped[column], location);
        mapped[column] = pages[i];
    }
    transaction.update_mapping(*pages_, row, begin, pages);
}

void HistoryDirectory::trim_mapping(TableRow row, std::uint32_t size) noexcept {
    auto& mapped = row_pages_[row.index];
    if (size > mapped.size()) std::terminate();
    for (std::uint32_t i = size; i < mapped.size(); ++i)
        pages_->unbind_table(mapped[i], {row.index, row.generation, i});
    mapped.resize(size);
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

PageId& HistoryDirectory::directory_slot(History& history, std::uint32_t index) {
    auto* node = &history.directory;
    auto width = directory_capacity_;
    for (;;) {
        if (!*node)
            *node = allocate_node(history);
        else if (!node->unique())
            *node = allocate_node(history, node->get());
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

HistoryDirectory::Directory HistoryDirectory::prefix_directory(History& reservation,
                                                               const Directory& node,
                                                               std::uint64_t width,
                                                               std::uint32_t count) {
    if (!node || !count) return {};
    if (count >= width) return node;
    auto result = allocate_node(reservation, node.get());
    if (width == kChunkPages)
        std::fill(result->pages.begin() + count, result->pages.end(), PageId{});
    else {
        width /= 2;
        if (count <= width) {
            result->left = prefix_directory(reservation, node->left, width, count);
            result->right.reset();
        } else
            result->right = prefix_directory(reservation, node->right, width, count - width);
    }
    return result;
}
} // namespace ninfer::models::context
