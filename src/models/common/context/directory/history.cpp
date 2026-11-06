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

std::uint32_t chunk_count(std::uint64_t pages) {
    return static_cast<std::uint32_t>((pages + 31) / 32);
}

std::uint32_t chunk_limit(std::uint32_t pages, std::uint32_t histories) {
    // Every occupied or reserved chunk contains a page, except the shared boundary
    // and the prepared append boundary of each history.
    const auto limit = std::uint64_t(pages) + 2ULL * histories;
    if (limit > UINT32_MAX) throw std::overflow_error("KV chunk directory exceeds uint32");
    return static_cast<std::uint32_t>(limit);
}

std::size_t append_chunks(std::uint32_t prefix, std::uint32_t growth) {
    return growth ? chunk_count(std::uint64_t(growth) + (prefix ? 31 : 0)) : 0;
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
    if (valid()) {
        store_->add_prefix(history_, frontier_);
        set_use(other.use_, other.memory_);
    }
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
      frontier_(std::exchange(other.frontier_, 0)), use_(other.use_), memory_(other.memory_) {}

HistoryPrefix& HistoryPrefix::operator=(HistoryPrefix&& other) noexcept {
    if (this != &other) {
        reset();
        store_    = std::exchange(other.store_, nullptr);
        history_  = std::exchange(other.history_, {});
        frontier_ = std::exchange(other.frontier_, 0);
        use_      = other.use_;
        memory_   = other.memory_;
    }
    return *this;
}

void HistoryPrefix::set_use(ContextUse use, ContextMemory memory) {
    if (use_ == use && memory_ == memory) return;
    if (!valid()) throw std::logic_error("KV prefix has no owner");
    if (use == ContextUse::Request)
        store_->update_request_prefix(history_, frontier_, memory, true);
    if (use_ == ContextUse::Request)
        store_->update_request_prefix(history_, frontier_, memory_, false);
    use_    = use;
    memory_ = memory;
}

void HistoryPrefix::reset() noexcept {
    if (valid()) {
        if (use_ == ContextUse::Request)
            store_->update_request_prefix(history_, frontier_, memory_, false);
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
      chunks_(chunk_limit(pages.capacity(), history_capacity), metadata),
      histories_(history_capacity, metadata), free_count_(0), row_pages_(row_count) {
    if (!history_capacity || !page_capacity || !row_count || row_count > INT32_MAX)
        throw std::invalid_argument("KV history capacity is inconsistent");
    metadata_.acquire(row_generations_.capacity() * sizeof(std::uint32_t) +
                      (row_active_.capacity() + 7) / 8 +
                      row_pages_.capacity() * sizeof(std::vector<PageId>));
    metadata_.reserve(publish_, page_capacity);
    metadata_.reserve(materialize_, page_capacity);
    for (auto& row : row_pages_) metadata_.reserve(row, page_capacity);
    pages.configure_histories(row_count, page_capacity, history_capacity);
}

std::size_t HistoryDirectory::metadata_bytes(std::uint32_t histories, std::uint32_t rows,
                                             std::uint32_t page_capacity, std::uint32_t ranks) {
    const auto directory = chunk_count(page_capacity) * sizeof(ChunkRef);
    const auto boundary  = 2 * (sizeof(Chunk) + sizeof(Chunk*));
    // Admission materializes descriptors/chunks in slabs; retain one rounding slab independent
    // of the number of cache histories selected by the capacity forecast.
    const auto slabs =
        255 * (sizeof(History) + sizeof(std::uint32_t) + sizeof(Chunk) + sizeof(Chunk*));
    return sizeof(HistoryDirectory) + slabs +
           PageDirectory::coverage_metadata_bytes(histories + 255) +
           PageDirectory::table_metadata_bytes(rows, page_capacity) +
           std::size_t(histories) *
               (sizeof(History) + (std::size_t(ranks) + 1) * sizeof(std::uint32_t) + directory +
                boundary) +
           std::size_t(rows) * (sizeof(std::uint32_t) + sizeof(std::vector<PageId>) +
                                std::size_t(page_capacity) * sizeof(PageId)) +
           ((std::size_t(rows) + 63) / 64) * 8 + std::size_t(page_capacity) * sizeof(PageId) * 2;
}

std::uint32_t HistoryDirectory::capacity_for_metadata(std::size_t bytes) noexcept {
    return static_cast<std::uint32_t>(
        std::min<std::size_t>(UINT32_MAX / 4, bytes / sizeof(History)));
}

std::size_t HistoryDirectory::page_metadata_bytes() noexcept {
    return (sizeof(Chunk) + sizeof(Chunk*) + kChunkPages - 1) / kChunkPages;
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
    pages_->reserve_coverage(histories_.size());
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
    std::map<std::uint32_t, std::uint32_t> dropped;
    const auto include = [&](std::uint32_t index) {
        result.histories.push_back(index);
        const auto& history = histories_[index];
        for (const auto* chunk : history.spare_chunks) result.chunks.push_back(chunk->index);
        for (const auto& chunk : history.directory)
            if (++dropped[chunk->index] == chunk->references) result.chunks.push_back(chunk->index);
    };
    for (const auto handle : released) {
        const auto& history = require(handle);
        if (history.owned)
            throw std::logic_error("active KV writer cannot be retired by a cache quote");
        if (std::find(result.histories.begin(), result.histories.end(), handle.index) !=
            result.histories.end())
            continue;
        if (history.branch_descriptor.remaining()) include(history.branch_descriptor.last());
        include(handle.index);
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
            segment += sizeof(History) + history.placement.capacity() * sizeof(std::uint32_t) +
                       history.spare_chunks.capacity() * sizeof(Chunk*) +
                       history.directory.capacity() * sizeof(ChunkRef) +
                       history.prefixes.capacity() * sizeof(PrefixReferences);
        }
        bytes += segment;
    }
    for (std::uint32_t first = 0; first < chunks_.size(); first += 256) {
        if (!chunks_.is_allocated(first)) continue;
        const auto end = first + std::min(256U, chunks_.size() - first);
        bool empty     = true;
        for (auto i = first; i < end; ++i)
            empty &=
                (!chunks_[i].references && !chunks_[i].reserved) ||
                std::find(retired.chunks.begin(), retired.chunks.end(), i) != retired.chunks.end();
        if (empty) bytes += std::size_t(end - first) * sizeof(Chunk);
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
                              history.spare_chunks.capacity() * sizeof(Chunk*) +
                              history.directory.capacity() * sizeof(ChunkRef) +
                              history.prefixes.capacity() * sizeof(PrefixReferences));
        }
        histories_.release_segment(first);
    }
    for (std::uint32_t first = 0; first < chunks_.size(); first += 256) {
        if (!chunks_.is_allocated(first)) continue;
        const auto end = std::min<std::uint64_t>(chunks_.size(), std::uint64_t(first) + 256);
        bool empty     = true;
        for (auto i = first; i < end; ++i) empty &= !chunks_[i].references && !chunks_[i].reserved;
        if (!empty) continue;
        free_chunks_.erase(std::remove_if(free_chunks_.begin(), free_chunks_.end(),
                                          [first, end](const Chunk* chunk) {
                                              return chunk->index >= first && chunk->index < end;
                                          }),
                           free_chunks_.end());
        chunks_.release_segment(first);
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
    std::size_t bytes = 0, extra_chunks = 0, reallocation = 0;
    const auto vector_bytes = [&](std::size_t capacity, std::size_t required, std::size_t element) {
        if (required > capacity) reallocation = std::max(reallocation, capacity * element);
        return required > capacity ? (required - capacity) * element : 0;
    };
    const auto descriptor = [&]() -> std::optional<std::uint32_t> {
        std::uint32_t index;
        if (retired_count)
            index = retired.histories[--retired_count];
        else if (free_count)
            index = free_[--free_count];
        else {
            if (new_begin == new_end) {
                if (materialized == histories_.limit()) return std::nullopt;
                new_begin = materialized;
                new_end =
                    std::min<std::uint64_t>(histories_.limit(), std::uint64_t(materialized) + 256);
                materialized = new_end;
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
    const auto prepare = [&](std::uint32_t index, std::size_t directory, std::size_t chunks,
                             std::size_t prefixes) {
        const auto* history = histories_.is_allocated(index) ? &histories_[index] : nullptr;
        const bool reused = std::find(retired.histories.begin(), retired.histories.end(), index) !=
                            retired.histories.end();
        const auto held = history && !reused ? history->spare_chunks.size() : 0;
        extra_chunks += chunks > held ? chunks - held : 0;
        bytes +=
            vector_bytes(history ? history->directory.capacity() : 0, directory, sizeof(ChunkRef));
        bytes +=
            vector_bytes(history ? history->spare_chunks.capacity() : 0, chunks, sizeof(Chunk*));
        bytes += vector_bytes(history ? history->prefixes.capacity() : 0, prefixes,
                              sizeof(PrefixReferences));
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
        const auto* history  = demand.history.valid() ? &histories_[index] : nullptr;
        const auto prefix    = history ? history->page_count : demand.prefix_pages;
        const auto directory = chunk_count(
            std::min<std::uint64_t>(page_capacity_, std::uint64_t(prefix) + demand.growth_pages));
        prepare(index, directory, append_chunks(prefix, demand.growth_pages),
                history ? history->prefixes.size() + 1 : 1);
        if (history && history->retained) {
            const auto branch = history->branch_descriptor.remaining()
                                    ? std::optional{history->branch_descriptor.last()}
                                    : descriptor();
            if (!branch) return std::nullopt;
            prepare(*branch, std::max<std::size_t>(directory, history->directory.capacity()), 0, 0);
        }
    }
    bytes += pages_->coverage_metadata_requirement(materialized);
    bytes += vector_bytes(free_.capacity(), materialized, sizeof(std::uint32_t));
    const auto new_payload = std::size_t(materialized - histories_.size()) * sizeof(History);
    bytes += histories_.growth_bytes(materialized) - new_payload;
    auto available  = free_chunks_.size() + retired.chunks.size();
    auto chunk_size = chunks_.size();
    for (std::uint32_t first = 0; available < extra_chunks && first < chunks_.limit();
         first += 256) {
        if (chunks_.is_allocated(first)) continue;
        const auto count = std::min(256U, chunks_.limit() - first);
        available += count;
        bytes += std::size_t(count) * sizeof(Chunk);
        chunk_size = std::max(chunk_size, first + count);
    }
    if (available < extra_chunks) return std::nullopt;
    bytes += vector_bytes(free_chunks_.capacity(), chunk_size, sizeof(Chunk*));
    bytes +=
        chunks_.growth_bytes(chunk_size) - std::size_t(chunk_size - chunks_.size()) * sizeof(Chunk);
    return bytes + reallocation;
}

MetadataReservation HistoryDirectory::reserve_control(const HistoryMetadataDemand& demand) {
    MetadataReservation descriptor;
    History* history;
    if (demand.history.valid())
        history = &require(demand.history);
    else {
        descriptor = reserve_descriptors(1);
        history    = &histories_[descriptor.last()];
    }
    const auto prefix = demand.history.valid() ? history->page_count : demand.prefix_pages;
    metadata_.reserve(history->directory,
                      chunk_count(std::min<std::uint64_t>(
                          page_capacity_, std::uint64_t(prefix) + demand.growth_pages)));
    reserve_chunks(*history, append_chunks(prefix, demand.growth_pages));
    metadata_.reserve(history->prefixes, history->prefixes.size() + 1);
    if (history->retained) reserve_branch(*history);
    return descriptor;
}

void HistoryDirectory::reserve_chunks(History& history, std::size_t count) {
    if (history.spare_chunks.size() >= count) return;
    const auto extra = count - history.spare_chunks.size();
    metadata_.reserve(history.spare_chunks, count);
    while (free_chunks_.size() < extra) {
        auto first = chunks_.size();
        for (std::uint32_t i = 0; i < chunks_.size(); i += 256)
            if (!chunks_.is_allocated(i)) {
                first = i;
                break;
            }
        if (first == chunks_.limit()) throw std::runtime_error("KV chunk metadata exhausted");
        const auto end = std::min<std::uint64_t>(chunks_.limit(), std::uint64_t(first) + 256);
        metadata_.reserve(free_chunks_, std::max<std::uint64_t>(chunks_.size(), end));
        chunks_.ensure(first);
        for (auto i = first; i < end; ++i) {
            chunks_[i].owner = this;
            chunks_[i].index = i;
            free_chunks_.push_back(&chunks_[i]);
        }
    }
    for (std::size_t i = 0; i < extra; ++i) {
        auto* chunk = free_chunks_.back();
        free_chunks_.pop_back();
        chunk->reserved = true;
        history.spare_chunks.push_back(chunk);
    }
}

void HistoryDirectory::reserve_branch(History& history) {
    if (!history.branch_descriptor.remaining()) history.branch_descriptor = reserve_descriptors(1);
    auto& branch = histories_[history.branch_descriptor.last()];
    metadata_.reserve(branch.directory, history.directory.capacity());
}

void HistoryDirectory::release_spares(History& history) noexcept {
    for (auto* chunk : history.spare_chunks) {
        chunk->reserved = false;
        free_chunks_.push_back(chunk);
    }
    history.spare_chunks.clear();
}

void HistoryDirectory::trim_prepared_metadata(History& history, std::uint32_t records) noexcept {
    const auto keep = append_chunks(history.page_count, records);
    while (history.spare_chunks.size() > keep) {
        auto* chunk = history.spare_chunks.back();
        history.spare_chunks.pop_back();
        chunk->reserved = false;
        free_chunks_.push_back(chunk);
    }
    if (!records || !history.retained) history.branch_descriptor.reset();
}

HistoryDirectory::ChunkRef HistoryDirectory::allocate_chunk(History& reservation,
                                                            const Chunk* source) {
    if (reservation.spare_chunks.empty())
        throw std::logic_error("KV directory mutation exceeds its chunk reservation");
    auto* chunk = reservation.spare_chunks.back();
    reservation.spare_chunks.pop_back();
    if (source) chunk->pages = source->pages;
    chunk->references = 1;
    chunk->reserved   = false;
    return ChunkRef(chunk);
}

void HistoryDirectory::release_chunk(Chunk* chunk) noexcept {
    if (!chunk->references) std::terminate();
    if (--chunk->references) return;
    chunk->pages = {};
    free_chunks_.push_back(chunk);
}

HistoryDirectory::ChunkRef::~ChunkRef() { reset(); }

HistoryDirectory::ChunkRef::ChunkRef(const ChunkRef& other) noexcept : chunk_(other.chunk_) {
    if (chunk_) ++chunk_->references;
}

HistoryDirectory::ChunkRef& HistoryDirectory::ChunkRef::operator=(const ChunkRef& other) noexcept {
    if (this != &other) {
        ChunkRef copy(other);
        *this = std::move(copy);
    }
    return *this;
}

HistoryDirectory::ChunkRef::ChunkRef(ChunkRef&& other) noexcept
    : chunk_(std::exchange(other.chunk_, nullptr)) {}

HistoryDirectory::ChunkRef& HistoryDirectory::ChunkRef::operator=(ChunkRef&& other) noexcept {
    if (this != &other) {
        reset();
        chunk_ = std::exchange(other.chunk_, nullptr);
    }
    return *this;
}

bool HistoryDirectory::ChunkRef::unique() const noexcept {
    return chunk_ && chunk_->references == 1;
}

void HistoryDirectory::ChunkRef::reset() noexcept {
    if (auto* chunk = std::exchange(chunk_, nullptr)) chunk->owner->release_chunk(chunk);
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

void HistoryDirectory::update_request_prefix(HistoryId h, std::uint32_t frontier,
                                             ContextMemory memory, bool add) {
    auto& history     = require(h);
    const auto domain = static_cast<std::size_t>(memory);
    auto entry        = std::lower_bound(history.prefixes.begin(), history.prefixes.end(), frontier,
                                         [](const auto& value, auto key) { return value.frontier < key; });
    if (entry == history.prefixes.end() || entry->frontier != frontier)
        throw std::logic_error("KV request prefix is not retained");
    auto& count       = entry->requests[domain];
    const auto before = history.request_frontiers[domain];
    auto after        = before;
    if (add) {
        if (count == UINT32_MAX) throw std::overflow_error("KV request prefix count overflow");
        ++count;
        after = std::max(before, frontier);
    } else {
        if (!count) throw std::logic_error("KV request prefix is not held");
        if (--count == 0 && before == frontier) {
            after = 0;
            for (auto it = history.prefixes.rbegin(); it != history.prefixes.rend(); ++it)
                if (it->requests[domain]) {
                    after = it->frontier;
                    break;
                }
        }
    }
    const auto old_pages = page_count(before), new_pages = page_count(after);
    for (auto i = std::min(old_pages, new_pages); i < std::max(old_pages, new_pages); ++i) {
        const auto page = membership(history, i);
        if (after > before)
            pages_->retain_request_reference(page, memory);
        else
            pages_->release_request_reference(page, memory);
    }
    history.request_frontiers[domain] = after;
}

void HistoryDirectory::add_prefix(HistoryId h, std::uint32_t frontier) {
    auto& history = require(h);
    if (frontier > history.frontier)
        throw std::invalid_argument("KV retained prefix is outside committed history");
    const auto old = history.retained;
    auto entry     = std::lower_bound(history.prefixes.begin(), history.prefixes.end(), frontier,
                                      [](const auto& value, auto key) { return value.frontier < key; });
    if (entry == history.prefixes.end() || entry->frontier != frontier) {
        const auto index = entry - history.prefixes.begin();
        metadata_.reserve(history.prefixes, history.prefixes.size() + 1);
        entry = history.prefixes.insert(history.prefixes.begin() + index, {frontier, 0});
    }
    ++entry->count;
    ++history.prefix_count;
    history.retained = std::max(old, frontier);
    update_ranges(history, std::max(history.owned ? history.frontier : 0, old),
                  std::max(history.owned ? history.frontier : 0, history.retained), old,
                  history.retained);
}

void HistoryDirectory::remove_prefix(HistoryId h, std::uint32_t frontier) noexcept {
    try {
        auto& history = require(h);
        auto it =
            std::lower_bound(history.prefixes.begin(), history.prefixes.end(), frontier,
                             [](const auto& value, auto key) { return value.frontier < key; });
        if (it == history.prefixes.end() || it->frontier != frontier || !it->count)
            std::terminate();
        const auto old = history.retained;
        if (--it->count == 0) history.prefixes.erase(it);
        --history.prefix_count;
        history.retained = history.prefixes.empty() ? 0 : history.prefixes.rbegin()->frontier;
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
            return prefix && prefix->history() == h && prefix->frontier() == it->frontier;
        });
        if (erased > it->count) throw std::logic_error("KV release counts a prefix more than once");
        if (erased < it->count) return it->frontier;
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
    for (std::uint32_t i = count; i < history.page_count; ++i)
        if (!pages_->can_release_reference(membership(history, i)))
            throw std::logic_error("KV history still has an unowned reader");
    for (std::uint32_t i = count; i < history.page_count; ++i)
        if (!pages_->release_reference(membership(history, i)))
            throw std::logic_error("KV history still has an unowned reader");
    history.directory.resize(chunk_count(count));
    history.page_count = count;
}

void HistoryDirectory::collect(HistoryId h) {
    auto& history = require(h);
    if (history.owned) return;
    truncate_membership(history, page_count(history.retained));
    history.frontier = history.retained;
    if (history.prefix_count) return;
    release_spares(history);
    auto directory = std::move(history.directory);
    directory.clear();
    auto placement    = std::move(history.placement);
    auto spare_chunks = std::move(history.spare_chunks);
    auto prefixes     = std::move(history.prefixes);
    placement.clear();
    prefixes.clear();
    history              = History{};
    history.directory    = std::move(directory);
    history.placement    = std::move(placement);
    history.spare_chunks = std::move(spare_chunks);
    history.prefixes     = std::move(prefixes);
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
    auto& to = require(result.handle());
    copy_prefix(to, from, page_count(frontier));
    to.frontier = frontier;
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
                       result.growth_pages + result.tail_copies, page_count(frontier)};
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
    return history.branch_descriptor.remaining() != 0 || !history.spare_chunks.empty();
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
    to.spare_chunks.swap(from.spare_chunks);
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
        pages_->replace_table(mapped[column], pages[i], location);
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
    const auto& page = history.directory[index / kChunkPages]->pages[index % kChunkPages];
    return {pages_->pool_id(), page.index, page.generation};
}

HistoryDirectory::LocalPage& HistoryDirectory::directory_slot(History& history,
                                                              std::uint32_t index) {
    const auto count = chunk_count(std::uint64_t(index) + 1);
    if (index >= page_capacity_ || count > history.directory.capacity())
        throw std::logic_error("KV directory index exceeds its prepared capacity");
    history.directory.resize(std::max<std::size_t>(history.directory.size(), count));
    auto& chunk = history.directory[index / kChunkPages];
    if (!chunk)
        chunk = allocate_chunk(history);
    else if (!chunk.unique())
        chunk = allocate_chunk(history, chunk.get());
    return chunk->pages[index % kChunkPages];
}

void HistoryDirectory::copy_prefix(History& destination, const History& source,
                                   std::uint32_t pages) {
    const auto count = chunk_count(pages);
    if (!destination.directory.empty() || count > destination.directory.capacity() ||
        pages > source.page_count)
        throw std::logic_error("KV prefix has no prepared chunk index");
    for (std::uint32_t i = 0; i < count; ++i) destination.directory.push_back(source.directory[i]);
}
} // namespace ninfer::models::context
