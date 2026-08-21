#include "core/kv_host_cache.h"

#include "core/device.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer {

namespace {
constexpr std::size_t kChunkBytes = 256ULL << 20;
}

KvHostCache::KvHostCache(const Config& config) : config_(config) {
    if (config_.budget_bytes == 0 || config_.text_page_bytes == 0 ||
        config_.anchor_state_bytes == 0) {
        throw std::invalid_argument("KV host cache requires nonzero budget and payload sizes");
    }
}

KvHostCache::~KvHostCache() {
    for (std::byte* chunk : chunks_) { cudaFreeHost(chunk); }
}

std::size_t KvHostCache::page_bytes(PageKind kind) const noexcept {
    return kind == PageKind::Text ? config_.text_page_bytes : config_.backend_page_bytes;
}

std::unordered_map<std::uint64_t, KvHostCache::PageEntry>&
KvHostCache::table(PageKind kind) noexcept {
    return kind == PageKind::Text ? text_pages_ : backend_pages_;
}

const std::unordered_map<std::uint64_t, KvHostCache::PageEntry>&
KvHostCache::table(PageKind kind) const noexcept {
    return kind == PageKind::Text ? text_pages_ : backend_pages_;
}

bool KvHostCache::has_page(PageKind kind, std::uint64_t key) const noexcept {
    return table(kind).contains(key);
}

bool KvHostCache::pin_page(PageKind kind, std::uint64_t key) {
    auto& entries    = table(kind);
    const auto entry = entries.find(key);
    if (entry == entries.end()) { return false; }
    ++entry->second.refcount;
    staged_pins_.emplace_back(kind, key);
    return true;
}

std::uint32_t KvHostCache::present_prefix(PageKind kind,
                                          std::span<const std::uint64_t> keys) const {
    const auto& entries = table(kind);
    std::uint32_t count = 0;
    for (const std::uint64_t key : keys) {
        if (!entries.contains(key)) { break; }
        ++count;
    }
    return count;
}

std::optional<KvHostCache::AnchorMeta> KvHostCache::find_anchor(std::uint64_t chain_key) const {
    const auto found = segments_.find(chain_key);
    if (found == segments_.end()) { return std::nullopt; }
    return found->second.meta;
}

bool KvHostCache::reserve_slot(std::size_t bytes, Slot* out) {
    std::vector<Slot>* free_list = nullptr;
    if (bytes == config_.text_page_bytes) {
        free_list = &free_slots_text_;
    } else if (bytes == config_.backend_page_bytes) {
        free_list = &free_slots_backend_;
    } else if (bytes == config_.anchor_state_bytes) {
        free_list = &free_slots_anchor_;
    } else {
        throw std::logic_error("KV host cache slot request does not match a payload class");
    }
    while (true) {
        if (!free_list->empty()) {
            *out = free_list->back();
            free_list->pop_back();
            used_bytes_ += out->bytes;
            return true;
        }
        if (current_chunk_left_ >= bytes) {
            std::byte* chunk = chunks_.back();
            out->host        = chunk + next_chunk_offset_;
            out->bytes       = bytes;
            next_chunk_offset_ += bytes;
            current_chunk_left_ -= bytes;
            used_bytes_ += bytes;
            return true;
        }
        const std::size_t chunk_bytes =
            std::min(kChunkBytes, config_.budget_bytes - carved_bytes_);
        if (chunk_bytes >= bytes) {
            void* host = nullptr;
            if (cudaHostAlloc(&host, chunk_bytes, cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                if (!evict_one_idle_segment()) { return false; }
                continue;
            }
            chunks_.push_back(static_cast<std::byte*>(host));
            carved_bytes_ += chunk_bytes;
            next_chunk_offset_  = 0;
            current_chunk_left_ = chunk_bytes;
            continue;
        }
        if (!evict_one_idle_segment()) { return false; }
    }
}

void KvHostCache::unpin_staged() noexcept {
    for (const auto& [kind, key] : staged_pins_) {
        auto& entries    = table(kind);
        const auto entry = entries.find(key);
        if (entry == entries.end()) { continue; }
        if (--entry->second.refcount == 0) {
            release_slot(entry->second.slot);
            stats_.stored_bytes -= entry->second.slot.bytes;
            --stats_.stored_pages;
            entries.erase(entry);
        }
    }
    staged_pins_.clear();
}

void KvHostCache::release_slot(const Slot& slot) noexcept {
    if (slot.host == nullptr) { return; }
    used_bytes_ -= slot.bytes;
    if (slot.bytes == config_.text_page_bytes) {
        free_slots_text_.push_back(slot);
    } else if (slot.bytes == config_.backend_page_bytes) {
        free_slots_backend_.push_back(slot);
    } else {
        free_slots_anchor_.push_back(slot);
    }
}

bool KvHostCache::evict_one_idle_segment() {
    if (lru_.empty()) { return false; }
    const std::uint64_t victim = lru_.back();
    const auto found           = segments_.find(victim);
    if (found == segments_.end()) {
        lru_.pop_back();
        return true;
    }
    drop_segment(victim, found->second);
    ++stats_.evicted_segments;
    return true;
}

void KvHostCache::drop_segment(std::uint64_t chain_key, Segment& segment) noexcept {
    const auto drop_keys = [&](PageKind kind, const std::vector<std::uint64_t>& keys) {
        auto& entries = table(kind);
        for (const std::uint64_t key : keys) {
            const auto entry = entries.find(key);
            if (entry == entries.end()) { continue; }
            if (--entry->second.refcount == 0) {
                release_slot(entry->second.slot);
                stats_.stored_bytes -= entry->second.slot.bytes;
                --stats_.stored_pages;
                entries.erase(entry);
            }
        }
    };
    drop_keys(PageKind::Text, segment.text_keys);
    drop_keys(PageKind::Backend, segment.backend_keys);
    release_slot(segment.anchor_slot);
    const auto index = anchors_by_page_.find(segment.index_page_key);
    if (index != anchors_by_page_.end()) {
        std::erase(index->second, chain_key);
        if (index->second.empty()) { anchors_by_page_.erase(index); }
    }
    lru_.erase(segment.lru_position);
    segments_.erase(chain_key);
    --stats_.stored_segments;
}

void KvHostCache::copy_slices(std::byte* host, std::span<const KvHostCopySlice> device_slices,
                              std::size_t expected_bytes, cudaMemcpyKind kind,
                              cudaStream_t stream) {
    std::size_t offset = 0;
    for (const KvHostCopySlice& slice : device_slices) {
        if (kind == cudaMemcpyDeviceToHost) {
            CUDA_CHECK(cudaMemcpyAsync(host + offset, slice.device_ptr, slice.bytes, kind, stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(slice.device_ptr, host + offset, slice.bytes, kind, stream));
        }
        offset += slice.bytes;
    }
    if (offset != expected_bytes) {
        throw std::logic_error("KV host cache copy slices do not cover the payload class");
    }
}

bool KvHostCache::stage_page(PageKind kind, std::uint64_t key,
                             std::span<const KvHostCopySlice> device_slices, cudaStream_t stream) {
    if (const auto existing = table(kind).find(key); existing != table(kind).end()) {
        // Pin shared pages so budget-driven eviction of their owning segment cannot free
        // them between the missing-page scan and seal_segment.
        ++existing->second.refcount;
        staged_pins_.emplace_back(kind, key);
        return true;
    }
    for (const StagedPage& staged : staged_pages_) {
        if (staged.kind == kind && staged.key == key) { return true; }
    }
    Slot slot{};
    if (!reserve_slot(page_bytes(kind), &slot)) { return false; }
    copy_slices(slot.host, device_slices, slot.bytes, cudaMemcpyDeviceToHost, stream);
    stats_.writeback_bytes += slot.bytes;
    staged_pages_.push_back(StagedPage{kind, key, slot});
    return true;
}

bool KvHostCache::stage_anchor(std::uint64_t chain_key, const AnchorMeta& meta,
                               std::span<const KvHostCopySlice> state_slices,
                               cudaStream_t stream) {
    if (segments_.contains(chain_key)) {
        touch_segment(chain_key);
        return false;
    }
    Slot slot{};
    if (!reserve_slot(config_.anchor_state_bytes, &slot)) { return false; }
    copy_slices(slot.host, state_slices, slot.bytes, cudaMemcpyDeviceToHost, stream);
    stats_.writeback_bytes += slot.bytes;
    Segment segment{};
    segment.meta        = meta;
    segment.anchor_slot = slot;
    staged_anchor_.emplace(chain_key, std::move(segment));
    return true;
}

std::span<const std::uint64_t>
KvHostCache::find_anchor_keys(std::uint64_t index_page_key) const {
    const auto found = anchors_by_page_.find(index_page_key);
    if (found == anchors_by_page_.end()) { return {}; }
    return found->second;
}

void KvHostCache::seal_segment(std::uint64_t chain_key, std::span<const std::uint64_t> text_keys,
                               std::span<const std::uint64_t> backend_keys,
                               std::uint64_t index_page_key) {
    if (!staged_anchor_ || staged_anchor_->first != chain_key) {
        throw std::logic_error("sealing a KV host cache segment without a staged anchor");
    }
    // Validate the whole reference set before mutating anything, so an invariant violation
    // leaves the store consistent and the transaction abortable.
    const auto covered = [&](PageKind kind, std::span<const std::uint64_t> keys) {
        for (const std::uint64_t key : keys) {
            if (table(kind).contains(key)) { continue; }
            const bool staged =
                std::any_of(staged_pages_.begin(), staged_pages_.end(),
                            [&](const StagedPage& page) {
                                return page.kind == kind && page.key == key;
                            });
            if (!staged) { return false; }
        }
        return true;
    };
    if (!covered(PageKind::Text, text_keys) || !covered(PageKind::Backend, backend_keys)) {
        throw std::logic_error("KV host cache segment references a missing page");
    }
    for (const StagedPage& staged : staged_pages_) {
        auto& entries = table(staged.kind);
        entries.emplace(staged.key, PageEntry{staged.slot, 0});
        stats_.stored_bytes += staged.slot.bytes;
        ++stats_.stored_pages;
    }
    staged_pages_.clear();
    const auto bump = [&](PageKind kind, std::span<const std::uint64_t> keys) {
        auto& entries = table(kind);
        for (const std::uint64_t key : keys) {
            const auto entry = entries.find(key);
            if (entry == entries.end()) {
                throw std::logic_error("KV host cache segment references a missing page");
            }
            ++entry->second.refcount;
        }
    };
    bump(PageKind::Text, text_keys);
    bump(PageKind::Backend, backend_keys);
    unpin_staged();
    Segment segment        = std::move(staged_anchor_->second);
    segment.text_keys      = {text_keys.begin(), text_keys.end()};
    segment.backend_keys   = {backend_keys.begin(), backend_keys.end()};
    segment.index_page_key = index_page_key;
    anchors_by_page_[index_page_key].push_back(chain_key);
    lru_.push_front(chain_key);
    segment.lru_position = lru_.begin();
    segments_.emplace(chain_key, std::move(segment));
    staged_anchor_.reset();
    ++stats_.stored_segments;
}

void KvHostCache::abort_segment() {
    unpin_staged();
    for (const StagedPage& staged : staged_pages_) { release_slot(staged.slot); }
    staged_pages_.clear();
    if (staged_anchor_) {
        release_slot(staged_anchor_->second.anchor_slot);
        staged_anchor_.reset();
    }
}

void KvHostCache::restore_page(PageKind kind, std::uint64_t key,
                               std::span<const KvHostCopySlice> device_slices,
                               cudaStream_t stream) {
    const auto entry = table(kind).find(key);
    if (entry == table(kind).end()) {
        throw std::logic_error("restoring a KV host cache page that is not stored");
    }
    copy_slices(entry->second.slot.host, device_slices, entry->second.slot.bytes,
                cudaMemcpyHostToDevice, stream);
    stats_.restored_bytes += entry->second.slot.bytes;
}

void KvHostCache::restore_anchor_state(std::uint64_t chain_key,
                                       std::span<const KvHostCopySlice> state_slices,
                                       cudaStream_t stream) {
    const auto found = segments_.find(chain_key);
    if (found == segments_.end()) {
        throw std::logic_error("restoring a KV host cache anchor that is not stored");
    }
    copy_slices(found->second.anchor_slot.host, state_slices, found->second.anchor_slot.bytes,
                cudaMemcpyHostToDevice, stream);
    stats_.restored_bytes += found->second.anchor_slot.bytes;
}

void KvHostCache::note_hit(std::uint32_t tokens) {
    ++stats_.hit_requests;
    stats_.hit_tokens += tokens;
}

void KvHostCache::touch_segment(std::uint64_t chain_key) {
    const auto found = segments_.find(chain_key);
    if (found == segments_.end()) { return; }
    lru_.erase(found->second.lru_position);
    lru_.push_front(chain_key);
    found->second.lru_position = lru_.begin();
}

} // namespace ninfer
