#include "targets/qwen3_6/impl/runtime/prefix_seed_store.h"

#include "core/device.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {

namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime  = 1099511628211ULL;

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("PrefixSeedStore: ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

} // namespace

std::uint64_t prefix_seed_hash(std::span<const TokenId> tokens) {
    std::uint64_t hash = kFnvOffset;
    for (const TokenId token : tokens) {
        std::uint64_t value = static_cast<std::uint32_t>(token);
        for (int i = 0; i < 4; ++i) {
            hash ^= (value >> (8 * i)) & 0xFFULL;
            hash *= kFnvPrime;
        }
    }
    return hash;
}

PrefixSeedStore::~PrefixSeedStore() noexcept { release(); }

void PrefixSeedStore::release() noexcept {
    entries_.clear();
    arena_used_             = 0;
    arena_bytes_            = 0;
    state_layers_           = 0;
    conv_slot_bytes_        = 0;
    recurrent_slot_bytes_   = 0;
    hidden_bytes_           = 0;
    if (arena_ != nullptr) {
        (void)cudaFree(arena_);
        arena_ = nullptr;
    }
}

void PrefixSeedStore::initialize(std::size_t budget_bytes,
                                 const LinearAttentionStatePool& state_pool,
                                 const PagedKVPool& text_pool, const PagedKVPool* backend_pool,
                                 std::size_t hidden_bytes) {
    (void)text_pool;
    (void)backend_pool;
    if (arena_ != nullptr) { throw std::logic_error("PrefixSeedStore is already initialized"); }
    if (budget_bytes == 0) { return; }
    state_layers_ = state_pool.layer_count();
    if (state_layers_ == 0) { throw std::invalid_argument("prefix seeds require GDN state"); }
    conv_slot_bytes_      = state_pool.conv_slot(0, 0).bytes();
    recurrent_slot_bytes_ = state_pool.recurrent_slot(0, 0).bytes();
    if (hidden_bytes == 0) {
        throw std::invalid_argument("prefix seeds require a hidden image size");
    }
    hidden_bytes_ = hidden_bytes;
    const std::size_t minimum = state_image_bytes() + hidden_bytes_ + (1ULL << 20);
    if (budget_bytes < minimum) {
        throw std::invalid_argument("prefix cache budget is below one seed entry");
    }
    check_cuda(cudaMalloc(&arena_, budget_bytes), "arena allocation");
    arena_bytes_ = budget_bytes;
    arena_used_  = 0;
    std::fprintf(stderr, "ninfer: prefix-seed store enabled (%zu MiB)\n", budget_bytes >> 20);
}

std::size_t PrefixSeedStore::state_image_bytes() const noexcept {
    return static_cast<std::size_t>(state_layers_) * (conv_slot_bytes_ + recurrent_slot_bytes_);
}

std::size_t PrefixSeedStore::kv_page_bytes(const PagedKVPool& pool) const {
    std::size_t bytes = 0;
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        bytes += pool.plane(plane).bytes() / pool.page_group_count();
    }
    return bytes;
}

std::int64_t PrefixSeedStore::find(const PreparedPromptData& prompt) const {
    std::int64_t best      = -1;
    std::uint32_t best_len = 0;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        const Entry& entry = entries_[i];
        if (entry.frontier <= best_len || entry.frontier >= prompt.token_ids.size()) { continue; }
        if (!entry_matches(static_cast<std::int64_t>(i), prompt)) { continue; }
        best     = static_cast<std::int64_t>(i);
        best_len = entry.frontier;
    }
    return best;
}

bool PrefixSeedStore::contains(const PreparedPromptData& prompt, std::uint32_t frontier) const {
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].frontier == frontier &&
            entry_matches(static_cast<std::int64_t>(i), prompt)) {
            return true;
        }
    }
    return false;
}

std::uint32_t PrefixSeedStore::entry_frontier(std::int64_t entry) const {
    return entries_.at(static_cast<std::size_t>(entry)).frontier;
}

std::int32_t PrefixSeedStore::entry_rope_delta(std::int64_t entry) const {
    return entries_.at(static_cast<std::size_t>(entry)).rope_delta;
}

std::span<const TokenId> PrefixSeedStore::tokens(std::int64_t entry) const {
    const Entry& e = entries_.at(static_cast<std::size_t>(entry));
    return std::span<const TokenId>(e.ledger.data(), e.ledger.size());
}

bool PrefixSeedStore::entry_matches(std::int64_t index, const PreparedPromptData& prompt) const {
    const Entry& entry = entries_.at(static_cast<std::size_t>(index));
    if (entry.frontier > prompt.token_ids.size()) { return false; }
    const std::span<const TokenId> head(prompt.token_ids.data(), entry.frontier);
    if (prefix_seed_hash(head) != entry.hash) { return false; }
    if (!std::equal(entry.ledger.begin(), entry.ledger.end(), prompt.token_ids.begin())) {
        return false;
    }
    return prefix_matches(prompt, entry.ledger, entry.identity, entry.frontier);
}

void PrefixSeedStore::copy_state_image(const LinearAttentionStatePool& state_pool,
                                       std::int32_t slot, std::byte* arena_base,
                                       std::size_t offset, bool to_arena,
                                       cudaStream_t stream) const {
    std::size_t cursor = offset;
    for (std::uint32_t layer = 0; layer < state_layers_; ++layer) {
        const Tensor conv      = state_pool.conv_slot(layer, slot);
        const Tensor recurrent = state_pool.recurrent_slot(layer, slot);
        std::byte* arena_conv      = arena_base + cursor;
        std::byte* arena_recurrent = arena_base + cursor + conv_slot_bytes_;
        if (to_arena) {
            check_cuda(cudaMemcpyAsync(arena_conv, conv.data, conv_slot_bytes_,
                                       cudaMemcpyDeviceToDevice, stream),
                       "conv state export");
            check_cuda(cudaMemcpyAsync(arena_recurrent, recurrent.data, recurrent_slot_bytes_,
                                       cudaMemcpyDeviceToDevice, stream),
                       "recurrent state export");
        } else {
            check_cuda(cudaMemcpyAsync(conv.data, arena_conv, conv_slot_bytes_,
                                       cudaMemcpyDeviceToDevice, stream),
                       "conv state import");
            check_cuda(cudaMemcpyAsync(recurrent.data, arena_recurrent, recurrent_slot_bytes_,
                                       cudaMemcpyDeviceToDevice, stream),
                       "recurrent state import");
        }
        cursor += conv_slot_bytes_ + recurrent_slot_bytes_;
    }
}

void PrefixSeedStore::copy_kv_pages(const PagedKVPool& pool, const PagedKVAllocation& allocation,
                                    std::uint32_t pages, std::byte* arena_base, std::size_t offset,
                                    bool to_arena, cudaStream_t stream) const {
    const std::span<const std::int32_t> ids = allocation.page_ids();
    if (ids.size() < pages) {
        throw std::logic_error("prefix seed KV span exceeds the mapped allocation");
    }
    std::size_t cursor = offset;
    for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
        const Tensor& plane          = pool.plane(plane_index);
        const std::size_t page_bytes = plane.bytes() / pool.page_group_count();
        auto* plane_base             = static_cast<std::byte*>(plane.data);
        std::uint32_t logical        = 0;
        while (logical < pages) {
            // Coalesce physically-consecutive pages into one transfer.
            std::uint32_t run = 1;
            while (logical + run < pages && ids[logical + run] == ids[logical + run - 1] + 1) {
                ++run;
            }
            std::byte* pool_ptr =
                plane_base + static_cast<std::size_t>(ids[logical]) * page_bytes;
            std::byte* arena_ptr    = arena_base + cursor;
            const std::size_t bytes = static_cast<std::size_t>(run) * page_bytes;
            if (to_arena) {
                check_cuda(cudaMemcpyAsync(arena_ptr, pool_ptr, bytes, cudaMemcpyDeviceToDevice,
                                           stream),
                           "KV page export");
            } else {
                check_cuda(cudaMemcpyAsync(pool_ptr, arena_ptr, bytes, cudaMemcpyDeviceToDevice,
                                           stream),
                           "KV page import");
            }
            cursor += bytes;
            logical += run;
        }
    }
}

void PrefixSeedStore::capture(const PreparedPromptData& prompt, std::uint32_t frontier,
                              std::int32_t rope_delta, const LinearAttentionStatePool& state_pool,
                              std::int32_t state_slot, const Tensor& hidden,
                              const PagedKVPool& text_pool, const PagedKVAllocation& text_kv,
                              const PagedKVPool* backend_pool,
                              const PagedKVAllocation* backend_kv, cudaStream_t stream) {
    if (!enabled()) { return; }
    if (frontier == 0 || frontier > prompt.token_ids.size()) {
        throw std::invalid_argument("prefix seed frontier does not lie inside the prompt");
    }
    if (contains(prompt, frontier)) { return; }
    if (hidden.data == nullptr || hidden.bytes() < hidden_bytes_) {
        throw std::invalid_argument("prefix seed capture requires the captured hidden state");
    }

    const std::uint32_t text_pages =
        1U + (frontier - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
    const std::uint32_t backend_pages =
        (backend_pool != nullptr && backend_kv != nullptr) ? text_pages : 0U;

    Entry entry;
    entry.hash =
        prefix_seed_hash(std::span<const TokenId>(prompt.token_ids.data(), frontier));
    entry.frontier   = frontier;
    entry.rope_delta = rope_delta;
    entry.ledger.assign(prompt.token_ids.begin(),
                        prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(frontier));
    entry.identity.reserve(frontier);
    entry.identity.assign(prompt);
    entry.identity.truncate(frontier);

    const std::size_t text_bytes = static_cast<std::size_t>(text_pages) * kv_page_bytes(text_pool);
    const std::size_t backend_bytes =
        backend_pages != 0 ? static_cast<std::size_t>(backend_pages) * kv_page_bytes(*backend_pool)
                           : 0ULL;
    const std::size_t total =
        state_image_bytes() + hidden_bytes_ + text_bytes + backend_bytes;
    if (total > arena_bytes_) { return; } // cannot ever fit; skip silently
    if (arena_used_ + total > arena_bytes_) {
        // Generation flush: the bump arena reclaims space only wholesale. Captures are cheap and
        // repopulate on demand, so correctness never depends on retained entries.
        entries_.clear();
        arena_used_ = 0;
    }

    entry.arena_offset = arena_used_;
    entry.arena_bytes  = total;
    entry.state_offset = entry.arena_offset;
    entry.hidden_offset = entry.state_offset + state_image_bytes();
    entry.text_kv_offset = entry.hidden_offset + hidden_bytes_;
    entry.text_pages     = text_pages;
    entry.backend_kv_offset = entry.text_kv_offset + text_bytes;
    entry.backend_pages     = backend_pages;

    auto* base = static_cast<std::byte*>(arena_);
    copy_state_image(state_pool, state_slot, base, entry.state_offset, /*to_arena=*/true, stream);
    check_cuda(cudaMemcpyAsync(base + entry.hidden_offset, hidden.data, hidden_bytes_,
                               cudaMemcpyDeviceToDevice, stream),
               "hidden export");
    copy_kv_pages(text_pool, text_kv, text_pages, base, entry.text_kv_offset, /*to_arena=*/true,
                  stream);
    if (backend_pages != 0) {
        copy_kv_pages(*backend_pool, *backend_kv, backend_pages, base, entry.backend_kv_offset,
                      /*to_arena=*/true, stream);
    }

    arena_used_ += total;
    entries_.push_back(std::move(entry));
    std::fprintf(stderr, "ninfer: prefix seed captured frontier=%u bytes=%zu entries=%zu\n",
                 frontier, total, entries_.size());
}

void PrefixSeedStore::restore(std::int64_t index, const LinearAttentionStatePool& state_pool,
                              std::int32_t state_slot, Tensor& tail_hidden,
                              const PagedKVPool& text_pool, const PagedKVAllocation& text_kv,
                              const PagedKVPool* backend_pool,
                              const PagedKVAllocation* backend_kv, cudaStream_t stream) const {
    const Entry& entry = entries_.at(static_cast<std::size_t>(index));
    if (tail_hidden.data == nullptr || tail_hidden.bytes() < hidden_bytes_) {
        throw std::invalid_argument("prefix seed restore requires the lane tail-hidden tensor");
    }
    if (entry.backend_pages != 0 && (backend_pool == nullptr || backend_kv == nullptr)) {
        throw std::logic_error("prefix seed entry carries backend KV the engine no longer has");
    }
    auto* base = static_cast<std::byte*>(arena_);
    copy_state_image(state_pool, state_slot, base, entry.state_offset, /*to_arena=*/false, stream);
    check_cuda(cudaMemcpyAsync(tail_hidden.data, base + entry.hidden_offset, hidden_bytes_,
                               cudaMemcpyDeviceToDevice, stream),
               "hidden import");
    copy_kv_pages(text_pool, text_kv, entry.text_pages, base, entry.text_kv_offset,
                  /*to_arena=*/false, stream);
    if (entry.backend_pages != 0) {
        copy_kv_pages(*backend_pool, *backend_kv, entry.backend_pages, base,
                      entry.backend_kv_offset, /*to_arena=*/false, stream);
    }
}

} // namespace ninfer::targets::qwen3_6::detail
