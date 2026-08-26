#pragma once

// Content-addressed store of immutable prompt-prefix state snapshots ("seeds").
//
// A seed captures the complete sequence state at a message-boundary frontier F produced by a
// real prefill: every Linear Attention layer's convolution and recurrent state, the Text (and,
// when MTP is active, backend) KV page payloads for tokens [0,F), the hidden state at F-1, and
// the host token ledger with its prefix identity. A later request whose prompt begins with the
// identical F tokens seeds a fresh lane by copying the entry in, then prefills only its suffix.
//
// Entries are immutable and restore-by-copy, so any number of concurrent requests can seed from
// the same entry; nothing is claimed or consumed. The store owns one fixed device arena sized at
// startup (GPU residency stays process-fixed) and holds no KV-pool pages, so admission
// accounting for the shared pools is unchanged. All device transfers are ordered on the caller's
// stream; the store is mutated only from the GPU executor lane.

#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class PrefixSeedStore {
public:
    PrefixSeedStore() = default;
    ~PrefixSeedStore() noexcept;

    PrefixSeedStore(const PrefixSeedStore&)            = delete;
    PrefixSeedStore& operator=(const PrefixSeedStore&) = delete;

    /**
     * Allocates the device arena. budget_bytes==0 leaves the store disabled. The layouts
     * fix every entry's device image sizes except the per-entry KV span, which scales with the
     * entry frontier. Safe to call again after release(); throws if an arena is already live.
     */
    void initialize(std::size_t budget_bytes, const LinearAttentionStatePool& state_pool,
                    const PagedKVPool& text_pool, const PagedKVPool* backend_pool,
                    std::size_t hidden_bytes);

    // Frees the arena and drops every entry. enabled() becomes false. Must run on the GPU
    // executor with the device stream idle.
    void release() noexcept;

    [[nodiscard]] bool enabled() const noexcept { return arena_ != nullptr; }
    [[nodiscard]] std::size_t arena_bytes() const noexcept { return arena_bytes_; }
    [[nodiscard]] std::size_t entry_count() const noexcept { return entries_.size(); }

    /** Exact-token-prefix probe. Returns the entry index or -1. */
    [[nodiscard]] std::int64_t find(const PreparedPromptData& prompt) const;

    /** True when an entry already covers exactly this prompt's first `frontier` tokens. */
    [[nodiscard]] bool contains(const PreparedPromptData& prompt, std::uint32_t frontier) const;

    [[nodiscard]] std::uint32_t entry_frontier(std::int64_t entry) const;
    [[nodiscard]] std::int32_t entry_rope_delta(std::int64_t entry) const;
    [[nodiscard]] bool entry_matches(std::int64_t entry, const PreparedPromptData& prompt) const;

    /**
     * Copies the state at `frontier` into a new entry. `state_slot` names the Linear Attention
     * pool slot holding the captured image (the lane's rewrite-checkpoint slot immediately after
     * the in-graph capture), `hidden` the captured hidden state at frontier-1, and the
     * allocations the sequence's live KV whose leading pages cover [0,frontier). When the
     * remaining arena cannot hold the new entry, the store flushes every resident entry and
     * resets the bump offset, then captures if the empty arena can hold it. Silently skips
     * capture when the entry cannot fit even in an empty arena.
     */
    void capture(const PreparedPromptData& prompt, std::uint32_t frontier, std::int32_t rope_delta,
                 const LinearAttentionStatePool& state_pool, std::int32_t state_slot,
                 const Tensor& hidden, const PagedKVPool& text_pool,
                 const PagedKVAllocation& text_kv, const PagedKVPool* backend_pool,
                 const PagedKVAllocation* backend_kv, cudaStream_t stream);

    /**
     * Copies entry state into a lane: Linear Attention image into `state_slot`, KV payloads into
     * the leading pages of the destination allocations, and the entry hidden into `tail_hidden`.
     * Host-side sequence fields (ledger, identity, frontiers) are the caller's responsibility,
     * fed from tokens()/entry_rope_delta().
     */
    void restore(std::int64_t entry, const LinearAttentionStatePool& state_pool,
                 std::int32_t state_slot, Tensor& tail_hidden, const PagedKVPool& text_pool,
                 const PagedKVAllocation& text_kv, const PagedKVPool* backend_pool,
                 const PagedKVAllocation* backend_kv, cudaStream_t stream) const;

    [[nodiscard]] std::span<const TokenId> tokens(std::int64_t entry) const;

private:
    struct Entry {
        std::uint64_t hash      = 0;
        std::uint32_t frontier  = 0;
        std::int32_t rope_delta = 0;
        std::vector<TokenId> ledger;
        ResidentPrefixIdentity identity;
        std::size_t arena_offset = 0;
        std::size_t arena_bytes  = 0;
        std::size_t state_offset = 0; // conv+recurrent images, layer-major
        std::size_t hidden_offset = 0;
        std::size_t text_kv_offset = 0;
        std::uint32_t text_pages   = 0;
        std::size_t backend_kv_offset = 0;
        std::uint32_t backend_pages   = 0;
    };

    [[nodiscard]] std::size_t state_image_bytes() const noexcept;
    [[nodiscard]] std::size_t kv_page_bytes(const PagedKVPool& pool) const;
    void copy_kv_pages(const PagedKVPool& pool, const PagedKVAllocation& allocation,
                       std::uint32_t pages, std::byte* arena_base, std::size_t offset,
                       bool to_arena, cudaStream_t stream) const;
    void copy_state_image(const LinearAttentionStatePool& state_pool, std::int32_t slot,
                          std::byte* arena_base, std::size_t offset, bool to_arena,
                          cudaStream_t stream) const;

    void* arena_             = nullptr;
    std::size_t arena_bytes_ = 0;
    std::size_t arena_used_  = 0; // bump offset; a full arena is reclaimed by wholesale flush
    std::deque<Entry> entries_;

    // Fixed per-entry geometry captured at initialize().
    std::uint32_t state_layers_        = 0;
    std::size_t conv_slot_bytes_       = 0; // one layer's conv image for one slot
    std::size_t recurrent_slot_bytes_  = 0; // one layer's recurrent image for one slot
    std::size_t hidden_bytes_          = 0;
};

[[nodiscard]] std::uint64_t prefix_seed_hash(std::span<const TokenId> tokens);

} // namespace ninfer::targets::qwen3_6::detail
