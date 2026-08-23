#pragma once

// Provider/materializer boundary for the host KV cache.
//
// The provider is pure storage: it owns the parked entries, does the prefix
// lookup, and hands out the buffers a park writes into. The materializer -
// the page copies and hidden-state restore - lives in the target runtime,
// because bytes-per-page and the hidden tensors are target-specific. The
// handoff object between the two is HostKvEntry: the provider allocates it
// (and its slab), the materializer fills it.
//
// The in-process pinned-RAM cache (HostKvCache) is the only implementation
// today. An external provider (LMCache, NVMe) slots in behind this interface
// without touching the materializer; async staging and remote eviction are
// later widenings of the same seam, not constraints designed for now.

#include "core/host_kv_budget.h"
#include "targets/qwen3_6/impl/runtime/host_kv_sequence.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// One parked sequence: its pages live in `slab`, everything the planner reads
// lives here. clear_lane() drops the ledger, frontiers, prefix identity and
// rewrite checkpoint along with the KV, and plan_request_for_lane() consults
// exactly those to choose append over rewind, so a park that saved only pages
// would restore a sequence the planner then treats as empty.
struct HostKvEntry {
    ParkedSequenceMeta meta;
    HostKvSlab* slab = nullptr;

    std::vector<TokenId> ledger;
    ResidentPrefixIdentity prefix_identity;
    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::int32_t rope_delta          = 0;
    bool tail_hidden_valid           = false;

    // Stored as fields rather than a RewriteCheckpoint: that struct lives in
    // program.h, the file upstream churns most, and this header has no other
    // reason to depend on it. RewriteCheckpointKind comes from the exported
    // prepared_prompt.h.
    bool checkpoint_valid                 = false;
    RewriteCheckpointKind checkpoint_kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t checkpoint_frontier     = 0;

    std::uint64_t last_used = 0;
    // Stable identity, assigned on insert and never reused. An entry's index
    // shifts when an earlier entry is dropped, so eviction protection keys on
    // this id rather than the index.
    std::uint64_t id = 0;
};

// How much of an entry a given prompt can reuse, and therefore how many pages
// the restore must copy back.
struct HostKvMatch {
    std::size_t entry_index    = 0;
    std::uint64_t entry_id     = 0;
    std::uint32_t reuse_tokens = 0;  // frontier the prompt matches to
};

// How much of ONE specific entry a given prompt reuses: the append frontier if
// the prompt extends it, else the rewrite checkpoint if the prompt rewinds to
// it, else 0 (the prompt matches neither, so the entry is not usable for it).
// This is the per-entry core of HostKvCache::find(), exposed so an id-pinned
// restore computes the reuse for the exact entry it restores rather than
// re-selecting the best match - which parking another lane can change
// mid-transaction.
[[nodiscard]] inline std::uint32_t host_kv_entry_reuse(const HostKvEntry& entry,
                                                       const PreparedPromptData& prompt) {
    if (entry.execution_frontier != 0 &&
        prefix_matches(prompt, entry.ledger, entry.prefix_identity, entry.execution_frontier)) {
        return entry.execution_frontier;
    }
    if (entry.checkpoint_valid && entry.checkpoint_frontier != 0 &&
        entry.checkpoint_frontier <= prompt.token_ids.size() &&
        prefix_matches(prompt, entry.ledger, entry.prefix_identity, entry.checkpoint_frontier)) {
        return entry.checkpoint_frontier;
    }
    return 0;
}

// Storage side of the host KV cache. Implementations own the parked entries
// and the buffers they live in; the target runtime materializes them back
// into lanes.
class HostKvProvider {
public:
    virtual ~HostKvProvider() = default;

    [[nodiscard]] virtual std::size_t budget_bytes() const = 0;
    // Bytes not currently handed out; 0 for providers that don't track it.
    [[nodiscard]] virtual std::size_t used_bytes() const { return 0; }
    // Largest single contiguous free range; 0 for providers that don't track it.
    [[nodiscard]] virtual std::size_t largest_free_range() const { return 0; }
    [[nodiscard]] virtual std::size_t size() const = 0;
    [[nodiscard]] virtual HostKvEntry& entry(std::size_t index) = 0;
    [[nodiscard]] virtual const HostKvEntry& entry(std::size_t index) const = 0;

    // Best entry for this prompt, preferring the one that reuses the most
    // tokens, so the cache and the lane chooser agree on what "best" means.
    [[nodiscard]] virtual std::optional<HostKvMatch> find(const PreparedPromptData& prompt) const = 0;

    // Stable-id lookup: the evicting-restore transaction binds to the exact
    // entry the probe deferred and must keep addressing that entry even if
    // parking another lane inserts a better match mid-transaction. Returns the
    // entry's index, or nullopt when the entry was evicted.
    [[nodiscard]] virtual std::optional<std::size_t> find_by_id(std::uint64_t entry_id) const = 0;

    // Takes a buffer of `needed_bytes` for a new park, evicting the least
    // recently used entries until one fits (fit-driven: a fragmented free set
    // may evict an entry even when total free bytes suffice). `protect_id`
    // names an entry that must never be evicted to make room (0 protects
    // nothing); the caller uses it to keep the entry a restore is about to
    // consume from being sacrificed to the park that precedes it. Returns
    // nullptr only if no buffer can be made available (every entry is
    // protected, or the budget cannot hold `needed_bytes`).
    [[nodiscard]] virtual HostKvSlab* acquire_slab(std::uint64_t protect_id,
                                                   std::size_t needed_bytes) = 0;

    // Returns a buffer taken by acquire_slab() when the park it was for did
    // not happen. Without this a failed park would leak the buffer.
    virtual void release_slab(HostKvSlab* slab) = 0;

    virtual void insert(std::unique_ptr<HostKvEntry> entry) = 0;

    // Removes an entry and returns its buffer. Used both by eviction and
    // after a successful restore.
    virtual void drop(std::size_t index) = 0;

    // Refreshes an entry's recency so LRU eviction in acquire_slab() ranks by
    // last access, not insertion order. Called when a lookup positively uses an
    // entry that must nonetheless survive the call: restore_lane() defers
    // restoration while the pool cannot reserve the entitlement, leaving the
    // entry cached; without a touch it would keep its insertion recency and be
    // the first evicted once the budget fills, forcing the session it served through a
    // full re-prefill. Pure probes (find/host_kv_reusable_tokens) stay
    // non-touching so the cache tracks actual use, not curiosity.
    virtual void touch(std::size_t index) = 0;
};

}  // namespace ninfer::targets::qwen3_6::detail