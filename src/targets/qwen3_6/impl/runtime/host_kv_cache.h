#pragma once

// L2 cache of evicted sequences, held in pinned host RAM.
//
// At --max-concurrency 1 a second request forces the retained sequence out of
// its lane, and today that discards its KV: the next turn re-prefills from
// scratch, measured at 28s average and 95s worst at 231k context. Copying the
// pages to host RAM instead costs 0.31s round-trip for the same 231k, so the
// sequence becomes restorable rather than lost.
//
// Entries are keyed by *session identity*, not by lane. A parked sequence no
// longer occupies a lane, and whichever lane it used may be handed to an
// unrelated request in the meantime, so lane is not a stable name for it.
// Lookup instead replays the planner's own test, prefix_matches(), against each
// entry's ledger and prefix identity.

#include "core/host_kv_arena.h"
#include "core/host_kv_log.h"
#include "targets/qwen3_6/impl/runtime/host_kv_provider.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class HostKvCache : public HostKvProvider {
public:
    HostKvCache(std::size_t slab_count, std::size_t slab_bytes)
        : arena_(slab_count, slab_bytes) {}

    [[nodiscard]] std::size_t size() const noexcept override { return entries_.size(); }
    [[nodiscard]] std::size_t free_slabs() const noexcept { return arena_.free_slabs(); }
    [[nodiscard]] std::size_t slab_bytes() const noexcept override { return arena_.slab_bytes(); }

    [[nodiscard]] HostKvEntry& entry(std::size_t index) override { return *entries_[index]; }
    [[nodiscard]] const HostKvEntry& entry(std::size_t index) const override { return *entries_[index]; }

    // Best entry for this prompt, preferring the one that reuses the most
    // tokens. Mirrors find_admission_lane's "most reusable wins" rule so the
    // cache and the lane chooser agree on what "best" means.
    //
    // Checks the append frontier first, then the rewrite checkpoint: 21% of
    // observed requests rewind rather than append, so an entry that cannot be
    // appended to is still frequently useful.
    [[nodiscard]] std::optional<HostKvMatch> find(const PreparedPromptData& prompt) const override {
        std::optional<HostKvMatch> best;
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            const HostKvEntry& e = *entries_[i];
            std::uint32_t reuse  = 0;
            if (e.execution_frontier != 0 &&
                prefix_matches(prompt, e.ledger, e.prefix_identity, e.execution_frontier)) {
                reuse = e.execution_frontier;
            } else if (e.checkpoint_valid && e.checkpoint_frontier != 0 &&
                       e.checkpoint_frontier <= prompt.token_ids.size() &&
                       prefix_matches(prompt, e.ledger, e.prefix_identity,
                                      e.checkpoint_frontier)) {
                reuse = e.checkpoint_frontier;
            }
            if (reuse != 0 && (!best || reuse > best->reuse_tokens)) {
                best = HostKvMatch{.entry_index = i, .entry_id = e.id, .reuse_tokens = reuse};
            }
        }
        return best;
    }

    // Takes a slab for a new park, evicting the least recently used entry when
    // none is free. `protect_id` is never evicted: it is the entry a restore is
    // about to consume, and sacrificing it to the park that precedes the restore
    // would defeat the cache. Returns nullptr when no slab can be made available.
    [[nodiscard]] HostKvSlab* acquire_slab(std::uint64_t protect_id = 0) override {
        if (HostKvSlab* slab = arena_.acquire()) { return slab; }
        if (entries_.empty()) { return nullptr; }

        std::size_t victim = 0;
        std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
        bool have_victim    = false;
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i]->id == protect_id) { continue; }
            if (entries_[i]->last_used < oldest) {
                oldest      = entries_[i]->last_used;
                victim      = i;
                have_victim = true;
            }
        }
        if (!have_victim) { return nullptr; }  // every entry is protected
        host_kv_log("host KV LRU evicting entry " + std::to_string(victim) +
                    " (" + std::to_string(entries_[victim]->execution_frontier) +
                    " tokens) to make room");
        drop(victim);
        return arena_.acquire();
    }

    // Returns a slab taken by acquire_slab() when the park it was for did not
    // happen. Without this a failed park would leak the slab.
    void release_slab(HostKvSlab* slab) override { arena_.release(slab); }

    void insert(std::unique_ptr<HostKvEntry> entry) override {
        entry->id        = ++id_clock_;
        entry->last_used = ++clock_;
        entries_.push_back(std::move(entry));
    }

    // Removes an entry and returns its slab to the arena. Used both by LRU
    // eviction and after a successful restore.
    void drop(std::size_t index) override {
        if (index >= entries_.size()) { return; }
        arena_.release(entries_[index]->slab);
        entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
    }

    void touch(std::size_t index) override {
        if (index < entries_.size()) { entries_[index]->last_used = ++clock_; }
    }

private:
    HostKvArena arena_;
    std::vector<std::unique_ptr<HostKvEntry>> entries_;
    std::uint64_t clock_     = 0;
    std::uint64_t id_clock_  = 0;
};

}  // namespace ninfer::targets::qwen3_6::detail
