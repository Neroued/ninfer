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

#include "core/host_kv_budget.h"
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
    explicit HostKvCache(std::size_t budget_bytes) : budget_(budget_bytes) {}

    [[nodiscard]] std::size_t size() const noexcept override { return entries_.size(); }
    [[nodiscard]] std::size_t budget_bytes() const noexcept override {
        return budget_.budget_bytes();
    }
    [[nodiscard]] std::size_t used_bytes() const noexcept override {
        return budget_.used_bytes();
    }
    [[nodiscard]] std::size_t largest_free_range() const noexcept override {
        return budget_.largest_free_range();
    }

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
            const std::uint32_t reuse = host_kv_entry_reuse(*entries_[i], prompt);
            if (reuse != 0 && (!best || reuse > best->reuse_tokens)) {
                best = HostKvMatch{.entry_index = i, .entry_id = entries_[i]->id, .reuse_tokens = reuse};
            }
        }
        return best;
    }

    // Stable-id lookup (see the interface): the evicting-restore transaction
    // addresses the exact entry it deferred, not whichever entry is currently
    // the best match.
    [[nodiscard]] std::optional<std::size_t> find_by_id(std::uint64_t entry_id) const override {
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i]->id == entry_id) { return i; }
        }
        return std::nullopt;
    }

    // Takes a buffer of `needed_bytes` for a new park, evicting the least
    // recently used entries until one fits. Eviction is fit-driven, not purely
    // size-driven: a fragmented free set may force evicting an entry even when
    // total free bytes suffice (`largest_free_range()` exposes this).
    // `protect_id` is never evicted: it is the entry a restore is about to
    // consume, and sacrificing it to the park that precedes the restore would
    // defeat the cache. Returns nullptr when no buffer can be made available.
    [[nodiscard]] HostKvSlab* acquire_slab(std::uint64_t protect_id,
                                           std::size_t needed_bytes) override {
        if (HostKvSlab* slab = budget_.acquire(needed_bytes)) { return slab; }
        // Evicting only helps if the request can ever fit as one contiguous
        // range. The protected entry (if any) is the only immovable region: it
        // sits at a fixed offset and splits the address space, so a sum of free
        // bytes can overstate what a single range can hold. Bail out before
        // evicting anything when even releasing every unprotected entry cannot
        // make room - without this, an unsatisfiable park would evict the whole
        // cache one entry at a time and still fail. The model assumes the
        // protected entry is the only immovable region: a slab handed out by a
        // concurrent acquire_slab but not yet inserted is invisible to this
        // scan, so a caller must not hold such a slab across another
        // acquire_slab (park_lane is the only caller today, and it runs
        // single-threaded).
        std::size_t protected_offset = 0, protected_bytes = 0;
        for (const auto& e : entries_) {
            if (e->id == protect_id) {
                protected_offset = budget_.slab_offset(e->slab);
                protected_bytes  = e->slab->bytes();
                break;
            }
        }
        if (!budget_.can_satisfy(needed_bytes, protected_offset, protected_bytes)) {
            return nullptr;
        }
        // No single free range fits yet: evict LRU entries (skipping the
        // protected one) until the budget can hold `needed_bytes`, or no
        // unprotected entry remains.
        while (true) {
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
            if (HostKvSlab* slab = budget_.acquire(needed_bytes)) { return slab; }
        }
    }

    // Returns a slab taken by acquire_slab() when the park it was for did not
    // happen. Without this a failed park would leak the slab.
    void release_slab(HostKvSlab* slab) override { budget_.release(slab); }

    void insert(std::unique_ptr<HostKvEntry> entry) override {
        entry->id        = ++id_clock_;
        entry->last_used = ++clock_;
        entries_.push_back(std::move(entry));
    }

    // Removes an entry and returns its slab to the arena. Used both by LRU
    // eviction and after a successful restore.
    void drop(std::size_t index) override {
        if (index >= entries_.size()) { return; }
        budget_.release(entries_[index]->slab);
        entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
    }

    void touch(std::size_t index) override {
        if (index < entries_.size()) { entries_[index]->last_used = ++clock_; }
    }

private:
    HostKvBudget budget_;
    std::vector<std::unique_ptr<HostKvEntry>> entries_;
    std::uint64_t clock_     = 0;
    std::uint64_t id_clock_  = 0;
};

}  // namespace ninfer::targets::qwen3_6::detail
