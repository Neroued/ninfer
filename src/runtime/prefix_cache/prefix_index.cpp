#include "runtime/prefix_cache/prefix_index.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

namespace ninfer::runtime::prefix_cache {
namespace {

constexpr std::uint8_t kBacked   = 0;
constexpr std::uint8_t kUnbacked = 1;
constexpr std::uint8_t kNoList   = 0xff;

[[noreturn]] void invariant(const char* message) { throw std::logic_error(message); }

} // namespace

PrefixCacheIndex::PrefixCacheIndex(const PrefixIndexConfig& config, PrefixIndexBackend& backend)
    : config_(config), backend_(&backend) {
    if (config.max_nodes == 0 || config.max_snapshots == 0) {
        throw std::invalid_argument("prefix cache index requires node and snapshot capacity");
    }
    if (config.max_nodes > kNoId / 2 || config.max_snapshots > kNoId / 2) {
        throw std::invalid_argument("prefix cache index capacity is not representable");
    }
    if (config.host_slabs != 0 && config.image_slabs == 0) {
        throw std::invalid_argument("prefix cache host tier requires a positive image slab count");
    }
    nodes_.resize(config.max_nodes);
    free_nodes_.resize(config.max_nodes);
    for (std::uint32_t i = 0; i < config.max_nodes; ++i) {
        free_nodes_[i] = config.max_nodes - 1U - i;
    }
    snapshots_.resize(config.max_snapshots);
    free_snapshots_.resize(config.max_snapshots);
    for (std::uint32_t i = 0; i < config.max_snapshots; ++i) {
        free_snapshots_[i] = config.max_snapshots - 1U - i;
    }
    lru_links_.resize(static_cast<std::size_t>(config.max_nodes) + config.max_snapshots);
    // A min-heap: allocation always takes the lowest free slab, so a batch of single-slab
    // allocations comes out ascending and mostly consecutive and its copies coalesce into runs.
    free_slabs_.resize(config.host_slabs);
    for (std::uint32_t i = 0; i < config.host_slabs; ++i) { free_slabs_[i] = i; }
    slot_state_.assign(config.device_snapshot_slots, SlotState::Free);
    free_slots_ = config.device_snapshot_slots;
    slot_owner_.assign(config.device_snapshot_slots, kNoId);
    children_.reserve(config.max_nodes);
}

void PrefixCacheIndex::set_cost(const CacheCostModel& cost) {
    config_.cost = cost;
    for (std::uint32_t index = 0; index < snapshots_.size(); ++index) {
        if (snapshots_[index].occupied) { update_priority(index); }
    }
}

// ---- handles -----------------------------------------------------------------------------------

bool PrefixCacheIndex::valid(NodeRef node) const noexcept {
    return node.index < nodes_.size() && nodes_[node.index].occupied &&
           nodes_[node.index].generation == node.generation;
}

bool PrefixCacheIndex::valid(SnapshotRef snapshot) const noexcept {
    return snapshot.index < snapshots_.size() && snapshots_[snapshot.index].occupied &&
           snapshots_[snapshot.index].generation == snapshot.generation;
}

PrefixCacheIndex::Node& PrefixCacheIndex::require(NodeRef node) {
    if (!valid(node)) { invariant("stale prefix cache node reference"); }
    return nodes_[node.index];
}

const PrefixCacheIndex::Node& PrefixCacheIndex::require(NodeRef node) const {
    if (!valid(node)) { invariant("stale prefix cache node reference"); }
    return nodes_[node.index];
}

PrefixCacheIndex::Snapshot& PrefixCacheIndex::require(SnapshotRef snapshot) {
    if (!valid(snapshot)) { invariant("stale prefix cache snapshot reference"); }
    return snapshots_[snapshot.index];
}

const PrefixCacheIndex::Snapshot& PrefixCacheIndex::require(SnapshotRef snapshot) const {
    if (!valid(snapshot)) { invariant("stale prefix cache snapshot reference"); }
    return snapshots_[snapshot.index];
}

NodeRef PrefixCacheIndex::ref_of_node(std::uint32_t index) const noexcept {
    if (index == kNoId) { return {}; }
    return NodeRef{index, nodes_[index].generation};
}

SnapshotRef PrefixCacheIndex::ref_of_snapshot(std::uint32_t index) const noexcept {
    return SnapshotRef{index, snapshots_[index].generation};
}

std::uint64_t PrefixCacheIndex::child_key(std::uint32_t parent,
                                          std::uint64_t lookup_hash) noexcept {
    return lookup_hash ^ ((static_cast<std::uint64_t>(parent) + 1ULL) * 0x9e3779b97f4a7c15ULL);
}

// ---- lookup ------------------------------------------------------------------------------------

std::optional<NodeRef> PrefixCacheIndex::find_child(NodeRef parent, std::uint64_t lookup_hash,
                                                    std::span<const TokenId> block_tokens,
                                                    std::uint64_t extra) const {
    if (block_tokens.size() != kBlockTokens) {
        throw std::invalid_argument("prefix cache block must contain 64 tokens");
    }
    const std::uint32_t parent_index = parent.valid() ? parent.index : kNoId;
    if (parent.valid()) { (void)require(parent); }
    const auto [begin, end] = children_.equal_range(child_key(parent_index, lookup_hash));
    for (auto it = begin; it != end; ++it) {
        const Node& node = nodes_[it->second];
        if (node.parent == parent_index && node.hash == lookup_hash && node.extra == extra &&
            std::equal(block_tokens.begin(), block_tokens.end(), node.tokens.begin())) {
            return ref_of_node(it->second);
        }
    }
    return std::nullopt;
}

bool PrefixCacheIndex::snapshot_valid(const Snapshot& snapshot) const noexcept {
    const bool image = snapshot.device_slot != kNoId || snapshot.host == CopyState::Resident;
    const bool tail  = snapshot.tail_len == 0 || snapshot.tail_device_copy == CopyState::Resident ||
                       (config_.host_blocks && snapshot.host == CopyState::Resident);
    return image && tail;
}

bool PrefixCacheIndex::tail_matches(const Snapshot& snapshot,
                                    std::span<const TokenId> prompt) const noexcept {
    const std::uint32_t begin = snapshot.frontier - snapshot.tail_len;
    if (prompt.size() < snapshot.frontier) { return false; }
    return std::equal(snapshot.tail.begin(), snapshot.tail.begin() + snapshot.tail_len,
                      prompt.begin() + begin);
}

MatchResult PrefixCacheIndex::match(std::span<const TokenId> tokens,
                                    std::span<const std::uint64_t> block_hashes,
                                    std::span<const std::uint64_t> block_extras,
                                    std::uint32_t prompt_tokens) const {
    if (tokens.size() < prompt_tokens) {
        throw std::invalid_argument("prefix cache match prompt exceeds its tokens");
    }
    MatchResult result;
    if (prompt_tokens < 2) { return result; }
    const std::uint32_t max_frontier = prompt_tokens - 1U;
    const std::uint32_t max_blocks   = max_frontier / kBlockTokens;

    std::vector<std::uint32_t> host_only;
    std::vector<std::uint32_t> filling;
    NodeRef parent;
    for (std::uint32_t block = 0; block < max_blocks && block < block_hashes.size(); ++block) {
        const std::uint64_t extra = block_extras.empty() ? 0ULL : block_extras[block];
        const auto child          = find_child(
            parent, block_hashes[block],
            tokens.subspan(static_cast<std::size_t>(block) * kBlockTokens, kBlockTokens), extra);
        if (!child) { break; }
        const Node& node                    = nodes_[child->index];
        const std::uint32_t prior_host_only = host_only.empty() ? 0U : host_only.back();
        const std::uint32_t prior_filling   = filling.empty() ? 0U : filling.back();
        host_only.push_back(prior_host_only + (node.device == CopyState::Absent ? 1U : 0U));
        filling.push_back(prior_filling + (node.device == CopyState::Filling ? 1U : 0U));
        result.path.push_back(*child);
        parent = *child;
    }

    auto consider = [&](std::uint32_t snapshot_index, std::uint32_t path_blocks) {
        const Snapshot& snapshot = snapshots_[snapshot_index];
        if (snapshot.frontier > max_frontier || !snapshot_valid(snapshot) ||
            !tail_matches(snapshot, tokens)) {
            return;
        }
        MatchCandidate candidate;
        candidate.snapshot         = ref_of_snapshot(snapshot_index);
        candidate.frontier         = snapshot.frontier;
        candidate.path_blocks      = path_blocks;
        candidate.host_only_blocks = path_blocks == 0 ? 0U : host_only[path_blocks - 1U];
        candidate.filling_blocks   = path_blocks == 0 ? 0U : filling[path_blocks - 1U];
        candidate.tail             = snapshot.tail_len != 0;
        candidate.tail_on_device   = snapshot.tail_device_copy == CopyState::Resident;
        candidate.image_on_device  = snapshot.device_slot != kNoId;
        candidate.restore_bytes =
            static_cast<std::uint64_t>(candidate.host_only_blocks) * config_.block_bytes +
            (candidate.image_on_device ? 0ULL : config_.image_bytes) +
            (candidate.tail && !candidate.tail_on_device ? config_.block_bytes : 0ULL);
        result.candidates.push_back(candidate);
    };
    for (const std::uint32_t snapshot : root_snapshots_) { consider(snapshot, 0); }
    for (std::size_t depth = 0; depth < result.path.size(); ++depth) {
        for (const std::uint32_t snapshot : nodes_[result.path[depth].index].snapshots) {
            consider(snapshot, static_cast<std::uint32_t>(depth + 1));
        }
    }
    std::stable_sort(result.candidates.begin(), result.candidates.end(),
                     [](const MatchCandidate& left, const MatchCandidate& right) {
                         return left.frontier > right.frontier;
                     });
    return result;
}

AdmissionChoice PrefixCacheIndex::choose(const MatchResult& match,
                                         std::uint32_t prompt_tokens) const {
    AdmissionChoice best;
    best.predicted_seconds     = config_.cost.prefill_seconds(0, prompt_tokens);
    std::uint64_t best_restore = 0;
    for (std::size_t i = 0; i < match.candidates.size(); ++i) {
        const MatchCandidate& candidate = match.candidates[i];
        const double seconds =
            config_.cost.restore_seconds(candidate.restore_bytes) +
            config_.cost.prefill_seconds(candidate.frontier, prompt_tokens - candidate.frontier);
        const bool better = seconds < best.predicted_seconds ||
                            (seconds == best.predicted_seconds && best.candidate &&
                             candidate.restore_bytes < best_restore);
        if (better) {
            best.candidate         = i;
            best.predicted_seconds = seconds;
            best_restore           = candidate.restore_bytes;
        }
    }
    return best;
}

void PrefixCacheIndex::note_hit(SnapshotRef snapshot) {
    Snapshot& entry = require(snapshot);
    ++entry.hits;
    ++counters_.snapshot_hits;
    entry.last_hit_tick = ++tick_;
    entry.priority_base = inflation_;
    if (!entry.superseded) {
        update_priority(snapshot.index);
        return;
    }
    // Resumed from again (a retry, or a second branch): retained until this lineage moves on.
    entry.superseded      = false;
    entry.superseded_tick = 0;
    adjust_retained(entry.anchor, +1);
    std::vector<std::uint32_t> dependents;
    collect_retained_dependents(snapshot.index, dependents);
    revalue_around(snapshot.index, dependents);
}

bool PrefixCacheIndex::shared_below(const Snapshot& snapshot) const noexcept {
    // The blocks right after the snapshot: children of its anchor whose tokens begin with its
    // tail. Two of them, or a branch further down the one chain they start, mean another
    // conversation has continued from this prefix differently.
    const std::vector<std::uint32_t>& first =
        snapshot.anchor == kNoId ? root_children_ : nodes_[snapshot.anchor].children;
    std::uint32_t chain = kNoId;
    for (const std::uint32_t child : first) {
        const auto tail = snapshot.tail.begin();
        if (!std::equal(tail, tail + snapshot.tail_len, nodes_[child].tokens.begin())) { continue; }
        if (chain != kNoId) { return true; }
        chain = child;
    }
    while (chain != kNoId) {
        const std::vector<std::uint32_t>& children = nodes_[chain].children;
        if (children.size() > 1) { return true; }
        chain = children.empty() ? kNoId : children.front();
    }
    return false;
}

void PrefixCacheIndex::supersede(SnapshotRef snapshot) {
    Snapshot& entry = require(snapshot);
    // A boundary serves the conversations that share it, so it is superseded only while the tree
    // shows no other conversation below it: until then it is just this lineage's snapshot.
    if (entry.superseded || (entry.kind == SnapshotKind::Boundary && shared_below(entry))) {
        return;
    }
    std::vector<std::uint32_t> dependents;
    collect_retained_dependents(snapshot.index, dependents);
    entry.superseded      = true;
    entry.superseded_tick = ++tick_;
    adjust_retained(entry.anchor, -1);
    // Its retained dependents now value what lies between them and the retained ancestor above.
    revalue_around(snapshot.index, dependents);
}

// ---- pins --------------------------------------------------------------------------------------

void PrefixCacheIndex::acquire_path(std::span<const NodeRef> path) {
    std::uint32_t expected_parent = kNoId;
    for (const NodeRef ref : path) {
        const Node& node = require(ref);
        if (node.parent != expected_parent) { invariant("prefix cache path is not a root path"); }
        expected_parent = ref.index;
    }
    for (const NodeRef ref : path) {
        ++nodes_[ref.index].pins;
        refresh_node(ref.index);
    }
}

void PrefixCacheIndex::release_path(std::span<const NodeRef> path) {
    std::uint32_t expected_parent = kNoId;
    for (const NodeRef ref : path) {
        const Node& node = require(ref);
        if (node.parent != expected_parent) { invariant("prefix cache path is not a root path"); }
        if (node.pins == 0) { invariant("prefix cache path released without a pin"); }
        expected_parent = ref.index;
    }
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        --nodes_[it->index].pins;
        refresh_node(it->index);
    }
}

void PrefixCacheIndex::pin_path_from(std::uint32_t node) {
    for (std::uint32_t current = node; current != kNoId; current = nodes_[current].parent) {
        ++nodes_[current].pins;
        refresh_node(current);
    }
}

void PrefixCacheIndex::unpin_path_from(std::uint32_t node, bool) {
    for (std::uint32_t current = node; current != kNoId; current = nodes_[current].parent) {
        if (nodes_[current].pins == 0) { invariant("prefix cache node unpinned without a pin"); }
        --nodes_[current].pins;
        refresh_node(current);
    }
}

void PrefixCacheIndex::pin_node(NodeRef node) {
    (void)require(node);
    pin_path_from(node.index);
}

void PrefixCacheIndex::unpin_node(NodeRef node) {
    (void)require(node);
    unpin_path_from(node.index, true);
}

void PrefixCacheIndex::pin_snapshot(SnapshotRef snapshot) {
    ++require(snapshot).pins;
    refresh_tail(snapshot.index);
}

void PrefixCacheIndex::unpin_snapshot(SnapshotRef snapshot) {
    Snapshot& entry = require(snapshot);
    if (entry.pins == 0) { invariant("prefix cache snapshot unpinned without a pin"); }
    --entry.pins;
    refresh_tail(snapshot.index);
}

// ---- membership maintenance ------------------------------------------------------------------

void PrefixCacheIndex::refresh_node(std::uint32_t index) {
    const Node& node = nodes_[index];
    const bool lru   = node.occupied && node.device == CopyState::Resident && node.pins == 0;
    const std::uint8_t list =
        lru ? (node.host == CopyState::Resident ? kBacked : kUnbacked) : kNoList;
    if (lru_links_[index].list != list) {
        if (lru_links_[index].list != kNoList) { lru_remove(index); }
        if (list != kNoList) { lru_append(index, list); }
    }
    const bool dead =
        node.occupied && node.host == CopyState::Resident && node.live_below == 0 && node.pins == 0;
    if (dead != node.in_dead) {
        if (dead) {
            dead_append(index);
        } else {
            dead_remove(index);
        }
    }
}

void PrefixCacheIndex::refresh_tail(std::uint32_t index) {
    const Snapshot& snapshot  = snapshots_[index];
    const std::uint32_t entry = config_.max_nodes + index;
    const bool lru    = snapshot.occupied && snapshot.tail_len != 0 &&
                        snapshot.tail_device_copy == CopyState::Resident && snapshot.pins == 0;
    const bool backed = config_.host_blocks && snapshot.host == CopyState::Resident;
    const std::uint8_t list = lru ? (backed ? kBacked : kUnbacked) : kNoList;
    if (lru_links_[entry].list != list) {
        if (lru_links_[entry].list != kNoList) { lru_remove(entry); }
        if (list != kNoList) { lru_append(entry, list); }
    }
}

void PrefixCacheIndex::lru_remove(std::uint32_t entry) {
    LruLink& link = lru_links_[entry];
    if (link.list == kNoList) { return; }
    LruList& list = lru_[link.list];
    if (link.prev != kNoId) {
        lru_links_[link.prev].next = link.next;
    } else {
        list.head = link.next;
    }
    if (link.next != kNoId) {
        lru_links_[link.next].prev = link.prev;
    } else {
        list.tail = link.prev;
    }
    --list.count;
    link = LruLink{};
}

void PrefixCacheIndex::lru_append(std::uint32_t entry, std::uint8_t list_id) {
    LruLink& link = lru_links_[entry];
    LruList& list = lru_[list_id];
    link.list     = list_id;
    link.prev     = list.tail;
    link.next     = kNoId;
    if (list.tail != kNoId) {
        lru_links_[list.tail].next = entry;
    } else {
        list.head = entry;
    }
    list.tail = entry;
    ++list.count;
}

std::uint32_t PrefixCacheIndex::lru_pop(std::uint8_t list_id) {
    const std::uint32_t entry = lru_[list_id].head;
    if (entry != kNoId) { lru_remove(entry); }
    return entry;
}

void PrefixCacheIndex::dead_remove(std::uint32_t index) {
    Node& node = nodes_[index];
    if (!node.in_dead) { return; }
    if (node.dead_prev != kNoId) {
        nodes_[node.dead_prev].dead_next = node.dead_next;
    } else {
        dead_head_ = node.dead_next;
    }
    if (node.dead_next != kNoId) {
        nodes_[node.dead_next].dead_prev = node.dead_prev;
    } else {
        dead_tail_ = node.dead_prev;
    }
    node.dead_prev = kNoId;
    node.dead_next = kNoId;
    node.in_dead   = false;
}

void PrefixCacheIndex::dead_append(std::uint32_t index) {
    Node& node     = nodes_[index];
    node.dead_prev = dead_tail_;
    node.dead_next = kNoId;
    if (dead_tail_ != kNoId) {
        nodes_[dead_tail_].dead_next = index;
    } else {
        dead_head_ = index;
    }
    dead_tail_   = index;
    node.in_dead = true;
}

void PrefixCacheIndex::adjust_live(std::uint32_t anchor, int delta) {
    for (std::uint32_t current = anchor; current != kNoId; current = nodes_[current].parent) {
        Node& node = nodes_[current];
        if (delta < 0 && node.live_below == 0) {
            invariant("prefix cache live snapshot count underflow");
        }
        node.live_below = delta < 0 ? node.live_below - 1U : node.live_below + 1U;
        refresh_node(current);
    }
}

void PrefixCacheIndex::set_device(Node& node, CopyState state) noexcept {
    if (node.device == CopyState::Resident) { --device_resident_; }
    if (state == CopyState::Resident) { ++device_resident_; }
    node.device = state;
}

void PrefixCacheIndex::set_slot(std::uint32_t slot, SlotState state) noexcept {
    if (slot_state_[slot] == SlotState::Free) { --free_slots_; }
    if (state == SlotState::Free) { ++free_slots_; }
    slot_state_[slot] = state;
}

// ---- tree mutation -----------------------------------------------------------------------------

InsertResult PrefixCacheIndex::insert_block(NodeRef parent, std::uint64_t lookup_hash,
                                            std::span<const TokenId> block_tokens,
                                            std::uint64_t extra, std::uint32_t device_id) {
    const std::uint32_t parent_index = parent.valid() ? parent.index : kNoId;
    if (parent.valid() && require(parent).pins == 0) {
        invariant("prefix cache insertion parent is not pinned by the inserting sequence");
    }
    if (device_id == kNoId) { throw std::invalid_argument("prefix cache block has no device id"); }
    if (const auto existing = find_child(parent, lookup_hash, block_tokens, extra)) {
        Node& node = nodes_[existing->index];
        ++node.pins;
        bool attached = false;
        if (node.device == CopyState::Absent) {
            // A host-only block recomputed by this sequence regains its Device copy.
            node.device_id = device_id;
            set_device(node, CopyState::Resident);
            attached = true;
        }
        refresh_node(existing->index);
        return InsertResult{*existing, false, attached};
    }
    if (free_nodes_.empty()) { invariant("prefix cache node capacity exhausted"); }
    const std::uint32_t index = free_nodes_.back();
    free_nodes_.pop_back();
    Node& node    = nodes_[index];
    node.occupied = true;
    node.parent   = parent_index;
    node.depth    = parent.valid() ? nodes_[parent_index].depth + 1U : 0U;
    node.hash     = lookup_hash;
    node.extra    = extra;
    std::copy(block_tokens.begin(), block_tokens.end(), node.tokens.begin());
    node.children.clear();
    node.snapshots.clear();
    node.device_id = device_id;
    node.device    = CopyState::Absent;
    set_device(node, CopyState::Resident);
    node.host_slab      = kNoId;
    node.host           = CopyState::Absent;
    node.pins           = 1;
    node.live_below     = 0;
    node.retained_below = 0;
    if (parent.valid()) {
        nodes_[parent_index].children.push_back(index);
    } else {
        root_children_.push_back(index);
    }
    children_.emplace(child_key(parent_index, lookup_hash), index);
    ++node_count_;
    refresh_node(index);
    return InsertResult{ref_of_node(index), true, true};
}

void PrefixCacheIndex::begin_device_fill(NodeRef ref, std::uint32_t device_id) {
    Node& node = require(ref);
    if (node.device != CopyState::Absent || node.host != CopyState::Resident || node.pins == 0 ||
        device_id == kNoId) {
        invariant("prefix cache device fill requires a pinned host-only node");
    }
    set_device(node, CopyState::Filling);
    node.device_id = device_id;
    refresh_node(ref.index);
}

void PrefixCacheIndex::complete_device_fill(NodeRef ref) {
    Node& node = require(ref);
    if (node.device != CopyState::Filling) { invariant("prefix cache device fill not open"); }
    set_device(node, CopyState::Resident);
    refresh_node(ref.index);
}

void PrefixCacheIndex::abort_device_fill(NodeRef ref) {
    Node& node = require(ref);
    if (node.device != CopyState::Filling) { invariant("prefix cache device fill not open"); }
    const std::uint32_t id = node.device_id;
    set_device(node, CopyState::Absent);
    node.device_id = kNoId;
    refresh_node(ref.index);
    backend_->release_device_block(id);
}

std::uint32_t PrefixCacheIndex::device_evictable_blocks() const noexcept {
    return lru_[kBacked].count + lru_[kUnbacked].count;
}

std::uint32_t PrefixCacheIndex::device_backed_evictable_blocks() const noexcept {
    return lru_[kBacked].count;
}

void PrefixCacheIndex::drop_node_device_copy(std::uint32_t index) {
    Node& node = nodes_[index];
    if (node.device != CopyState::Resident) { invariant("prefix cache drops a non-resident page"); }
    const std::uint32_t id = node.device_id;
    set_device(node, CopyState::Absent);
    node.device_id = kNoId;
    refresh_node(index);
    backend_->release_device_block(id);
}

void PrefixCacheIndex::drop_node_host_copy(std::uint32_t index) {
    Node& node = nodes_[index];
    if (node.host != CopyState::Resident) { invariant("prefix cache drops a non-resident slab"); }
    const std::uint32_t slab = node.host_slab;
    node.host                = CopyState::Absent;
    node.host_slab           = kNoId;
    free_slab(slab);
    refresh_node(index);
}

std::uint32_t PrefixCacheIndex::evict_device_blocks(std::uint32_t blocks) {
    return evict_device_entries(blocks, true);
}

std::uint32_t PrefixCacheIndex::evict_backed_device_blocks(std::uint32_t blocks) {
    return evict_device_entries(blocks, false);
}

std::uint32_t PrefixCacheIndex::evict_device_entries(std::uint32_t blocks, bool unbacked) {
    std::uint32_t released = 0;
    while (released < blocks) {
        std::uint32_t entry = lru_pop(kBacked);
        if (entry == kNoId && unbacked) { entry = lru_pop(kUnbacked); }
        if (entry == kNoId) { break; }
        if (entry < config_.max_nodes) {
            Node& node = nodes_[entry];
            if (node.host == CopyState::Resident) {
                drop_node_device_copy(entry);
                ++counters_.device_block_evictions;
                ++released;
            } else {
                // Unbacked: the node loses its only copy together with its (unpinned) subtree.
                std::uint32_t pages = 0;
                std::vector<std::uint32_t> stack{entry};
                while (!stack.empty()) {
                    const std::uint32_t current = stack.back();
                    stack.pop_back();
                    if (nodes_[current].device == CopyState::Resident) { ++pages; }
                    for (const std::uint32_t snapshot : nodes_[current].snapshots) {
                        if (snapshots_[snapshot].tail_device_copy == CopyState::Resident) {
                            ++pages;
                        }
                    }
                    for (const std::uint32_t child : nodes_[current].children) {
                        stack.push_back(child);
                    }
                }
                remove_subtree(entry);
                ++counters_.unbacked_node_losses;
                counters_.device_block_evictions += pages;
                released += pages;
            }
        } else {
            const std::uint32_t index = entry - config_.max_nodes;
            Snapshot& snapshot        = snapshots_[index];
            ++counters_.device_block_evictions;
            ++released;
            if (config_.host_blocks && snapshot.host == CopyState::Resident) {
                const std::uint32_t id    = snapshot.tail_device;
                snapshot.tail_device      = kNoId;
                snapshot.tail_device_copy = CopyState::Absent;
                refresh_tail(index);
                backend_->release_device_block(id);
            } else {
                remove_snapshot(index);
            }
        }
    }
    return released;
}

// ---- host residency ---------------------------------------------------------------------------

void PrefixCacheIndex::free_slab(std::uint32_t slab) {
    free_slabs_.push_back(slab);
    std::push_heap(free_slabs_.begin(), free_slabs_.end(), std::greater<>{});
}

bool PrefixCacheIndex::allocate_slabs(std::uint32_t count, std::vector<std::uint32_t>& out,
                                      std::uint32_t protect_snapshot, double claim) {
    out.clear();
    if (count > config_.host_slabs) { return false; }
    while (free_slabs_.size() < count) {
        if (dead_head_ != kNoId) {
            const std::uint32_t victim = dead_head_;
            ++counters_.host_dead_reclaims;
            if (nodes_[victim].device == CopyState::Absent) {
                // The dead node has no other copy; its subtree is dead and unpinned too.
                remove_subtree(victim);
            } else {
                drop_node_host_copy(victim);
            }
            continue;
        }
        const std::uint32_t victim = pick_victim(true, protect_snapshot);
        if (victim == kNoId) { return false; }
        Snapshot& chosen = snapshots_[victim];
        // A retained snapshot only yields its slabs to something worth at least as much: GDSF
        // never inserts what it would evict next.
        if (!chosen.superseded && chosen.priority > claim) { return false; }
        if (chosen.superseded) {
            // Only requests diverging before its lineage's newer snapshot could use it: both
            // copies go, and GDSF time does not advance.
            remove_snapshot(victim);
            ++counters_.superseded_evictions;
            ++counters_.host_snapshot_evictions;
            continue;
        }
        inflation_ = std::max(inflation_, chosen.priority);
        if (chosen.device_slot != kNoId &&
            (chosen.tail_len == 0 || chosen.tail_device_copy == CopyState::Resident)) {
            // Still complete on the Device: only its host copy yields its slabs.
            for (const std::uint32_t slab : chosen.host_slabs) { free_slab(slab); }
            chosen.host_slabs.clear();
            chosen.host = CopyState::Absent;
            refresh_tail(victim);
        } else {
            remove_snapshot(victim);
        }
        ++counters_.host_snapshot_evictions;
    }
    for (std::uint32_t taken = 0; taken < count; ++taken) {
        std::pop_heap(free_slabs_.begin(), free_slabs_.end(), std::greater<>{});
        out.push_back(free_slabs_.back());
        free_slabs_.pop_back();
    }
    return true;
}

std::optional<std::uint32_t> PrefixCacheIndex::begin_host_fill(NodeRef ref) {
    Node& node = require(ref);
    if (node.device != CopyState::Resident || node.host != CopyState::Absent || node.pins == 0) {
        invariant("prefix cache host fill requires a pinned device-only node");
    }
    if (!config_.host_blocks) { return std::nullopt; }
    std::vector<std::uint32_t> slab;
    if (!allocate_slabs(1, slab, kNoId)) { return std::nullopt; }
    Node& filled     = nodes_[ref.index];
    filled.host      = CopyState::Filling;
    filled.host_slab = slab.front();
    refresh_node(ref.index);
    return slab.front();
}

void PrefixCacheIndex::complete_host_fill(NodeRef ref) {
    Node& node = require(ref);
    if (node.host != CopyState::Filling) { invariant("prefix cache host fill not open"); }
    node.host = CopyState::Resident;
    refresh_node(ref.index);
}

void PrefixCacheIndex::abort_host_fill(NodeRef ref) {
    Node& node = require(ref);
    if (node.host != CopyState::Filling) { invariant("prefix cache host fill not open"); }
    free_slab(node.host_slab);
    node.host      = CopyState::Absent;
    node.host_slab = kNoId;
    refresh_node(ref.index);
}

bool PrefixCacheIndex::begin_snapshot_host_fill(SnapshotRef ref) {
    Snapshot& snapshot = require(ref);
    if (snapshot.host != CopyState::Absent || snapshot.pins == 0) {
        invariant("prefix cache snapshot host fill requires a pinned device-only snapshot");
    }
    const std::uint32_t count =
        config_.image_slabs + (config_.host_blocks && snapshot.tail_len != 0 ? 1U : 0U);
    std::vector<std::uint32_t> slabs;
    // The image goes to the Host only if it is worth what it displaces; otherwise it stays
    // Device-only.
    if (!allocate_slabs(count, slabs, ref.index, snapshot.priority)) { return false; }
    Snapshot& filled  = snapshots_[ref.index];
    filled.host_slabs = std::move(slabs);
    filled.host       = CopyState::Filling;
    return true;
}

void PrefixCacheIndex::complete_snapshot_host_fill(SnapshotRef ref) {
    Snapshot& snapshot = require(ref);
    if (snapshot.host != CopyState::Filling) { invariant("prefix cache snapshot fill not open"); }
    snapshot.host = CopyState::Resident;
    refresh_tail(ref.index);
}

void PrefixCacheIndex::abort_snapshot_host_fill(SnapshotRef ref) {
    Snapshot& snapshot = require(ref);
    if (snapshot.host != CopyState::Filling) { invariant("prefix cache snapshot fill not open"); }
    for (const std::uint32_t slab : snapshot.host_slabs) { free_slab(slab); }
    snapshot.host_slabs.clear();
    snapshot.host = CopyState::Absent;
    refresh_tail(ref.index);
}

void PrefixCacheIndex::begin_tail_device_fill(SnapshotRef ref, std::uint32_t device_id) {
    Snapshot& snapshot = require(ref);
    if (!config_.host_blocks || snapshot.tail_len == 0 ||
        snapshot.tail_device_copy != CopyState::Absent || snapshot.host != CopyState::Resident ||
        snapshot.pins == 0 || device_id == kNoId) {
        invariant("prefix cache tail fill requires a pinned host-backed tail");
    }
    snapshot.tail_device      = device_id;
    snapshot.tail_device_copy = CopyState::Filling;
    refresh_tail(ref.index);
}

void PrefixCacheIndex::complete_tail_device_fill(SnapshotRef ref) {
    Snapshot& snapshot = require(ref);
    if (snapshot.tail_device_copy != CopyState::Filling) {
        invariant("prefix cache tail fill not open");
    }
    snapshot.tail_device_copy = CopyState::Resident;
    refresh_tail(ref.index);
}

void PrefixCacheIndex::abort_tail_device_fill(SnapshotRef ref) {
    Snapshot& snapshot = require(ref);
    if (snapshot.tail_device_copy != CopyState::Filling) {
        invariant("prefix cache tail fill not open");
    }
    const std::uint32_t id    = snapshot.tail_device;
    snapshot.tail_device      = kNoId;
    snapshot.tail_device_copy = CopyState::Absent;
    refresh_tail(ref.index);
    backend_->release_device_block(id);
}

// ---- snapshots ---------------------------------------------------------------------------------

std::optional<std::uint32_t> PrefixCacheIndex::acquire_device_slot(double claim) {
    for (std::uint32_t slot = 0; slot < slot_state_.size(); ++slot) {
        if (slot_state_[slot] == SlotState::Free) {
            set_slot(slot, SlotState::Staging);
            return slot;
        }
    }
    const auto take = [&](std::uint32_t slot) {
        ++counters_.device_slot_evictions;
        set_slot(slot, SlotState::Staging);
        return slot;
    };
    const auto drop_device_copy = [&](std::uint32_t slot) {
        const std::uint32_t owner = slot_owner_[slot];
        backend_->drop_snapshot_device_image(ref_of_snapshot(owner));
        snapshots_[owner].device_slot = kNoId;
        slot_owner_[slot]             = kNoId;
        return take(slot);
    };
    // A superseded owner goes first, backed or not: its lineage resumes from a newer snapshot.
    std::uint32_t superseded = kNoId;
    std::uint32_t unbacked   = kNoId;
    std::uint32_t backed     = kNoId;
    for (std::uint32_t slot = 0; slot < slot_state_.size(); ++slot) {
        if (slot_state_[slot] != SlotState::Owned) { continue; }
        const Snapshot& owner = snapshots_[slot_owner_[slot]];
        if (owner.pins != 0) { continue; }
        if (owner.superseded) {
            if (superseded == kNoId ||
                owner.superseded_tick < snapshots_[slot_owner_[superseded]].superseded_tick) {
                superseded = slot;
            }
        } else if (owner.host != CopyState::Resident) {
            const Snapshot* best = unbacked == kNoId ? nullptr : &snapshots_[slot_owner_[unbacked]];
            if (best == nullptr || owner.priority < best->priority ||
                (owner.priority == best->priority && owner.last_hit_tick < best->last_hit_tick)) {
                unbacked = slot;
            }
        } else if (backed == kNoId ||
                   owner.last_hit_tick < snapshots_[slot_owner_[backed]].last_hit_tick) {
            backed = slot;
        }
    }
    if (superseded != kNoId) {
        if (snapshots_[slot_owner_[superseded]].host == CopyState::Resident) {
            return drop_device_copy(superseded);
        }
        remove_snapshot(slot_owner_[superseded]);
        return take(superseded);
    }
    // A Device-only owner is lost when its slot is taken, so value decides: the least valuable
    // goes, before any backed owner gives up its Device copy, but only for a new snapshot worth
    // at least as much. Host eviction leaves such owners behind, and without this they hold their
    // slots for good; a short request's snapshots must not push out the one a long conversation
    // resumes from.
    if (unbacked != kNoId && snapshots_[slot_owner_[unbacked]].priority <= claim) {
        inflation_ = std::max(inflation_, snapshots_[slot_owner_[unbacked]].priority);
        remove_snapshot(slot_owner_[unbacked]);
        return take(unbacked);
    }
    // A backed owner keeps its Host copy: the least recently hit gives up its Device copy.
    if (backed != kNoId) { return drop_device_copy(backed); }
    return std::nullopt;
}

double PrefixCacheIndex::estimate_priority(std::uint32_t base_frontier, std::uint32_t frontier,
                                           bool tail) const noexcept {
    // update_priority for a snapshot that has not been published yet: no hits, and the blocks
    // between its nearest retained snapshot and itself are its own.
    if (frontier <= base_frontier) { return inflation_; }
    const double saved = config_.cost.prefill_seconds(base_frontier, frontier - base_frontier) -
                         config_.cost.restore_seconds(config_.image_bytes);
    const std::uint64_t exclusive_blocks = frontier / kBlockTokens - base_frontier / kBlockTokens;
    const double size =
        static_cast<double>(config_.image_bytes) +
        static_cast<double>(tail ? config_.block_bytes : 0ULL) +
        static_cast<double>(exclusive_blocks) * static_cast<double>(config_.block_bytes);
    return inflation_ + std::max(0.0, saved) / std::max(size, 1.0);
}

void PrefixCacheIndex::release_device_slot(std::uint32_t slot) {
    if (slot >= slot_state_.size() || slot_state_[slot] != SlotState::Staging) {
        invariant("prefix cache releases a device slot it did not stage");
    }
    set_slot(slot, SlotState::Free);
}

PublishResult PrefixCacheIndex::publish_snapshot(NodeRef anchor, std::uint32_t frontier,
                                                 std::span<const TokenId> tail,
                                                 std::optional<std::uint32_t> tail_device_id,
                                                 std::uint32_t device_slot, SnapshotKind kind) {
    if (device_slot >= slot_state_.size() || slot_state_[device_slot] != SlotState::Staging) {
        invariant("prefix cache snapshot image is not in a staging slot");
    }
    if (tail.size() >= kBlockTokens || tail_device_id.has_value() != !tail.empty() ||
        (tail_device_id && *tail_device_id == kNoId)) {
        throw std::invalid_argument("prefix cache snapshot tail is inconsistent");
    }
    const std::uint32_t anchor_index = anchor.valid() ? anchor.index : kNoId;
    const std::uint32_t base = anchor.valid() ? (require(anchor).depth + 1U) * kBlockTokens : 0U;
    if (frontier == 0 || frontier != base + static_cast<std::uint32_t>(tail.size())) {
        throw std::invalid_argument("prefix cache snapshot frontier does not match its anchor");
    }
    std::vector<std::uint32_t>& anchored =
        anchor.valid() ? nodes_[anchor_index].snapshots : root_snapshots_;
    for (const std::uint32_t existing : anchored) {
        const Snapshot& snapshot = snapshots_[existing];
        if (snapshot.tail_len == tail.size() &&
            std::equal(tail.begin(), tail.end(), snapshot.tail.begin())) {
            set_slot(device_slot, SlotState::Free);
            if (tail_device_id) { backend_->release_device_block(*tail_device_id); }
            return PublishResult{ref_of_snapshot(existing), false};
        }
    }
    if (free_snapshots_.empty()) {
        // Make room by evicting a superseded snapshot, otherwise the GDSF minimum.
        const std::uint32_t victim = pick_victim(false, kNoId);
        if (victim == kNoId) {
            set_slot(device_slot, SlotState::Free);
            if (tail_device_id) { backend_->release_device_block(*tail_device_id); }
            return PublishResult{};
        }
        if (snapshots_[victim].superseded) {
            ++counters_.superseded_evictions;
        } else {
            inflation_ = std::max(inflation_, snapshots_[victim].priority);
        }
        remove_snapshot(victim);
        ++counters_.host_snapshot_evictions;
    }
    const std::uint32_t index = free_snapshots_.back();
    free_snapshots_.pop_back();
    Snapshot& snapshot = snapshots_[index];
    snapshot.occupied  = true;
    snapshot.anchor    = anchor_index;
    snapshot.frontier  = frontier;
    snapshot.tail_len  = static_cast<std::uint32_t>(tail.size());
    std::copy(tail.begin(), tail.end(), snapshot.tail.begin());
    snapshot.tail_device      = tail_device_id.value_or(kNoId);
    snapshot.tail_device_copy = tail.empty() ? CopyState::Absent : CopyState::Resident;
    snapshot.device_slot      = device_slot;
    snapshot.host_slabs.clear();
    snapshot.host            = CopyState::Absent;
    snapshot.pins            = 0;
    snapshot.kind            = kind;
    snapshot.hits            = 0;
    snapshot.last_hit_tick   = ++tick_;
    snapshot.priority_base   = inflation_;
    snapshot.superseded      = false;
    snapshot.superseded_tick = 0;
    set_slot(device_slot, SlotState::Owned);
    slot_owner_[device_slot] = index;
    anchored.push_back(index);
    ++snapshot_count_;
    if (anchor.valid()) {
        adjust_live(anchor_index, +1);
        adjust_retained(anchor_index, +1);
    }
    link_snapshot(index);
    std::vector<std::uint32_t> dependents;
    collect_retained_dependents(index, dependents);
    revalue_around(index, dependents);
    refresh_tail(index);
    return PublishResult{ref_of_snapshot(index), true};
}

void PrefixCacheIndex::update_priority(std::uint32_t index) {
    // GDSF value of keeping this snapshot, given that superseded snapshots go first: without it,
    // a request continuing its lineage resumes from the nearest retained snapshot above it and
    // prefills the rest, and evicting it frees its image, its tail and the blocks no other
    // retained snapshot keeps.
    Snapshot& snapshot                    = snapshots_[index];
    const std::uint32_t ancestor          = retained_ancestor(index);
    const std::uint32_t ancestor_frontier = ancestor == kNoId ? 0U : snapshots_[ancestor].frontier;
    const std::uint32_t own               = snapshot.superseded ? 0U : 1U;
    std::uint64_t exclusive_blocks        = 0;
    for (std::uint32_t current = snapshot.anchor;
         current != kNoId && nodes_[current].retained_below <= own;
         current = nodes_[current].parent) {
        ++exclusive_blocks;
    }
    const double saved =
        config_.cost.prefill_seconds(ancestor_frontier, snapshot.frontier - ancestor_frontier) -
        config_.cost.restore_seconds(config_.image_bytes);
    const double cost = std::max(0.0, saved);
    const double size =
        static_cast<double>(config_.image_bytes) +
        static_cast<double>(snapshot.tail_len != 0 ? config_.block_bytes : 0ULL) +
        static_cast<double>(exclusive_blocks) * static_cast<double>(config_.block_bytes);
    const double frequency = 1.0 + static_cast<double>(snapshot.hits);
    snapshot.priority      = snapshot.priority_base + frequency * cost / std::max(size, 1.0);
}

// ---- snapshot ancestry -------------------------------------------------------------------------

bool PrefixCacheIndex::on_path(const Snapshot& above, const Snapshot& below) const noexcept {
    if (above.frontier >= below.frontier) { return false; }
    const std::uint32_t above_blocks = above.anchor == kNoId ? 0U : nodes_[above.anchor].depth + 1U;
    const std::uint32_t below_blocks = below.anchor == kNoId ? 0U : nodes_[below.anchor].depth + 1U;
    if (above_blocks > below_blocks) { return false; }
    // `tail_block` is the node of `below`'s path holding `above`'s tail tokens (absent when both
    // share an anchor); the node above it must be `above`'s anchor.
    std::uint32_t tail_block = kNoId;
    std::uint32_t current    = below.anchor;
    while (current != kNoId && nodes_[current].depth >= above_blocks) {
        tail_block = current;
        current    = nodes_[current].parent;
    }
    if (current != above.anchor) { return false; }
    const auto tail = above.tail.begin();
    if (tail_block == kNoId) { return std::equal(tail, tail + above.tail_len, below.tail.begin()); }
    return std::equal(tail, tail + above.tail_len, nodes_[tail_block].tokens.begin());
}

std::uint32_t PrefixCacheIndex::find_ancestor(std::uint32_t index) const noexcept {
    // Walk up the anchor path; the first node anchoring a snapshot on this path holds the nearest.
    const Snapshot& snapshot = snapshots_[index];
    std::uint32_t best       = kNoId;
    std::uint32_t previous   = kNoId;
    std::uint32_t current    = snapshot.anchor;
    for (;;) {
        const std::vector<std::uint32_t>& anchored =
            current != kNoId ? nodes_[current].snapshots : root_snapshots_;
        // Tokens right after this node on the path: the snapshot's own tail at its anchor.
        const TokenId* next =
            previous == kNoId ? snapshot.tail.data() : nodes_[previous].tokens.data();
        for (const std::uint32_t other : anchored) {
            const Snapshot& candidate = snapshots_[other];
            if (other == index || candidate.frontier >= snapshot.frontier ||
                !std::equal(candidate.tail.begin(), candidate.tail.begin() + candidate.tail_len,
                            next)) {
                continue;
            }
            if (best == kNoId || candidate.frontier > snapshots_[best].frontier) { best = other; }
        }
        if (best != kNoId || current == kNoId) { return best; }
        previous = current;
        current  = nodes_[current].parent;
    }
}

std::vector<std::uint32_t>& PrefixCacheIndex::dependents_of(std::uint32_t snapshot) noexcept {
    return snapshot == kNoId ? top_snapshots_ : snapshots_[snapshot].dependents;
}

void PrefixCacheIndex::link_snapshot(std::uint32_t index) {
    Snapshot& snapshot = snapshots_[index];
    snapshot.ancestor  = find_ancestor(index);
    snapshot.dependents.clear();
    // Snapshots below the new one had its ancestor as their nearest; they now hang from it.
    std::vector<std::uint32_t>& siblings = dependents_of(snapshot.ancestor);
    for (std::size_t i = 0; i < siblings.size();) {
        const std::uint32_t other = siblings[i];
        if (on_path(snapshot, snapshots_[other])) {
            snapshots_[other].ancestor = index;
            snapshot.dependents.push_back(other);
            siblings[i] = siblings.back();
            siblings.pop_back();
        } else {
            ++i;
        }
    }
    siblings.push_back(index);
}

void PrefixCacheIndex::unlink_snapshot(std::uint32_t index) {
    Snapshot& snapshot                   = snapshots_[index];
    std::vector<std::uint32_t>& siblings = dependents_of(snapshot.ancestor);
    siblings.erase(std::find(siblings.begin(), siblings.end(), index));
    for (const std::uint32_t other : snapshot.dependents) {
        snapshots_[other].ancestor = snapshot.ancestor;
        siblings.push_back(other);
    }
    snapshot.dependents.clear();
    snapshot.ancestor = kNoId;
}

std::uint32_t PrefixCacheIndex::retained_ancestor(std::uint32_t index) const noexcept {
    std::uint32_t current = snapshots_[index].ancestor;
    while (current != kNoId && snapshots_[current].superseded) {
        current = snapshots_[current].ancestor;
    }
    return current;
}

void PrefixCacheIndex::collect_retained_dependents(std::uint32_t index,
                                                   std::vector<std::uint32_t>& out) const {
    std::vector<std::uint32_t> stack(snapshots_[index].dependents);
    while (!stack.empty()) {
        const std::uint32_t current = stack.back();
        stack.pop_back();
        if (snapshots_[current].superseded) {
            stack.insert(stack.end(), snapshots_[current].dependents.begin(),
                         snapshots_[current].dependents.end());
        } else {
            out.push_back(current);
        }
    }
}

void PrefixCacheIndex::adjust_retained(std::uint32_t anchor, int delta) {
    for (std::uint32_t current = anchor; current != kNoId; current = nodes_[current].parent) {
        Node& node = nodes_[current];
        if (delta < 0 && node.retained_below == 0) {
            invariant("prefix cache retained snapshot count underflow");
        }
        node.retained_below = delta < 0 ? node.retained_below - 1U : node.retained_below + 1U;
    }
}

void PrefixCacheIndex::revalue_around(std::uint32_t index,
                                      std::span<const std::uint32_t> dependents) {
    update_priority(index);
    for (const std::uint32_t other : dependents) { update_priority(other); }
    const std::uint32_t ancestor = retained_ancestor(index);
    if (ancestor != kNoId) { update_priority(ancestor); }
}

std::uint32_t PrefixCacheIndex::pick_victim(bool host_resident, std::uint32_t protect) const {
    std::uint32_t superseded = kNoId;
    std::uint32_t retained   = kNoId;
    for (std::uint32_t i = 0; i < snapshots_.size(); ++i) {
        const Snapshot& snapshot = snapshots_[i];
        if (!snapshot.occupied || i == protect || snapshot.pins != 0 ||
            (host_resident ? snapshot.host != CopyState::Resident
                           : snapshot.host == CopyState::Filling)) {
            continue;
        }
        if (snapshot.superseded) {
            if (superseded == kNoId ||
                snapshot.superseded_tick < snapshots_[superseded].superseded_tick) {
                superseded = i;
            }
        } else if (retained == kNoId || snapshot.priority < snapshots_[retained].priority ||
                   (snapshot.priority == snapshots_[retained].priority &&
                    snapshot.last_hit_tick < snapshots_[retained].last_hit_tick)) {
            retained = i;
        }
    }
    return superseded != kNoId ? superseded : retained;
}

void PrefixCacheIndex::remove_snapshot(std::uint32_t index) {
    Snapshot& snapshot = snapshots_[index];
    if (!snapshot.occupied) { invariant("prefix cache removes a free snapshot"); }
    if (snapshot.pins != 0) { invariant("prefix cache removes a pinned snapshot"); }
    // Snapshots valued against this one are revalued once it is gone.
    std::vector<std::uint32_t> dependents;
    std::uint32_t retained_above = kNoId;
    if (!snapshot.superseded) {
        collect_retained_dependents(index, dependents);
        retained_above = retained_ancestor(index);
        adjust_retained(snapshot.anchor, -1);
    }
    unlink_snapshot(index);
    snapshot.superseded      = false;
    snapshot.superseded_tick = 0;
    backend_->release_snapshot(ref_of_snapshot(index));
    std::vector<std::uint32_t>& anchored =
        snapshot.anchor != kNoId ? nodes_[snapshot.anchor].snapshots : root_snapshots_;
    anchored.erase(std::find(anchored.begin(), anchored.end(), index));
    if (snapshot.device_slot != kNoId) {
        set_slot(snapshot.device_slot, SlotState::Free);
        slot_owner_[snapshot.device_slot] = kNoId;
        snapshot.device_slot              = kNoId;
    }
    for (const std::uint32_t slab : snapshot.host_slabs) { free_slab(slab); }
    snapshot.host_slabs.clear();
    snapshot.host                   = CopyState::Absent;
    const std::uint32_t tail_device = snapshot.tail_device;
    const bool release_tail         = snapshot.tail_device_copy == CopyState::Resident;
    snapshot.tail_device            = kNoId;
    snapshot.tail_device_copy       = CopyState::Absent;
    snapshot.occupied               = false;
    lru_remove(config_.max_nodes + index);
    const std::uint32_t anchor = snapshot.anchor;
    snapshot.anchor            = kNoId;
    ++snapshot.generation;
    free_snapshots_.push_back(index);
    --snapshot_count_;
    if (anchor != kNoId) { adjust_live(anchor, -1); }
    if (release_tail) { backend_->release_device_block(tail_device); }
    for (const std::uint32_t other : dependents) { update_priority(other); }
    if (retained_above != kNoId) { update_priority(retained_above); }
}

void PrefixCacheIndex::release_node_storage(std::uint32_t index) {
    Node& node = nodes_[index];
    if (node.pins != 0 || node.device == CopyState::Filling || node.host == CopyState::Filling) {
        invariant("prefix cache removes a pinned or transferring node");
    }
    if (node.host == CopyState::Resident) { free_slab(node.host_slab); }
    node.host      = CopyState::Absent;
    node.host_slab = kNoId;
    lru_remove(index);
    dead_remove(index);
    const bool release_device = node.device == CopyState::Resident;
    const std::uint32_t id    = node.device_id;
    set_device(node, CopyState::Absent);
    node.device_id = kNoId;
    if (release_device) { backend_->release_device_block(id); }
}

void PrefixCacheIndex::remove_subtree(std::uint32_t root) {
    std::vector<std::uint32_t> order{root};
    for (std::size_t i = 0; i < order.size(); ++i) {
        for (const std::uint32_t child : nodes_[order[i]].children) { order.push_back(child); }
    }
    for (const std::uint32_t index : order) {
        while (!nodes_[index].snapshots.empty()) {
            remove_snapshot(nodes_[index].snapshots.back());
        }
    }
    // Detach the subtree root from its parent; descendants disappear with it.
    Node& top = nodes_[root];
    std::vector<std::uint32_t>& siblings =
        top.parent != kNoId ? nodes_[top.parent].children : root_children_;
    siblings.erase(std::find(siblings.begin(), siblings.end(), root));
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const std::uint32_t index = *it;
        Node& node                = nodes_[index];
        release_node_storage(index);
        const auto [begin, end] = children_.equal_range(child_key(node.parent, node.hash));
        for (auto entry = begin; entry != end; ++entry) {
            if (entry->second == index) {
                children_.erase(entry);
                break;
            }
        }
        node.children.clear();
        node.occupied       = false;
        node.parent         = kNoId;
        node.live_below     = 0;
        node.retained_below = 0;
        ++node.generation;
        free_nodes_.push_back(index);
        --node_count_;
    }
}

// ---- views -------------------------------------------------------------------------------------

NodeView PrefixCacheIndex::node(NodeRef ref) const {
    const Node& node = require(ref);
    NodeView view;
    view.depth                = node.depth;
    view.parent               = ref_of_node(node.parent);
    view.device_id            = node.device_id;
    view.device               = node.device;
    view.host_slab            = node.host_slab;
    view.host                 = node.host;
    view.pins                 = node.pins;
    view.live_snapshots_below = node.live_below;
    return view;
}

SnapshotView PrefixCacheIndex::snapshot(SnapshotRef ref) const {
    const Snapshot& snapshot = require(ref);
    SnapshotView view;
    view.anchor           = ref_of_node(snapshot.anchor);
    view.frontier         = snapshot.frontier;
    view.tail_len         = snapshot.tail_len;
    view.tail_device      = snapshot.tail_device;
    view.tail_device_copy = snapshot.tail_device_copy;
    view.device_slot      = snapshot.device_slot;
    view.host             = snapshot.host;
    view.host_slabs       = snapshot.host_slabs;
    view.pins             = snapshot.pins;
    view.kind             = snapshot.kind;
    view.hits             = snapshot.hits;
    view.superseded       = snapshot.superseded;
    view.tail             = std::span<const TokenId>(snapshot.tail.data(), snapshot.tail_len);
    return view;
}

BlockIdentity PrefixCacheIndex::block_identity(NodeRef ref) const {
    const Node& node = require(ref);
    return BlockIdentity{
        .parent      = ref_of_node(node.parent),
        .lookup_hash = node.hash,
        .extra       = node.extra,
        .tokens      = std::span<const TokenId>(node.tokens.data(), node.tokens.size()),
    };
}

// ---- persistence -------------------------------------------------------------------------------

void PrefixCacheIndex::collect_persistable(std::vector<NodeRef>& nodes,
                                           std::vector<SnapshotRef>& snapshots) const {
    nodes.clear();
    snapshots.clear();
    std::vector<bool> keep(nodes_.size(), false);
    for (std::uint32_t index = 0; index < snapshots_.size(); ++index) {
        const Snapshot& snapshot = snapshots_[index];
        // Superseded snapshots are the first to go and rarely resumed from: not worth saving.
        if (!snapshot.occupied || snapshot.host != CopyState::Resident || snapshot.superseded) {
            continue;
        }
        bool backed = true;
        for (std::uint32_t current = snapshot.anchor; current != kNoId && backed;
             current               = nodes_[current].parent) {
            backed = nodes_[current].host == CopyState::Resident;
        }
        if (!backed) { continue; }
        for (std::uint32_t current = snapshot.anchor; current != kNoId && !keep[current];
             current               = nodes_[current].parent) {
            keep[current] = true;
        }
        snapshots.push_back(ref_of_snapshot(index));
    }
    for (std::uint32_t index = 0; index < nodes_.size(); ++index) {
        if (keep[index]) { nodes.push_back(ref_of_node(index)); }
    }
    std::stable_sort(nodes.begin(), nodes.end(), [&](NodeRef left, NodeRef right) {
        return nodes_[left.index].depth < nodes_[right.index].depth;
    });
}

bool PrefixCacheIndex::take_free_slabs(std::uint32_t count, std::vector<std::uint32_t>& out) {
    out.clear();
    if (free_slabs_.size() < count) { return false; }
    for (std::uint32_t taken = 0; taken < count; ++taken) {
        std::pop_heap(free_slabs_.begin(), free_slabs_.end(), std::greater<>{});
        out.push_back(free_slabs_.back());
        free_slabs_.pop_back();
    }
    return true;
}

std::optional<RestoredBlock> PrefixCacheIndex::restore_host_block(NodeRef parent,
                                                                  std::uint64_t lookup_hash,
                                                                  std::span<const TokenId> tokens,
                                                                  std::uint64_t extra) {
    if (!config_.host_blocks || tokens.size() != kBlockTokens) {
        throw std::invalid_argument("prefix cache restore requires a Host tier and a full block");
    }
    const std::uint32_t parent_index = parent.valid() ? parent.index : kNoId;
    if (parent.valid() && require(parent).host != CopyState::Resident) {
        invariant("prefix cache restored block has no Host-resident parent");
    }
    if (find_child(parent, lookup_hash, tokens, extra)) { return std::nullopt; }
    std::vector<std::uint32_t> slab;
    if (free_nodes_.empty() || !take_free_slabs(1, slab)) { return std::nullopt; }
    const std::uint32_t index = free_nodes_.back();
    free_nodes_.pop_back();
    Node& node    = nodes_[index];
    node.occupied = true;
    node.parent   = parent_index;
    node.depth    = parent.valid() ? nodes_[parent_index].depth + 1U : 0U;
    node.hash     = lookup_hash;
    node.extra    = extra;
    std::copy(tokens.begin(), tokens.end(), node.tokens.begin());
    node.children.clear();
    node.snapshots.clear();
    node.device_id      = kNoId;
    node.host_slab      = slab.front();
    node.host           = CopyState::Resident;
    node.pins           = 0;
    node.live_below     = 0;
    node.retained_below = 0;
    if (parent.valid()) {
        nodes_[parent_index].children.push_back(index);
    } else {
        root_children_.push_back(index);
    }
    children_.emplace(child_key(parent_index, lookup_hash), index);
    ++node_count_;
    refresh_node(index);
    return RestoredBlock{ref_of_node(index), slab.front()};
}

std::optional<SnapshotRef> PrefixCacheIndex::restore_host_snapshot(NodeRef anchor,
                                                                   std::uint32_t frontier,
                                                                   std::span<const TokenId> tail,
                                                                   SnapshotKind kind,
                                                                   std::uint32_t hits) {
    if (!config_.host_blocks || tail.size() >= kBlockTokens) {
        throw std::invalid_argument("prefix cache restored snapshot is inconsistent");
    }
    const std::uint32_t anchor_index = anchor.valid() ? anchor.index : kNoId;
    const std::uint32_t base = anchor.valid() ? (require(anchor).depth + 1U) * kBlockTokens : 0U;
    if (frontier == 0 || frontier != base + static_cast<std::uint32_t>(tail.size()) ||
        (anchor.valid() && nodes_[anchor_index].host != CopyState::Resident)) {
        throw std::invalid_argument("prefix cache restored snapshot does not match its anchor");
    }
    std::vector<std::uint32_t>& anchored =
        anchor.valid() ? nodes_[anchor_index].snapshots : root_snapshots_;
    for (const std::uint32_t existing : anchored) {
        const Snapshot& other = snapshots_[existing];
        if (other.tail_len == tail.size() &&
            std::equal(tail.begin(), tail.end(), other.tail.begin())) {
            return std::nullopt;
        }
    }
    std::vector<std::uint32_t> slabs;
    const std::uint32_t count = config_.image_slabs + (tail.empty() ? 0U : 1U);
    if (free_snapshots_.empty() || !take_free_slabs(count, slabs)) {
        for (const std::uint32_t slab : slabs) { free_slab(slab); }
        return std::nullopt;
    }
    const std::uint32_t index = free_snapshots_.back();
    free_snapshots_.pop_back();
    Snapshot& snapshot = snapshots_[index];
    snapshot.occupied  = true;
    snapshot.anchor    = anchor_index;
    snapshot.frontier  = frontier;
    snapshot.tail_len  = static_cast<std::uint32_t>(tail.size());
    std::copy(tail.begin(), tail.end(), snapshot.tail.begin());
    snapshot.tail_device      = kNoId;
    snapshot.tail_device_copy = CopyState::Absent;
    snapshot.device_slot      = kNoId;
    snapshot.host_slabs       = std::move(slabs);
    snapshot.host             = CopyState::Resident;
    snapshot.pins             = 0;
    snapshot.kind             = kind;
    snapshot.hits             = hits;
    snapshot.last_hit_tick    = ++tick_;
    snapshot.priority_base    = inflation_;
    snapshot.superseded       = false;
    snapshot.superseded_tick  = 0;
    anchored.push_back(index);
    ++snapshot_count_;
    if (anchor.valid()) {
        adjust_live(anchor_index, +1);
        adjust_retained(anchor_index, +1);
    }
    link_snapshot(index);
    std::vector<std::uint32_t> dependents;
    collect_retained_dependents(index, dependents);
    revalue_around(index, dependents);
    refresh_tail(index);
    return ref_of_snapshot(index);
}

HostRestorePlan plan_host_restore(std::span<const std::int32_t> block_parents,
                                  std::span<const SavedSnapshotShape> snapshots,
                                  std::uint64_t host_slabs, const PrefixIndexConfig& config) {
    const std::size_t block_count = block_parents.size();
    HostRestorePlan plan{.blocks    = std::vector<bool>(block_count, false),
                         .snapshots = std::vector<bool>(snapshots.size(), false)};
    // Blocks on each block's path, itself included.
    std::vector<std::uint32_t> path_blocks(block_count);
    for (std::size_t index = 0; index < block_count; ++index) {
        const std::int32_t parent = block_parents[index];
        if (parent >= static_cast<std::int32_t>(index) || parent < -1) {
            throw std::invalid_argument("saved prefix cache blocks are not parents first");
        }
        path_blocks[index] = parent < 0 ? 1U : path_blocks[static_cast<std::size_t>(parent)] + 1U;
    }
    std::uint64_t total = block_count;
    for (const SavedSnapshotShape& snapshot : snapshots) {
        if (snapshot.anchor >= static_cast<std::int32_t>(block_count) || snapshot.anchor < -1 ||
            snapshot.slabs < config.image_slabs) {
            throw std::invalid_argument("saved prefix cache snapshot is inconsistent");
        }
        total += snapshot.slabs;
    }
    if (total <= host_slabs) {
        plan.blocks.assign(block_count, true);
        plan.snapshots.assign(snapshots.size(), true);
        plan.slabs = total;
        return plan;
    }

    std::vector<double> density(snapshots.size());
    for (std::size_t index = 0; index < snapshots.size(); ++index) {
        const SavedSnapshotShape& snapshot = snapshots[index];
        const std::uint64_t path =
            snapshot.anchor < 0 ? 0U : path_blocks[static_cast<std::size_t>(snapshot.anchor)];
        const double saved = config.cost.prefill_seconds(0, snapshot.frontier) -
                             config.cost.restore_seconds(config.image_bytes);
        const double size =
            static_cast<double>(config.image_bytes) +
            static_cast<double>(snapshot.slabs > config.image_slabs ? config.block_bytes : 0U) +
            static_cast<double>(path) * static_cast<double>(config.block_bytes);
        density[index] =
            (1.0 + static_cast<double>(snapshot.hits)) * std::max(0.0, saved) / std::max(size, 1.0);
    }
    std::vector<std::uint32_t> order(snapshots.size());
    for (std::uint32_t index = 0; index < order.size(); ++index) { order[index] = index; }
    std::stable_sort(order.begin(), order.end(), [&](std::uint32_t left, std::uint32_t right) {
        if (density[left] != density[right]) { return density[left] > density[right]; }
        return snapshots[left].frontier > snapshots[right].frontier;
    });

    std::uint64_t left = host_slabs;
    for (const std::uint32_t index : order) {
        const SavedSnapshotShape& snapshot = snapshots[index];
        std::uint64_t need                 = snapshot.slabs;
        for (std::int32_t block = snapshot.anchor;
             block >= 0 && !plan.blocks[static_cast<std::size_t>(block)];
             block = block_parents[static_cast<std::size_t>(block)]) {
            ++need;
        }
        if (need > left) { continue; }
        for (std::int32_t block = snapshot.anchor;
             block >= 0 && !plan.blocks[static_cast<std::size_t>(block)];
             block = block_parents[static_cast<std::size_t>(block)]) {
            plan.blocks[static_cast<std::size_t>(block)] = true;
        }
        plan.snapshots[index] = true;
        left -= need;
    }
    plan.slabs = host_slabs - left;
    return plan;
}

std::optional<SnapshotRef> PrefixCacheIndex::slot_owner(std::uint32_t slot) const {
    if (slot >= slot_state_.size() || slot_state_[slot] != SlotState::Owned) {
        return std::nullopt;
    }
    return ref_of_snapshot(slot_owner_[slot]);
}

PrefixIndexStats PrefixCacheIndex::stats() const noexcept {
    PrefixIndexStats stats        = counters_;
    stats.nodes                   = node_count_;
    stats.snapshots               = snapshot_count_;
    stats.host_free_slabs         = static_cast<std::uint32_t>(free_slabs_.size());
    stats.gdsf_inflation          = inflation_;
    stats.device_evictable_blocks = device_evictable_blocks();
    stats.device_resident_blocks  = device_resident_;
    stats.free_device_slots       = free_slots_;
    return stats;
}

void PrefixCacheIndex::check_invariants() const {
    std::vector<std::uint32_t> live(nodes_.size(), 0);
    std::vector<std::uint32_t> slab_uses(config_.host_slabs, 0);
    std::uint32_t occupied_nodes = 0;
    for (std::uint32_t i = 0; i < nodes_.size(); ++i) {
        const Node& node = nodes_[i];
        if (!node.occupied) {
            if (lru_links_[i].list != kNoList || node.in_dead) { invariant("free node is linked"); }
            continue;
        }
        ++occupied_nodes;
        if (node.parent != kNoId) {
            const Node& parent = nodes_[node.parent];
            if (!parent.occupied || parent.depth + 1U != node.depth) {
                invariant("node parent is inconsistent");
            }
            if (parent.pins < node.pins) { invariant("pinned node has a less pinned parent"); }
        } else if (node.depth != 0) {
            invariant("root child has nonzero depth");
        }
        if (node.device == CopyState::Absent && node.host == CopyState::Absent) {
            invariant("node has no copy");
        }
        if (node.host != CopyState::Absent) {
            if (node.host_slab >= config_.host_slabs) { invariant("node slab out of range"); }
            ++slab_uses[node.host_slab];
        }
        const bool lru = node.device == CopyState::Resident && node.pins == 0;
        const std::uint8_t list =
            lru ? (node.host == CopyState::Resident ? kBacked : kUnbacked) : kNoList;
        if (lru_links_[i].list != list) { invariant("node LRU membership is inconsistent"); }
        const bool dead =
            node.host == CopyState::Resident && node.live_below == 0 && node.pins == 0;
        if (dead != node.in_dead) { invariant("node dead membership is inconsistent"); }
    }
    if (occupied_nodes != node_count_) { invariant("node count mismatch"); }
    std::uint32_t resident = 0;
    for (const Node& node : nodes_) {
        if (node.occupied && node.device == CopyState::Resident) { ++resident; }
    }
    if (resident != device_resident_) { invariant("device resident count mismatch"); }
    if (static_cast<std::uint32_t>(
            std::count(slot_state_.begin(), slot_state_.end(), SlotState::Free)) != free_slots_) {
        invariant("free device slot count mismatch");
    }
    std::uint32_t occupied_snapshots = 0;
    std::vector<std::uint32_t> retained(nodes_.size(), 0);
    std::uint32_t top_level = 0;
    for (std::uint32_t i = 0; i < snapshots_.size(); ++i) {
        const Snapshot& snapshot = snapshots_[i];
        if (!snapshot.occupied) { continue; }
        ++occupied_snapshots;
        if (!snapshot_valid(snapshot)) { invariant("indexed snapshot is not valid"); }
        for (std::uint32_t current = snapshot.anchor; current != kNoId;
             current               = nodes_[current].parent) {
            ++live[current];
            if (!snapshot.superseded) { ++retained[current]; }
        }
        if (snapshot.ancestor != find_ancestor(i)) {
            invariant("snapshot ancestor is not the nearest snapshot above it");
        }
        const std::vector<std::uint32_t>& siblings =
            snapshot.ancestor == kNoId ? top_snapshots_ : snapshots_[snapshot.ancestor].dependents;
        if (std::count(siblings.begin(), siblings.end(), i) != 1) {
            invariant("snapshot is not listed once under its ancestor");
        }
        if (snapshot.ancestor == kNoId) { ++top_level; }
        for (const std::uint32_t other : snapshot.dependents) {
            if (!snapshots_[other].occupied || snapshots_[other].ancestor != i) {
                invariant("snapshot dependent does not name it as its ancestor");
            }
        }
        for (const std::uint32_t slab : snapshot.host_slabs) {
            if (slab >= config_.host_slabs) { invariant("snapshot slab out of range"); }
            ++slab_uses[slab];
        }
        if (snapshot.device_slot != kNoId &&
            (slot_state_[snapshot.device_slot] != SlotState::Owned ||
             slot_owner_[snapshot.device_slot] != i)) {
            invariant("snapshot slot ownership is inconsistent");
        }
    }
    if (occupied_snapshots != snapshot_count_) { invariant("snapshot count mismatch"); }
    if (top_level != top_snapshots_.size()) {
        invariant("top-level snapshot list is inconsistent");
    }
    for (std::uint32_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].occupied && nodes_[i].live_below != live[i]) {
            invariant("live snapshot count is inconsistent");
        }
        if (nodes_[i].occupied && nodes_[i].retained_below != retained[i]) {
            invariant("retained snapshot count is inconsistent");
        }
    }
    for (const std::uint32_t slab : free_slabs_) { ++slab_uses[slab]; }
    for (const std::uint32_t uses : slab_uses) {
        if (uses != 1) { invariant("host slab is leaked or shared"); }
    }
    std::array<std::uint32_t, 2> counted{};
    for (std::uint8_t list = 0; list < 2; ++list) {
        for (std::uint32_t entry = lru_[list].head; entry != kNoId;
             entry               = lru_links_[entry].next) {
            ++counted[list];
        }
        if (counted[list] != lru_[list].count) { invariant("LRU count mismatch"); }
    }
}

} // namespace ninfer::runtime::prefix_cache
