#include "core/context_pool.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>

namespace ninfer {
namespace {
constexpr auto absent                 = UINT32_MAX;
constexpr auto future                 = UINT32_MAX - 1;
constexpr std::size_t segment_records = 1024;

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > SIZE_MAX - a) throw std::overflow_error("ContextPool: size overflow");
    return a + b;
}

std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (a != 0 && b > SIZE_MAX / a) throw std::overflow_error("ContextPool: size overflow");
    return a * b;
}

std::size_t ceil_div(std::size_t n, std::size_t d) { return n / d + (n % d != 0); }

std::vector<std::size_t> count_requests(std::span<const RecordRequest> requests,
                                        std::size_t classes) {
    std::vector<std::size_t> counts(classes);
    for (const auto& request : requests) {
        if (request.record_class.index >= classes)
            throw std::out_of_range("ContextPool: unknown record class");
        auto& count = counts[request.record_class.index];
        count       = checked_add(count, request.count);
    }
    return counts;
}
} // namespace

ContextPoolGeometry describe_context_pool(std::span<const RecordSpec> specs) {
    ContextPoolGeometry geometry;
    for (const auto& spec : specs) {
        if (spec.bytes == 0 || spec.alignment == 0 || (spec.alignment & (spec.alignment - 1)) != 0)
            throw std::invalid_argument(
                "ContextPool: records need positive bytes and power-of-two alignment");
        auto found = std::find_if(geometry.classes.begin(), geometry.classes.end(),
                                  [&](const auto& entry) { return entry.spec == spec; });
        if (found != geometry.classes.end()) {
            geometry.spec_classes.push_back(
                {static_cast<std::uint32_t>(found - geometry.classes.begin())});
            continue;
        }
        if (geometry.classes.size() >= absent)
            throw std::overflow_error("ContextPool: too many record classes");
        const auto stride = checked_add(spec.bytes, spec.alignment - 1) & ~(spec.alignment - 1);
        geometry.spec_classes.push_back({static_cast<std::uint32_t>(geometry.classes.size())});
        geometry.classes.push_back({spec, stride, 0});
        geometry.alignment   = std::max(geometry.alignment, spec.alignment);
        geometry.block_bytes = std::max(geometry.block_bytes, stride);
    }
    if (geometry.classes.empty()) return geometry;
    geometry.block_bytes =
        checked_add(geometry.block_bytes, geometry.alignment - 1) & ~(geometry.alignment - 1);
    for (auto& record : geometry.classes)
        record.records_per_block = geometry.block_bytes / record.stride;
    return geometry;
}

std::size_t ContextPoolGeometry::required_bytes(std::span<const RecordRequest> demand) const {
    const auto counts  = count_requests(demand, classes.size());
    std::size_t blocks = 0;
    for (std::size_t i = 0; i < classes.size(); ++i)
        blocks = checked_add(blocks, ceil_div(counts[i], classes[i].records_per_block));
    return checked_multiply(blocks, block_bytes);
}

std::size_t ContextPoolGeometry::record_limit(std::size_t capacity_bytes) const {
    if (classes.empty()) return 0;
    std::size_t largest = 0;
    for (const auto& record : classes) largest = std::max(largest, record.records_per_block);
    return checked_multiply(capacity_bytes / block_bytes, largest);
}

struct PoolReservation::Data {
    ContextPool* owner  = nullptr;
    std::uint64_t token = 0;
    std::vector<std::vector<std::uint32_t>> records;
};

struct ContextPool::Impl {
    enum class State : std::uint8_t { Free, Reserved, Live, PendingRetirement, Future };

    struct Entry {
        State state                 = State::Free;
        std::uint32_t record_class  = absent;
        std::uint32_t block         = absent;
        std::uint32_t slot          = absent;
        std::uint64_t generation    = 0;
        std::uint64_t binding_epoch = 0;
        std::uint64_t reservation   = 0;
    };

    struct Block {
        std::uint32_t record_class = absent;
        std::uint32_t heap_index   = absent;
        std::vector<std::uint32_t> slots;
        std::vector<std::uint32_t> free;

        std::size_t occupied() const noexcept { return slots.size() - free.size(); }

        void initialize(std::uint32_t c, std::size_t count) {
            record_class = c;
            slots.assign(count, absent);
            free.resize(count);
            std::iota(free.rbegin(), free.rend(), 0u);
        }

        void clear() noexcept {
            record_class = absent;
            heap_index   = absent;
            slots.clear();
            free.clear();
        }
    };

    struct Class {
        std::vector<std::uint32_t> available;
        std::size_t free_records = 0;
        std::size_t blocks       = 0;
        std::array<std::size_t, 5> records{};
    };

    struct Relocation {
        std::uint32_t entry = absent;
        std::uint32_t block = absent;
        std::uint32_t slot  = absent;
    };

    struct Placement {
        std::uint32_t record_class = absent;
        std::uint32_t block        = absent;
        std::uint32_t slot         = absent;
    };

    struct Candidate {
        std::uint32_t block;
        std::size_t bytes;
        std::size_t occupied;
    };

    struct Hole {
        std::size_t occupied;
        std::uint32_t block;
    };

    struct Plan {
        PoolAssessment assessment;
        std::vector<Block> blocks;
        std::vector<Relocation> relocations;
        std::vector<RecordMove> moves;
        std::vector<Placement> placements;
        std::vector<std::uint32_t> new_entries;
        bool submitted                    = false;
        std::uint64_t owner               = 0;
        PoolReservation::Data* owner_data = nullptr;
    };

    std::uint32_t pool_id;
    std::size_t capacity;
    ContextPoolGeometry geometry;
    std::vector<Block> blocks;
    std::vector<Class> classes;
    std::vector<std::uint32_t> free_blocks;
    std::vector<std::unique_ptr<std::array<Entry, segment_records>>> segments;
    std::vector<std::uint32_t> free_entries;
    std::uint32_t entry_count = 0;
    std::size_t record_limit;
    std::uint64_t revision          = 1;
    std::uint64_t next_reservation  = 1;
    std::size_t occupied_bytes      = 0;
    std::size_t peak_occupied_bytes = 0;
    std::unique_ptr<Plan> maintenance;

    Impl(std::uint32_t id, std::size_t bytes, std::span<const RecordSpec> specs)
        : pool_id(id), capacity(bytes), geometry(describe_context_pool(specs)),
          record_limit(geometry.record_limit(bytes)) {
        if (pool_id == absent) throw std::invalid_argument("ContextPool: invalid pool ID");
        const auto count = geometry.block_bytes == 0 ? 0 : capacity / geometry.block_bytes;
        if (count >= future || record_limit >= future)
            throw std::overflow_error("ContextPool: allocation identity limit exceeded");
        blocks.resize(count);
        free_blocks.resize(count);
        std::iota(free_blocks.rbegin(), free_blocks.rend(), 0u);
        classes.resize(geometry.classes.size());
        for (auto& record_class : classes) record_class.available.reserve(count);
        segments.reserve(ceil_div(record_limit, segment_records));
    }

    Entry& entry(std::uint32_t index) noexcept {
        return (*segments[index / segment_records])[index % segment_records];
    }

    const Entry& entry(std::uint32_t index) const noexcept {
        return (*segments[index / segment_records])[index % segment_records];
    }

    AllocationId id(std::uint32_t index) const noexcept {
        return {pool_id, index, entry(index).generation};
    }

    bool valid(AllocationId allocation) const noexcept {
        return allocation.pool_id == pool_id && allocation.index < entry_count &&
               allocation.generation != 0 &&
               entry(allocation.index).generation == allocation.generation &&
               entry(allocation.index).state != State::Free;
    }

    Entry& require(AllocationId allocation) {
        if (!valid(allocation)) throw std::out_of_range("ContextPool: stale allocation");
        return entry(allocation.index);
    }

    const Entry& require(AllocationId allocation) const {
        if (!valid(allocation)) throw std::out_of_range("ContextPool: stale allocation");
        return entry(allocation.index);
    }

    void require_mutable() const {
        if (maintenance)
            throw std::logic_error("ContextPool: maintenance owns the placement ledger");
    }

    void require_not_submitted() const {
        if (maintenance && maintenance->submitted)
            throw std::logic_error("ContextPool: submitted maintenance owns physical positions");
    }

    std::uint64_t offset(std::uint32_t block, std::uint32_t slot, std::uint32_t c) const noexcept {
        return block * geometry.block_bytes + slot * geometry.classes[c].stride;
    }

    void charge(std::size_t bytes) noexcept {
        occupied_bytes += bytes;
        peak_occupied_bytes = std::max(peak_occupied_bytes, occupied_bytes);
    }

    void set_state(Entry& e, State state) noexcept {
        auto& counts = classes[e.record_class].records;
        --counts[static_cast<std::size_t>(e.state)];
        ++counts[static_cast<std::size_t>(state)];
        e.state = state;
    }

    bool denser(std::uint32_t left, std::uint32_t right) const noexcept {
        const auto a = blocks[left].occupied(), b = blocks[right].occupied();
        return a > b || (a == b && left < right);
    }

    void swap_heap(std::vector<std::uint32_t>& heap, std::size_t a, std::size_t b) noexcept {
        std::swap(heap[a], heap[b]);
        blocks[heap[a]].heap_index = static_cast<std::uint32_t>(a);
        blocks[heap[b]].heap_index = static_cast<std::uint32_t>(b);
    }

    void sift(std::vector<std::uint32_t>& heap, std::size_t index) noexcept {
        while (index > 0) {
            const auto parent = (index - 1) / 2;
            if (!denser(heap[index], heap[parent])) break;
            swap_heap(heap, index, parent);
            index = parent;
        }
        while (index * 2 + 1 < heap.size()) {
            auto child = index * 2 + 1;
            if (child + 1 < heap.size() && denser(heap[child + 1], heap[child])) ++child;
            if (!denser(heap[child], heap[index])) break;
            swap_heap(heap, index, child);
            index = child;
        }
    }

    void remove_available(std::uint32_t index) noexcept {
        auto& block = blocks[index];
        if (block.heap_index == absent) return;
        auto& heap          = classes[block.record_class].available;
        const auto position = block.heap_index;
        const auto last     = heap.back();
        heap.pop_back();
        if (position < heap.size()) {
            heap[position]          = last;
            blocks[last].heap_index = position;
            sift(heap, position);
        }
        block.heap_index = absent;
    }

    void update_available(std::uint32_t index) noexcept {
        auto& block = blocks[index];
        if (block.free.empty()) {
            remove_available(index);
            return;
        }
        auto& heap = classes[block.record_class].available;
        if (block.heap_index == absent) {
            block.heap_index = static_cast<std::uint32_t>(heap.size());
            heap.push_back(index);
        }
        sift(heap, block.heap_index);
    }

    void rebuild_indexes() noexcept {
        free_blocks.clear();
        for (auto& c : classes) {
            c.available.clear();
            c.free_records = 0;
            c.blocks       = 0;
        }
        for (std::uint32_t i = 0; i < blocks.size(); ++i) {
            auto& block      = blocks[i];
            block.heap_index = absent;
            if (block.record_class == absent) {
                free_blocks.push_back(i);
                continue;
            }
            auto& c = classes[block.record_class];
            ++c.blocks;
            c.free_records += block.free.size();
            update_available(i);
        }
    }

    void prepare_entries(std::size_t count) {
        const auto needed = count > free_entries.size() ? count - free_entries.size() : 0;
        const auto total  = checked_add(entry_count, needed);
        if (total > record_limit)
            throw std::logic_error("ContextPool: descriptor capacity disagrees with packing");
        const auto segment_count = ceil_div(total, segment_records);
        free_entries.reserve(std::min(record_limit, segment_count * segment_records));
        while (segments.size() < segment_count)
            segments.push_back(std::make_unique<std::array<Entry, segment_records>>());
    }

    std::uint32_t make_entry(std::uint32_t c, std::uint32_t block, std::uint32_t slot,
                             std::uint64_t owner, State state) noexcept {
        std::uint32_t index;
        if (free_entries.empty())
            index = entry_count++;
        else {
            index = free_entries.back();
            free_entries.pop_back();
        }
        auto& e = entry(index);
        e       = {state, c, block, slot, e.generation + 1, 1, owner};
        ++classes[c].records[static_cast<std::size_t>(state)];
        return index;
    }

    void recycle_entry(std::uint32_t index) noexcept {
        auto& e = entry(index);
        --classes[e.record_class].records[static_cast<std::size_t>(e.state)];
        e.state       = State::Free;
        e.reservation = 0;
        free_entries.push_back(index);
    }

    void erase_entry(std::uint32_t index) noexcept {
        const auto e        = entry(index);
        auto& block         = blocks[e.block];
        block.slots[e.slot] = absent;
        block.free.push_back(e.slot);
        ++classes[e.record_class].free_records;
        occupied_bytes -= geometry.classes[e.record_class].stride;
        if (block.occupied() == 0) {
            remove_available(e.block);
            auto& c = classes[e.record_class];
            c.free_records -= block.free.size();
            --c.blocks;
            block.clear();
            free_blocks.push_back(e.block);
        } else
            update_available(e.block);
        recycle_entry(index);
    }

    void erase_planned_entry(std::uint32_t index) noexcept {
        if (maintenance) {
            auto& plan    = *maintenance;
            const auto& e = entry(index);
            auto block    = e.block;
            auto slot     = e.slot;
            const auto relocation =
                std::find_if(plan.relocations.begin(), plan.relocations.end(),
                             [&](const auto& move) { return move.entry == index; });
            if (relocation != plan.relocations.end()) {
                block = relocation->block;
                slot  = relocation->slot;
                occupied_bytes -= geometry.classes[e.record_class].stride;
                plan.relocations.erase(relocation);
                std::erase_if(plan.moves,
                              [&](const auto& move) { return move.allocation.index == index; });
            }
            auto& target       = plan.blocks[block];
            target.slots[slot] = absent;
            target.free.push_back(slot);
            if (target.occupied() == 0) target.clear();
        }
        erase_entry(index);
    }

    void erase_future(std::uint32_t index) noexcept {
        auto& plan          = *maintenance;
        const auto& e       = entry(index);
        auto& block         = plan.blocks[e.block];
        block.slots[e.slot] = absent;
        block.free.push_back(e.slot);
        if (block.occupied() == 0) block.clear();
        recycle_entry(index);
    }

    PoolAssessment direct_assessment(std::span<const std::size_t> counts) const {
        PoolAssessment result{PoolAvailability::Ready, revision};
        if (maintenance) {
            result.status = PoolAvailability::MaintenanceInProgress;
            return result;
        }
        std::size_t needed = 0;
        for (std::size_t i = 0; i < counts.size(); ++i) {
            const auto additional =
                counts[i] > classes[i].free_records ? counts[i] - classes[i].free_records : 0;
            needed =
                checked_add(needed, ceil_div(additional, geometry.classes[i].records_per_block));
        }
        if (needed > free_blocks.size()) result.status = PoolAvailability::InsufficientCapacity;
        return result;
    }

    // Project counts and whole-block releases without constructing a placement. Only a
    // feasible, fragmented layout needs the relocation planner.
    std::optional<PoolAssessment>
    projected_assessment(std::span<const std::size_t> counts,
                         std::span<const AllocationId> releases) const {
        PoolAssessment result{PoolAvailability::Ready, revision};
        if (maintenance) {
            result.status = PoolAvailability::MaintenanceInProgress;
            return result;
        }
        std::vector<std::pair<std::uint32_t, std::uint32_t>> removed;
        removed.reserve(releases.size());
        for (const auto allocation : releases) {
            const auto& e = require(allocation);
            if (e.state != State::Live && e.state != State::Reserved)
                throw std::invalid_argument(
                    "ContextPool: projected release is not immediately releasable");
            removed.emplace_back(e.block, allocation.index);
        }
        std::sort(removed.begin(), removed.end());
        removed.erase(std::unique(removed.begin(), removed.end()), removed.end());
        std::vector<std::size_t> released(classes.size()), emptied(classes.size());
        auto empty = free_blocks.size();
        for (std::size_t begin = 0; begin < removed.size();) {
            auto end = begin + 1;
            while (end < removed.size() && removed[end].first == removed[begin].first) ++end;
            const auto& block = blocks[removed[begin].first];
            released[block.record_class] += end - begin;
            if (end - begin == block.occupied()) {
                ++emptied[block.record_class];
                ++empty;
            }
            begin = end;
        }
        std::size_t minimum = 0, needed = 0;
        for (std::size_t c = 0; c < classes.size(); ++c) {
            const auto per_block = geometry.classes[c].records_per_block;
            const auto occupied  = classes[c].blocks * per_block - classes[c].free_records;
            minimum              = checked_add(
                minimum, ceil_div(checked_add(occupied - released[c], counts[c]), per_block));
            const auto available = classes[c].free_records + released[c] - emptied[c] * per_block;
            if (counts[c] > available)
                needed = checked_add(needed, ceil_div(counts[c] - available, per_block));
        }
        if (minimum > blocks.size()) {
            result.status        = PoolAvailability::InsufficientCapacity;
            result.reclaim_bytes = checked_multiply(minimum - blocks.size(), geometry.block_bytes);
            return result;
        }
        if (needed <= empty) return result;
        return std::nullopt;
    }

    Plan plan(std::span<const std::size_t> counts, std::span<const AllocationId> releases,
              bool ignore_retirement = false) const {
        Plan result;
        result.assessment = {PoolAvailability::Ready, revision};
        if (maintenance) {
            result.assessment.status = PoolAvailability::MaintenanceInProgress;
            return result;
        }
        result.blocks = blocks;
        for (auto& block : result.blocks) block.free.reserve(block.slots.size());
        std::vector<bool> removed(entry_count);
        for (const auto allocation : releases) {
            const auto& e = require(allocation);
            if (e.state != State::Live && e.state != State::Reserved)
                throw std::invalid_argument(
                    "ContextPool: projected release is not immediately releasable");
            if (removed[allocation.index]) continue;
            removed[allocation.index] = true;
            auto& block               = result.blocks[e.block];
            block.slots[e.slot]       = absent;
            block.free.push_back(e.slot);
        }
        if (ignore_retirement) {
            for (std::uint32_t i = 0; i < entry_count; ++i) {
                const auto& e = entry(i);
                if (e.state != State::PendingRetirement || e.reservation != 0) continue;
                auto& block         = result.blocks[e.block];
                block.slots[e.slot] = absent;
                block.free.push_back(e.slot);
            }
        }
        std::vector<std::size_t> occupied(classes.size()), assigned(classes.size());
        std::size_t empty = 0;
        for (auto& block : result.blocks) {
            if (block.record_class == absent || block.occupied() == 0) {
                block.clear();
                ++empty;
                continue;
            }
            occupied[block.record_class] += block.occupied();
            ++assigned[block.record_class];
        }
        std::vector<std::size_t> required(classes.size()), spare(classes.size());
        std::size_t total = 0, needed = 0;
        for (std::size_t i = 0; i < classes.size(); ++i) {
            required[i] = ceil_div(checked_add(occupied[i], counts[i]),
                                   geometry.classes[i].records_per_block);
            total       = checked_add(total, required[i]);
            if (required[i] > assigned[i])
                needed += required[i] - assigned[i];
            else
                spare[i] = assigned[i] - required[i];
        }
        if (total > blocks.size()) {
            result.assessment.reclaim_bytes =
                checked_multiply(total - blocks.size(), geometry.block_bytes);
            result.assessment.status = PoolAvailability::InsufficientCapacity;
            return result;
        }

        std::vector<Candidate> candidates;
        for (std::uint32_t i = 0; i < result.blocks.size(); ++i) {
            const auto& block = result.blocks[i];
            if (block.record_class == absent || spare[block.record_class] == 0) continue;
            std::size_t payload = 0;
            bool movable        = true;
            for (auto index : block.slots) {
                if (index == absent) continue;
                const auto& e = entry(index);
                if (e.state == State::PendingRetirement && !ignore_retirement) movable = false;
                if (e.state == State::Live) payload += geometry.classes[e.record_class].spec.bytes;
            }
            if (movable) candidates.push_back({i, payload, block.occupied()});
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (a.bytes != b.bytes) return a.bytes < b.bytes;
            if (a.occupied != b.occupied) return a.occupied < b.occupied;
            return a.block < b.block;
        });
        std::vector<bool> donor(blocks.size());
        std::size_t missing = needed > empty ? needed - empty : 0;
        for (const auto& candidate : candidates) {
            if (missing == 0) break;
            const auto c = result.blocks[candidate.block].record_class;
            if (spare[c] == 0) continue;
            donor[candidate.block] = true;
            --spare[c];
            --missing;
        }
        if (missing != 0) {
            result.assessment.status = PoolAvailability::WaitingForRetirement;
            return result;
        }
        std::vector<std::vector<std::uint32_t>> destinations(classes.size());
        for (std::uint32_t i = 0; i < result.blocks.size(); ++i) {
            const auto& block = result.blocks[i];
            if (!donor[i] && block.record_class != absent && !block.free.empty())
                destinations[block.record_class].push_back(i);
        }
        for (std::uint32_t i = 0; i < result.blocks.size(); ++i) {
            if (!donor[i]) continue;
            auto& block   = result.blocks[i];
            auto& choices = destinations[block.record_class];
            for (auto index : block.slots) {
                if (index == absent) continue;
                if (choices.empty())
                    throw std::logic_error("ContextPool: donor has no disjoint destination");
                const auto target = choices.back();
                auto& destination = result.blocks[target];
                const auto slot   = destination.free.back();
                destination.free.pop_back();
                destination.slots[slot] = index;
                result.relocations.push_back({index, target, slot});
                const auto& e = entry(index);
                if (e.state == State::Live) {
                    const auto bytes = geometry.classes[e.record_class].spec.bytes;
                    result.moves.push_back({id(index), offset(e.block, e.slot, e.record_class),
                                            offset(target, slot, e.record_class), bytes});
                    result.assessment.move_bytes += bytes;
                }
                if (destination.free.empty()) choices.pop_back();
            }
            block.clear();
        }

        auto compare = [](const Hole& a, const Hole& b) {
            return a.occupied < b.occupied || (a.occupied == b.occupied && a.block > b.block);
        };
        std::vector<std::uint32_t> empty_blocks;
        for (std::uint32_t i = 0; i < result.blocks.size(); ++i)
            if (result.blocks[i].record_class == absent) empty_blocks.push_back(i);
        for (std::uint32_t c = 0; c < classes.size(); ++c) {
            std::priority_queue<Hole, std::vector<Hole>, decltype(compare)> holes(compare);
            for (std::uint32_t i = 0; i < result.blocks.size(); ++i) {
                const auto& block = result.blocks[i];
                if (block.record_class == c && !block.free.empty())
                    holes.push({block.occupied(), i});
            }
            for (std::size_t count = 0; count < counts[c]; ++count) {
                std::uint32_t target;
                if (holes.empty()) {
                    if (empty_blocks.empty())
                        throw std::logic_error(
                            "ContextPool: complete placement cannot be fulfilled");
                    target = empty_blocks.back();
                    empty_blocks.pop_back();
                    result.blocks[target].initialize(c, geometry.classes[c].records_per_block);
                } else {
                    target = holes.top().block;
                    holes.pop();
                }
                auto& block     = result.blocks[target];
                const auto slot = block.free.back();
                block.free.pop_back();
                block.slots[slot] = future;
                result.placements.push_back({c, target, slot});
                if (!block.free.empty()) holes.push({block.occupied(), target});
            }
        }
        result.assessment.move_count = result.moves.size();
        if (!result.relocations.empty())
            result.assessment.status = PoolAvailability::NeedsMaintenance;
        return result;
    }

    std::unique_ptr<PoolReservation::Data> reservation_data(ContextPool& owner,
                                                            std::span<const std::size_t> counts) {
        auto data   = std::make_unique<PoolReservation::Data>();
        data->owner = &owner;
        data->token = next_reservation;
        data->records.resize(counts.size());
        for (std::size_t c = 0; c < counts.size(); ++c) data->records[c].reserve(counts[c]);
        return data;
    }

    void install_plan(std::unique_ptr<Plan> plan, PoolReservation::Data& data) noexcept {
        plan->owner      = data.token;
        plan->owner_data = &data;
        for (const auto& placement : plan->placements) {
            const auto index = make_entry(placement.record_class, placement.block, placement.slot,
                                          data.token, State::Future);
            plan->blocks[placement.block].slots[placement.slot] = index;
            plan->new_entries.push_back(index);
            data.records[placement.record_class].push_back(index);
        }
        if (plan->relocations.empty()) {
            blocks = std::move(plan->blocks);
            for (auto index : plan->new_entries) {
                auto& e = entry(index);
                set_state(e, State::Reserved);
                charge(geometry.classes[e.record_class].stride);
            }
            rebuild_indexes();
        } else {
            for (const auto& move : plan->relocations)
                charge(geometry.classes[entry(move.entry).record_class].stride);
            maintenance = std::move(plan);
        }
    }

    void discard_cancelled() noexcept {
        for (std::uint32_t i = 0; i < entry_count; ++i) {
            const auto& e = entry(i);
            if (e.state == State::Reserved && e.reservation == 0) erase_entry(i);
        }
    }

    void discard_maintenance() noexcept {
        auto plan = std::move(maintenance);
        if (plan->owner_data) {
            plan->owner_data->records.clear();
            plan->owner_data->token = 0;
        }
        for (const auto& move : plan->relocations)
            occupied_bytes -= geometry.classes[entry(move.entry).record_class].stride;
        for (auto index : plan->new_entries)
            if (entry(index).state == State::Future) recycle_entry(index);
        discard_cancelled();
        ++revision;
    }
};

std::size_t ContextPoolGeometry::metadata_bytes(std::size_t capacity_bytes) const {
    using Pool             = ContextPool::Impl;
    const auto block_count = block_bytes == 0 ? 0 : capacity_bytes / block_bytes;
    const auto records     = record_limit(capacity_bytes);
    const auto segments    = ceil_div(records, segment_records);
    auto bytes             = checked_add(sizeof(ContextPool) + sizeof(Pool),
                                         checked_multiply(classes.size(), sizeof(Pool::Class)));
    bytes = checked_add(bytes, checked_multiply(classes.capacity(), sizeof(RecordClassGeometry)));
    bytes = checked_add(bytes, checked_multiply(spec_classes.capacity(), sizeof(RecordClassId)));
    bytes = checked_add(
        bytes, checked_multiply(segments, segment_records * sizeof(Pool::Entry) + sizeof(void*)));
    const auto block_metadata =
        2 * sizeof(Pool::Block) + sizeof(std::uint32_t) * (classes.size() + 1);
    bytes = checked_add(bytes, checked_multiply(block_count, block_metadata));
    // The live layout and one candidate layout each hold slot IDs and free-slot stacks. Descriptor
    // recycling is shared by classes. Planning only uses existing disjoint holes as move targets,
    // so at most half of all positions can be relocation sources at once.
    bytes = checked_add(bytes, checked_multiply(records, 5 * sizeof(std::uint32_t)));
    const auto move_metadata =
        checked_multiply(records / 2, 2 * (sizeof(Pool::Relocation) + sizeof(RecordMove)));
    const auto request_metadata =
        checked_multiply(records, 2 * sizeof(Pool::Placement) + sizeof(std::uint32_t));
    const auto planner_per_block =
        2 * sizeof(Pool::Candidate) + 2 * sizeof(Pool::Hole) + 4 * sizeof(std::uint32_t);
    const auto planner_per_class = 5 * sizeof(std::size_t) + sizeof(std::vector<std::uint32_t>);
    const auto planner_metadata =
        checked_add(checked_multiply(block_count, planner_per_block),
                    checked_add(ceil_div(records, 8) + ceil_div(block_count, 8),
                                checked_multiply(classes.size(), planner_per_class)));
    return checked_add(
        bytes,
        checked_add(sizeof(Pool::Plan),
                    checked_add(move_metadata, checked_add(request_metadata, planner_metadata))));
}

ContextPool::ContextPool(std::uint32_t pool_id, std::size_t capacity_bytes,
                         std::span<const RecordSpec> specs)
    : impl_(std::make_unique<Impl>(pool_id, capacity_bytes, specs)) {}

ContextPool::~ContextPool() = default;

std::uint32_t ContextPool::pool_id() const noexcept { return impl_->pool_id; }

std::size_t ContextPool::capacity_bytes() const noexcept { return impl_->capacity; }

std::size_t ContextPool::alignment() const noexcept { return impl_->geometry.alignment; }

std::size_t ContextPool::block_bytes() const noexcept { return impl_->geometry.block_bytes; }

std::size_t ContextPool::record_limit() const noexcept { return impl_->record_limit; }

std::size_t ContextPool::allocation_extent(std::size_t additional) const noexcept {
    const auto& p    = *impl_;
    const auto fresh = additional > p.free_entries.size() ? additional - p.free_entries.size() : 0;
    return p.entry_count + std::min(fresh, p.record_limit - p.entry_count);
}

std::uint64_t ContextPool::revision() const noexcept { return impl_->revision; }

RecordClassId ContextPool::record_class(std::size_t spec_index) const {
    return impl_->geometry.spec_classes.at(spec_index);
}

RecordSpec ContextPool::record_spec(RecordClassId record_class) const {
    return impl_->geometry.classes.at(record_class.index).spec;
}

ContextPoolOccupancy ContextPool::usage() const {
    const auto& p = *impl_;
    ContextPoolOccupancy result;
    result.capacity_bytes      = p.capacity;
    result.block_bytes         = p.geometry.block_bytes;
    result.free_block_bytes    = p.free_blocks.size() * p.geometry.block_bytes;
    result.tail_bytes          = p.capacity - p.blocks.size() * p.geometry.block_bytes;
    result.peak_occupied_bytes = p.peak_occupied_bytes;
    result.metadata_bytes      = sizeof(*this) + sizeof(Impl) +
                            p.blocks.capacity() * sizeof(Impl::Block) +
                            p.classes.capacity() * sizeof(Impl::Class) +
                            p.free_blocks.capacity() * sizeof(std::uint32_t) +
                            p.free_entries.capacity() * sizeof(std::uint32_t) +
                            p.segments.capacity() * sizeof(p.segments.front()) +
                            p.segments.size() * segment_records * sizeof(Impl::Entry);
    result.metadata_bytes += p.geometry.classes.capacity() * sizeof(RecordClassGeometry) +
                             p.geometry.spec_classes.capacity() * sizeof(RecordClassId);
    for (const auto& block : p.blocks)
        result.metadata_bytes +=
            (block.slots.capacity() + block.free.capacity()) * sizeof(std::uint32_t);
    result.classes.reserve(p.classes.size());
    for (std::size_t i = 0; i < p.classes.size(); ++i) {
        const auto& c        = p.classes[i];
        const auto& geometry = p.geometry.classes[i];
        const auto count     = [&](Impl::State state) {
            return c.records[static_cast<std::size_t>(state)];
        };
        RecordClassUsage usage{geometry.spec,
                               geometry.stride,
                               geometry.records_per_block,
                               c.blocks,
                               count(Impl::State::Live),
                               count(Impl::State::Reserved),
                               count(Impl::State::PendingRetirement),
                               count(Impl::State::Future),
                               c.free_records};
        result.live_bytes += usage.live_records * geometry.stride;
        result.reserved_bytes += usage.reserved_records * geometry.stride;
        result.pending_retirement_bytes += usage.pending_retirement_records * geometry.stride;
        result.future_reserved_bytes += usage.future_records * geometry.stride;
        result.free_class_bytes += usage.free_records * geometry.stride;
        result.padding_bytes +=
            c.blocks * (p.geometry.block_bytes - geometry.records_per_block * geometry.stride);
        result.metadata_bytes += c.available.capacity() * sizeof(std::uint32_t);
        result.classes.push_back(usage);
    }
    if (p.maintenance) {
        for (const auto& move : p.maintenance->relocations) {
            const auto c     = p.entry(move.entry).record_class;
            const auto bytes = p.geometry.classes[c].stride;
            result.relocation_bytes += bytes;
            result.reserved_bytes += bytes;
            result.free_class_bytes -= bytes;
            --result.classes[c].free_records;
        }
        result.metadata_bytes += sizeof(Impl::Plan) +
                                 p.maintenance->blocks.capacity() * sizeof(Impl::Block) +
                                 p.maintenance->relocations.capacity() * sizeof(Impl::Relocation) +
                                 p.maintenance->moves.capacity() * sizeof(RecordMove) +
                                 p.maintenance->placements.capacity() * sizeof(Impl::Placement) +
                                 p.maintenance->new_entries.capacity() * sizeof(std::uint32_t);
        for (const auto& block : p.maintenance->blocks)
            result.metadata_bytes +=
                (block.slots.capacity() + block.free.capacity()) * sizeof(std::uint32_t);
    }
    return result;
}

PoolAssessment ContextPool::assess(std::span<const RecordRequest> demand,
                                   std::span<const AllocationId> releases) const {
    const auto counts = count_requests(demand, impl_->classes.size());
    if (releases.empty()) {
        const auto direct = impl_->direct_assessment(counts);
        if (direct.status != PoolAvailability::InsufficientCapacity) return direct;
    }
    const auto projected = impl_->projected_assessment(counts, releases);
    auto result          = projected ? *projected : impl_->plan(counts, releases).assessment;
    if (result.status == PoolAvailability::InsufficientCapacity &&
        std::any_of(
            impl_->classes.begin(), impl_->classes.end(),
            [](const auto& c) {
                return c.records[static_cast<std::size_t>(Impl::State::PendingRetirement)] != 0;
            }) &&
        impl_->plan(counts, releases, true).assessment.feasible())
        result.status = PoolAvailability::WaitingForRetirement;
    return result;
}

PoolReserveResult ContextPool::reserve(std::span<const RecordRequest> demand) {
    auto& p           = *impl_;
    const auto counts = count_requests(demand, p.classes.size());
    auto assessment   = p.direct_assessment(counts);
    std::unique_ptr<Impl::Plan> plan;
    if (assessment.status == PoolAvailability::MaintenanceInProgress)
        return {assessment, std::nullopt};
    if (assessment.status != PoolAvailability::Ready) {
        plan       = std::make_unique<Impl::Plan>(p.plan(counts, {}));
        assessment = plan->assessment;
        if (!assessment.feasible()) {
            plan.reset();
            if (assessment.status == PoolAvailability::InsufficientCapacity &&
                p.plan(counts, {}, true).assessment.feasible())
                assessment.status = PoolAvailability::WaitingForRetirement;
            return {assessment, std::nullopt};
        }
    }
    auto data        = p.reservation_data(*this, counts);
    const auto total = std::accumulate(counts.begin(), counts.end(), std::size_t{0});
    p.prepare_entries(total);
    if (plan) {
        plan->new_entries.reserve(total);
        p.install_plan(std::move(plan), *data);
    } else {
        struct PreparedBlock {
            std::uint32_t index;
            Impl::Block block;
        };

        std::vector<PreparedBlock> prepared;
        std::size_t free_cursor = p.free_blocks.size();
        for (std::uint32_t c = 0; c < counts.size(); ++c) {
            const auto extra =
                counts[c] > p.classes[c].free_records ? counts[c] - p.classes[c].free_records : 0;
            const auto count = ceil_div(extra, p.geometry.classes[c].records_per_block);
            for (std::size_t i = 0; i < count; ++i) {
                Impl::Block block;
                block.initialize(c, p.geometry.classes[c].records_per_block);
                prepared.push_back({p.free_blocks[--free_cursor], std::move(block)});
            }
        }
        p.free_blocks.resize(free_cursor);
        for (auto& block : prepared) {
            auto& c = p.classes[block.block.record_class];
            c.free_records += block.block.free.size();
            ++c.blocks;
            p.blocks[block.index] = std::move(block.block);
            p.update_available(block.index);
        }
        for (std::uint32_t c = 0; c < counts.size(); ++c) {
            for (std::size_t i = 0; i < counts[c]; ++i) {
                const auto target = p.classes[c].available.front();
                auto& block       = p.blocks[target];
                const auto slot   = block.free.back();
                block.free.pop_back();
                --p.classes[c].free_records;
                const auto index =
                    p.make_entry(c, target, slot, data->token, Impl::State::Reserved);
                block.slots[slot] = index;
                data->records[c].push_back(index);
                p.charge(p.geometry.classes[c].stride);
                p.update_available(target);
            }
        }
    }
    ++p.next_reservation;
    ++p.revision;
    return {assessment, PoolReservation(std::move(data))};
}

PoolAssessment ContextPool::resize(PoolReservation& reservation,
                                   std::span<const RecordRequest> demand) {
    auto& p = *impl_;
    if (!reservation.belongs_to(*this))
        throw std::invalid_argument("ContextPool: foreign reservation");
    if (p.maintenance) return {PoolAvailability::MaintenanceInProgress, p.revision};
    if (!reservation.ready()) return {PoolAvailability::WaitingForRetirement, p.revision};
    const auto counts = count_requests(demand, p.classes.size());
    std::vector<AllocationId> releases;
    releases.reserve(reservation.remaining());
    for (const auto& records : reservation.data_->records)
        for (auto index : records) releases.push_back(p.id(index));
    auto plan       = std::make_unique<Impl::Plan>(p.plan(counts, releases));
    auto assessment = plan->assessment;
    if (!assessment.feasible()) return assessment;
    auto data        = p.reservation_data(*this, counts);
    const auto total = std::accumulate(counts.begin(), counts.end(), std::size_t{0});
    plan->new_entries.reserve(total);
    p.prepare_entries(total > releases.size() ? total - releases.size() : 0);
    for (const auto allocation : releases) p.erase_entry(allocation.index);
    reservation.data_.reset();
    p.install_plan(std::move(plan), *data);
    reservation.data_ = std::move(data);
    ++p.next_reservation;
    ++p.revision;
    return assessment;
}

bool ContextPool::valid(AllocationId allocation) const noexcept { return impl_->valid(allocation); }

RecordBinding ContextPool::resolve(AllocationId allocation) const {
    const auto& p = *impl_;
    const auto& e = p.require(allocation);
    if (e.state != Impl::State::Live && e.state != Impl::State::PendingRetirement)
        throw std::logic_error("ContextPool: allocation has not been materialized");
    if (p.maintenance && p.maintenance->submitted)
        throw std::logic_error("ContextPool: bindings unavailable during maintenance");
    return {allocation,
            {e.record_class},
            p.offset(e.block, e.slot, e.record_class),
            p.geometry.classes[e.record_class].spec.bytes,
            e.binding_epoch};
}

void ContextPool::release(AllocationId allocation) {
    auto& p = *impl_;
    p.require_not_submitted();
    if (p.require(allocation).state != Impl::State::Live)
        throw std::logic_error("ContextPool: release requires an owned materialized allocation");
    p.erase_planned_entry(allocation.index);
    ++p.revision;
}

void ContextPool::defer_release(AllocationId allocation) {
    auto& p = *impl_;
    auto& e = p.require(allocation);
    if (e.state != Impl::State::Live)
        throw std::logic_error("ContextPool: retirement already scheduled");
    p.set_state(e, Impl::State::PendingRetirement);
    ++p.revision;
}

void ContextPool::retire(AllocationId allocation) {
    auto& p = *impl_;
    p.require_not_submitted();
    auto& e = p.require(allocation);
    if (e.state != Impl::State::PendingRetirement)
        throw std::logic_error("ContextPool: allocation has no pending retirement");
    if (e.reservation == 0)
        p.erase_planned_entry(allocation.index);
    else {
        p.set_state(e, Impl::State::Reserved);
        ++e.generation;
        e.binding_epoch = 1;
        if (p.maintenance)
            std::erase_if(p.maintenance->moves, [&](const auto& move) {
                return move.allocation.index == allocation.index;
            });
    }
    ++p.revision;
}

void ContextPool::cancel_deferred_release(AllocationId allocation) {
    auto& p = *impl_;
    auto& e = p.require(allocation);
    if (e.state != Impl::State::PendingRetirement || e.reservation != 0)
        throw std::logic_error("ContextPool: deferred release belongs to another reservation");
    p.set_state(e, Impl::State::Live);
    ++p.revision;
}

void ContextPool::return_to_reservation(AllocationId allocation, PoolReservation& reservation) {
    auto& p = *impl_;
    p.require_mutable();
    if (!reservation.belongs_to(*this))
        throw std::invalid_argument("ContextPool: foreign reservation");
    auto& e = p.require(allocation);
    if (e.state != Impl::State::Live)
        throw std::logic_error("ContextPool: returning allocation is not materialized");
    reservation.data_->records[e.record_class].push_back(allocation.index);
    p.set_state(e, Impl::State::Reserved);
    e.reservation = reservation.data_->token;
    ++e.generation;
    e.binding_epoch = 1;
    ++p.revision;
}

void ContextPool::defer_to_reservation(AllocationId allocation, PoolReservation& reservation) {
    auto& p = *impl_;
    p.require_mutable();
    if (!reservation.belongs_to(*this))
        throw std::invalid_argument("ContextPool: foreign retirement reservation");
    auto& e       = p.require(allocation);
    auto& records = reservation.data_->records.at(e.record_class);
    if (e.state != Impl::State::Live || records.size() == records.capacity())
        throw std::logic_error("ContextPool: retirement exceeds its prepared return capacity");
    records.push_back(allocation.index);
    p.set_state(e, Impl::State::PendingRetirement);
    e.reservation = reservation.data_->token;
    ++p.revision;
}

void PoolReservation::prepare_returns(RecordClassId record_class, std::size_t count) {
    if (!valid()) throw std::logic_error("ContextPool: empty retirement reservation");
    auto& records = data_->records.at(record_class.index);
    records.reserve(records.size() + count);
}

bool ContextPool::has_maintenance() const noexcept { return impl_->maintenance != nullptr; }

std::span<const RecordMove> ContextPool::maintenance_moves() const noexcept {
    return impl_->maintenance ? std::span(impl_->maintenance->moves)
                              : std::span<const RecordMove>{};
}

void ContextPool::begin_maintenance() {
    auto& p = *impl_;
    if (!p.maintenance || p.maintenance->submitted)
        throw std::logic_error("ContextPool: no unsubmitted maintenance plan");
    p.maintenance->submitted = true;
    ++p.revision;
}

void ContextPool::complete_maintenance() {
    auto& p = *impl_;
    if (!p.maintenance || !p.maintenance->submitted)
        throw std::logic_error("ContextPool: maintenance was not submitted");
    auto plan = std::move(p.maintenance);
    for (const auto& move : plan->relocations) {
        auto& e = p.entry(move.entry);
        e.block = move.block;
        e.slot  = move.slot;
        ++e.binding_epoch;
        p.occupied_bytes -= p.geometry.classes[e.record_class].stride;
    }
    p.blocks = std::move(plan->blocks);
    for (auto index : plan->new_entries) {
        auto& e = p.entry(index);
        if (e.state != Impl::State::Future) continue;
        if (e.reservation == 0) {
            auto& block         = p.blocks[e.block];
            block.slots[e.slot] = absent;
            block.free.push_back(e.slot);
            if (block.occupied() == 0) block.clear();
            p.recycle_entry(index);
        } else {
            p.set_state(e, Impl::State::Reserved);
            p.charge(p.geometry.classes[e.record_class].stride);
        }
    }
    p.rebuild_indexes();
    p.discard_cancelled();
    ++p.revision;
}

void ContextPool::cancel_maintenance() {
    auto& p = *impl_;
    if (!p.maintenance) return;
    if (p.maintenance->submitted)
        throw std::logic_error(
            "ContextPool: submitted maintenance must complete before cancellation");
    p.discard_maintenance();
}

void ContextPool::abandon_maintenance_after_failure() {
    if (impl_->maintenance) impl_->discard_maintenance();
}

PoolReservation::PoolReservation() noexcept = default;

PoolReservation::PoolReservation(std::unique_ptr<Data> data) noexcept : data_(std::move(data)) {}

PoolReservation::~PoolReservation() { cancel(); }

PoolReservation::PoolReservation(PoolReservation&& other) noexcept = default;

PoolReservation& PoolReservation::operator=(PoolReservation&& other) noexcept {
    if (this != &other) {
        cancel();
        data_ = std::move(other.data_);
    }
    return *this;
}

bool PoolReservation::valid() const noexcept { return data_ && data_->token != 0; }

bool PoolReservation::belongs_to(const ContextPool& pool) const noexcept {
    return valid() && data_->owner == &pool;
}

bool PoolReservation::ready() const noexcept {
    if (!valid()) return false;
    const auto& p = *data_->owner->impl_;
    if (p.maintenance) return false;
    for (const auto& records : data_->records)
        for (auto index : records)
            if (p.entry(index).state != ContextPool::Impl::State::Reserved ||
                p.entry(index).reservation != data_->token)
                return false;
    return true;
}

std::size_t PoolReservation::remaining(RecordClassId record_class) const {
    return valid() ? data_->records.at(record_class.index).size() : 0;
}

std::size_t PoolReservation::remaining() const noexcept {
    std::size_t total = 0;
    if (data_)
        for (const auto& records : data_->records) total += records.size();
    return total;
}

AllocationId PoolReservation::materialize(RecordClassId record_class) {
    if (!valid()) throw std::logic_error("ContextPool: empty reservation");
    auto& p = *data_->owner->impl_;
    p.require_mutable();
    auto& records = data_->records.at(record_class.index);
    if (records.empty()) throw std::out_of_range("ContextPool: reservation exhausted");
    const auto found = std::find_if(records.rbegin(), records.rend(), [&](auto index) {
        const auto& e = p.entry(index);
        return e.state == ContextPool::Impl::State::Reserved && e.reservation == data_->token;
    });
    if (found == records.rend())
        throw std::logic_error("ContextPool: reservation depends on retirement");
    const auto index = *found;
    *found           = records.back();
    records.pop_back();
    auto& e = p.entry(index);
    p.set_state(e, ContextPool::Impl::State::Live);
    e.reservation = 0;
    ++p.revision;
    return p.id(index);
}

void PoolReservation::shrink(RecordClassId record_class, std::size_t remaining_count) {
    if (!data_) {
        if (remaining_count != 0)
            throw std::invalid_argument("ContextPool: cannot grow an empty reservation");
        return;
    }
    auto& p       = *data_->owner->impl_;
    auto& records = data_->records.at(record_class.index);
    if (remaining_count > records.size())
        throw std::invalid_argument("ContextPool: shrink cannot grow a reservation");
    while (records.size() > remaining_count) {
        const auto index = records.back();
        records.pop_back();
        auto& e = p.entry(index);
        if (e.reservation != data_->token) continue;
        if (e.state == ContextPool::Impl::State::Future)
            p.erase_future(index);
        else if (e.state == ContextPool::Impl::State::PendingRetirement ||
                 (p.maintenance && p.maintenance->submitted))
            e.reservation = 0;
        else
            p.erase_planned_entry(index);
    }
    if (p.maintenance && p.maintenance->owner == data_->token && !p.maintenance->submitted &&
        remaining() == 0)
        data_->owner->cancel_maintenance();
    ++p.revision;
}

void PoolReservation::cancel() noexcept {
    if (!data_) return;
    auto& p = *data_->owner->impl_;
    if (p.maintenance && p.maintenance->owner == data_->token && !p.maintenance->submitted)
        data_->owner->cancel_maintenance();
    for (const auto& records : data_->records) {
        for (auto index : records) {
            auto& e = p.entry(index);
            if (e.reservation != data_->token) continue;
            if (e.state == ContextPool::Impl::State::Future)
                p.erase_future(index);
            else if (e.state == ContextPool::Impl::State::PendingRetirement || p.maintenance)
                e.reservation = 0;
            else
                p.erase_entry(index);
        }
    }
    if (p.maintenance && p.maintenance->owner == data_->token) p.maintenance->owner_data = nullptr;
    ++p.revision;
    data_.reset();
}

} // namespace ninfer
