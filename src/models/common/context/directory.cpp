#include "models/common/context/directory.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace ninfer::models::context {
namespace {
std::vector<ContextRecordLayout> record_layouts(const ContextDirectorySpec& spec) {
    std::vector<ContextRecordLayout> result;
    for (const auto& state : spec.state.layouts) {
        ContextRecordLayout layout{state_record_layout(state.variant), {}};
        for (const auto& part : state.parts)
            layout.parts.push_back({part.rank, {part.image_bytes, part.alignment}});
        result.push_back(std::move(layout));
    }
    for (const auto& history : spec.pools) result.push_back({history.pool_id + 1, history.parts});
    return result;
}

std::size_t pool_metadata_budget(const ContextDirectorySpec& spec) {
    std::size_t result = 0;
    for (const auto& rank : spec.ranks) {
        std::vector<RecordSpec> local;
        for (const auto& layout : record_layouts(spec))
            for (const auto& part : layout.parts)
                if (part.rank == rank.rank) local.push_back(part.record);
        const auto geometry = describe_context_pool(local);
        result += geometry.metadata_bytes(rank.device_bytes);
        result += geometry.metadata_bytes(rank.host_bytes);
    }
    return result;
}

std::size_t directory_metadata_budget(const ContextDirectorySpec& spec, std::size_t core) {
    if (spec.metadata_budget_bytes < core)
        throw std::invalid_argument("context metadata budget excludes its pool ledger");
    return spec.metadata_budget_bytes - core;
}

ContextRange range(ContextRecordBinding binding) {
    return {binding.domain.memory, binding.offset, binding.bytes};
}

runtime::ContextTransferResource history_resource(std::uint32_t pool) {
    return {runtime::KvPoolId{static_cast<std::uint16_t>(pool)}};
}

void finalize_mappings(ContextActions& actions) {
    struct RowUpdates {
        std::uint32_t pool_id;
        std::int32_t row;
        std::vector<MappingUpdate> ranges;
    };

    std::vector<RowUpdates> rows;
    ContextActions ordered;
    ordered.operations.reserve(actions.operations.size());
    for (auto& action : actions.operations) {
        auto* update = std::get_if<MappingUpdate>(&action);
        if (!update) {
            ordered.operations.push_back(std::move(action));
            continue;
        }
        if (update->offsets.empty()) continue;
        auto row = std::find_if(rows.begin(), rows.end(), [&](const auto& existing) {
            return existing.pool_id == update->pool_id && existing.row == update->row;
        });
        if (row == rows.end()) {
            rows.push_back({update->pool_id, update->row, {}});
            row = rows.end() - 1;
        }
        auto& ranges         = row->ranges;
        const auto new_begin = update->logical_begin;
        const auto new_end   = new_begin + update->offsets.size();
        auto first           = std::lower_bound(ranges.begin(), ranges.end(), new_begin,
                                                [](const auto& range, auto begin) {
                                          return range.logical_begin + range.offsets.size() < begin;
                                      });
        auto last            = first;
        std::size_t begin = new_begin, end = new_end;
        while (last != ranges.end() && last->logical_begin <= end) {
            begin = std::min<std::size_t>(begin, last->logical_begin);
            end   = std::max(end, last->logical_begin + last->offsets.size());
            ++last;
        }
        if (first == last) {
            ranges.insert(first, std::move(*update));
            continue;
        }
        MappingUpdate merged{update->pool_id, update->row, static_cast<std::uint32_t>(begin),
                             std::vector<std::int64_t>(end - begin)};
        for (auto previous = first; previous != last; ++previous)
            std::copy(previous->offsets.begin(), previous->offsets.end(),
                      merged.offsets.begin() + (previous->logical_begin - begin));
        std::copy(update->offsets.begin(), update->offsets.end(),
                  merged.offsets.begin() + (new_begin - begin));
        const auto position = ranges.erase(first, last);
        ranges.insert(position, std::move(merged));
    }
    // Copies address records directly, so only the final mapping is needed after them.
    // Coalescing also bounds pinned upload storage by the total execution-table size.
    for (auto& row : rows)
        for (auto& range : row.ranges) ordered.operations.emplace_back(std::move(range));
    actions = std::move(ordered);
}
} // namespace

bool ContextDemand::empty() const noexcept {
    return new_states == 0 && logical_states.value_or(0) == 0 &&
           std::all_of(histories.begin(), histories.end(), [](const auto& item) {
               return item.count == 0 && item.logical_pages.value_or(0) == 0 && !item.metadata;
           });
}

void ContextDemand::add_history(std::uint32_t pool, const HistoryBindingDemand& value) {
    for (const auto& record : value.records)
        histories.push_back({pool, record.count, record.ranks, 0});
    if (value.records.empty()) histories.push_back({pool, 0, {}, 0});
    auto& first = histories[histories.size() - std::max<std::size_t>(1, value.records.size())];
    first.logical_pages = value.growth_pages + value.tail_copies;
    first.metadata      = value.metadata;
}

void ContextDemand::add_growth(const HistoryDirectory& directory, HistoryId history,
                               std::uint32_t count) {
    const auto request = directory.growth_demand(history, count);
    histories.push_back({history.pool_id,
                         request.count,
                         request.ranks,
                         {},
                         directory.growth_metadata_demand(history, count)});
}

namespace {
auto release_key(const RecordRelease& value) {
    return std::tuple(value.domain.rank, value.domain.memory, value.allocation.pool_id,
                      value.allocation.index, value.allocation.generation);
}

std::vector<RecordRelease> normalized_releases(std::span<const RecordRelease> records) {
    std::vector<RecordRelease> result(records.begin(), records.end());
    const auto less = [](const auto& left, const auto& right) {
        return release_key(left) < release_key(right);
    };
    if (!std::is_sorted(result.begin(), result.end(), less))
        std::sort(result.begin(), result.end(), less);
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
} // namespace

namespace {
auto logical_key(StateId value) { return std::tuple(value.index, value.generation); }

auto logical_key(PageId value) { return std::tuple(value.pool_id, value.index, value.generation); }

auto logical_key(HistoryId value) {
    return std::tuple(value.pool_id, value.index, value.generation);
}

auto logical_key(MetadataRelease value) {
    return std::tuple(reinterpret_cast<std::uintptr_t>(value.catalog), value.index,
                      value.generation);
}

auto logical_key(RecordRelease value) { return release_key(value); }

template <class T>
void normalize_release(std::vector<T>& values) {
    const auto less = [](const auto& a, const auto& b) { return logical_key(a) < logical_key(b); };
    if (!std::is_sorted(values.begin(), values.end(), less))
        std::sort(values.begin(), values.end(), less);
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

template <class T>
void merge_release(std::vector<T>& target, const std::vector<T>& source) {
    if (&target == &source || source.empty()) {
        normalize_release(target);
        return;
    }
    const auto prefix = target.size();
    target.insert(target.end(), source.begin(), source.end());
    const auto middle = target.begin() + prefix;
    const auto less = [](const auto& a, const auto& b) { return logical_key(a) < logical_key(b); };
    if (!std::is_sorted(target.begin(), middle, less)) std::sort(target.begin(), middle, less);
    if (!std::is_sorted(middle, target.end(), less)) std::sort(middle, target.end(), less);
    std::inplace_merge(target.begin(), middle, target.end(), less);
    target.erase(std::unique(target.begin(), target.end()), target.end());
}

template <class T>
bool equal_release(const std::vector<T>& left, const std::vector<T>& right) {
    if (left == right) return true;
    auto a = left;
    auto b = right;
    normalize_release(a);
    normalize_release(b);
    return a == b;
}

template <class T>
void subtract_release(std::vector<T>& target, const std::vector<T>& source) {
    if (source.empty()) return;
    normalize_release(target);
    auto removed = source;
    normalize_release(removed);
    const auto less = [](const auto& a, const auto& b) { return logical_key(a) < logical_key(b); };
    if (!std::includes(target.begin(), target.end(), removed.begin(), removed.end(), less))
        throw std::logic_error("foreign context release proof");
    auto next = removed.begin();
    auto kept = target.begin();
    for (auto entry = target.begin(); entry != target.end(); ++entry) {
        if (next != removed.end() && *entry == *next) {
            ++next;
        } else {
            *kept++ = *entry;
        }
    }
    target.erase(kept, target.end());
}
} // namespace

bool ContextReleaseSet::operator==(const ContextReleaseSet& other) const {
    return equal_release(records, other.records) && equal_release(states, other.states) &&
           equal_release(pages, other.pages) && equal_release(histories, other.histories) &&
           equal_release(catalog, other.catalog);
}

void ContextReleaseSet::normalize() {
    normalize_release(records);
    normalize_release(states);
    normalize_release(pages);
    normalize_release(histories);
    normalize_release(catalog);
}

void ContextReleaseSet::merge(const ContextReleaseSet& other) {
    merge_release(records, other.records);
    merge_release(states, other.states);
    merge_release(pages, other.pages);
    merge_release(histories, other.histories);
    merge_release(catalog, other.catalog);
}

ContextReleaseSet ContextReleaseSet::replicas(ContextMemory memory) const {
    ContextReleaseSet result;
    for (const auto& record : records)
        if (record.domain.memory == memory) result.records.push_back(record);
    return result;
}

void ContextReleaseSet::subtract(const ContextReleaseSet& other) {
    subtract_release(records, other.records);
    subtract_release(states, other.states);
    subtract_release(pages, other.pages);
    subtract_release(histories, other.histories);
    subtract_release(catalog, other.catalog);
}

RecordReservation ContextReservation::take_state() { return std::move(state_); }

RecordReservation ContextReservation::take_history(std::uint32_t id) {
    RecordReservation result;
    for (auto& history : histories_)
        if (history.pool_id == id) result.merge(std::move(history.records));
    return result;
}

bool ContextReservation::ready() const noexcept {
    return state_.ready() && std::all_of(histories_.begin(), histories_.end(),
                                         [](const auto& value) { return value.records.ready(); });
}

bool ContextReservation::empty() const noexcept {
    return state_.empty() && std::all_of(histories_.begin(), histories_.end(),
                                         [](const auto& value) { return value.records.empty(); });
}

bool HostWriteRequirements::empty() const noexcept {
    return !state_images &&
           std::all_of(kv_pages.begin(), kv_pages.end(), [](auto count) { return count == 0; });
}

KVPoolDirectory::KVPoolDirectory(const KVDirectorySpec& spec, RecordDomains& records,
                                 MetadataBudget& metadata)
    : pages_(records, metadata, spec.pool_id, spec.pool_id + 1, spec.logical_pages),
      histories_(pages_, metadata, spec.table_rows, spec.histories, spec.page_capacity) {}

ContextDirectory::ContextDirectory(ContextDirectorySpec spec)
    : spec_(std::move(spec)), core_metadata_budget_(pool_metadata_budget(spec_)),
      metadata_(directory_metadata_budget(spec_, core_metadata_budget_)),
      records_(spec_.ranks, record_layouts(spec_)), states_(spec_.state, records_, metadata_) {
    pools_.reserve(spec_.pools.size());
    for (const auto& history : spec_.pools) {
        if (history.pool_id != pools_.size())
            throw std::invalid_argument("context history IDs must be contiguous");
        pools_.push_back(std::make_unique<KVPoolDirectory>(history, records_, metadata_));
    }
}

KVPoolDirectory& ContextDirectory::pool(std::uint32_t id) { return *pools_.at(id); }

const KVPoolDirectory& ContextDirectory::pool(std::uint32_t id) const { return *pools_.at(id); }

std::size_t ContextDirectory::metadata_bytes(const ContextDirectorySpec& spec) {
    auto bytes = sizeof(ContextDirectory) + StateDirectory::metadata_bytes(spec.state);
    for (const auto& history : spec.pools)
        bytes += sizeof(KVPoolDirectory) +
                 PageDirectory::metadata_bytes(history.logical_pages, history.parts.size(),
                                               history.table_rows) +
                 HistoryDirectory::metadata_bytes(history.histories, history.table_rows,
                                                  history.page_capacity, history.parts.size()) +
                 std::size_t(history.logical_pages) * HistoryDirectory::page_metadata_bytes();
    return bytes;
}

runtime::ContextMetadataPlan ContextDirectory::metadata_plan(const ContextDirectorySpec& spec) {
    runtime::ContextMetadataPlan result;
    auto minimum                   = spec;
    minimum.state.logical_capacity = 0;
    for (auto& history : minimum.pools) history.logical_pages = 0;
    result.fixed_bytes = metadata_bytes(minimum);
    std::vector<RecordSpec> layouts;
    std::vector<std::size_t> costs;
    for (const auto& state : spec.state.layouts)
        for (const auto& part : state.parts) {
            layouts.push_back({part.image_bytes, part.alignment});
            costs.push_back(StateDirectory::object_metadata_bytes(state.parts.size()));
        }
    for (const auto& history : spec.pools)
        for (const auto& part : history.parts) {
            layouts.push_back(part.record);
            costs.push_back(
                PageDirectory::metadata_bytes(1, history.parts.size(), history.table_rows) -
                sizeof(PageDirectory) + HistoryDirectory::page_metadata_bytes());
        }
    const auto geometry = describe_context_pool(layouts);
    result.bytes_per_record.resize(geometry.classes.size());
    for (std::size_t i = 0; i < costs.size(); ++i)
        result.bytes_per_record[geometry.spec_classes[i].index] =
            std::max(result.bytes_per_record[geometry.spec_classes[i].index], costs[i]);
    for (const auto& rank : spec.ranks) {
        std::vector<RecordSpec> local;
        for (const auto& layout : record_layouts(spec))
            for (const auto& part : layout.parts)
                if (part.rank == rank.rank) local.push_back(part.record);
        const auto host_geometry = describe_context_pool(local);
        result.fixed_bytes += host_geometry.metadata_bytes(rank.host_bytes);
        std::size_t metadata_per_block = 0;
        for (std::size_t i = 0; i < host_geometry.classes.size(); ++i) {
            const auto& record = host_geometry.classes[i];
            const auto global =
                std::find_if(geometry.classes.begin(), geometry.classes.end(),
                             [&](const auto& item) { return item.spec == record.spec; });
            if (global == geometry.classes.end())
                throw std::logic_error("context metadata class is missing");
            metadata_per_block =
                std::max(metadata_per_block, record.records_per_block *
                                                 result.bytes_per_record[static_cast<std::size_t>(
                                                     global - geometry.classes.begin())]);
        }
        if (host_geometry.block_bytes)
            result.fixed_bytes +=
                (rank.host_bytes / host_geometry.block_bytes) * metadata_per_block;
    }
    return result;
}

std::vector<RecordDemand> ContextDirectory::record_demand(const ContextDemand& demand) const {
    std::vector<RecordDemand> result;
    result.reserve(demand.histories.size() + 1);
    result.push_back({state_record_layout(demand.state_variant), demand.memory, demand.new_states,
                      demand.state_ranks});
    for (const auto& history : demand.histories) {
        (void)pool(history.pool_id);
        result.push_back({history.pool_id + 1, demand.memory, history.count, history.ranks});
    }
    return result;
}

ContextAssessment ContextDirectory::assess(const ContextDemand& demand,
                                           const ContextReleaseSet& released) const {
    ContextAssessment result{demand};
    if (failed_) {
        result.status = ContextAvailability::Impossible;
        return result;
    }
    if (maintenance_active_ || maintenance_requested_) {
        result.status = ContextAvailability::Waiting;
        return result;
    }
    if (demand.empty()) {
        result.status = ContextAvailability::Ready;
        return result;
    }
    const auto requests = record_demand(demand);
    const auto releases = normalized_releases(released.records);
    result.placement    = records_.assess(requests, releases);
    result.status       = ContextAvailability::Ready;
    for (const auto& rank : spec_.ranks) {
        std::vector<RecordSpec> specs;
        for (const auto& layout : record_layouts(spec_))
            for (const auto& part : layout.parts)
                if (part.rank == rank.rank) specs.push_back(part.record);
        const auto geometry = describe_context_pool(specs);
        std::vector<RecordRequest> local;
        const auto& allocator = records_.pool({rank.rank, demand.memory});
        for (const auto& request : requests) {
            if (!request.ranks.empty() && std::find(request.ranks.begin(), request.ranks.end(),
                                                    rank.rank) == request.ranks.end())
                continue;
            const auto& layout = records_.layout(request.layout);
            const auto part =
                std::find_if(layout.parts.begin(), layout.parts.end(),
                             [&](const auto& value) { return value.rank == rank.rank; });
            if (part == layout.parts.end()) continue;
            const auto index = std::find(specs.begin(), specs.end(), part->record) - specs.begin();
            local.push_back({geometry.spec_classes[index], request.count});
        }
        if (geometry.required_bytes(local) > allocator.capacity_bytes()) {
            result.status = ContextAvailability::Impossible;
            return result;
        }
    }
    const auto& released_states = released.states;
    std::vector<std::vector<PageId>> released_pages(pools_.size());
    std::vector<std::vector<HistoryId>> released_histories(pools_.size());
    for (const auto page : released.pages) released_pages.at(page.pool_id).push_back(page);
    for (const auto history : released.histories)
        released_histories.at(history.pool_id).push_back(history);
    const auto logical_states = demand.logical_states.value_or(
        demand.memory == ContextMemory::Device ? demand.new_states : 0);
    const auto state_count =
        logical_states > released_states.size() ? logical_states - released_states.size() : 0;
    if (logical_states > states_.capacity()) {
        result.status = ContextAvailability::Impossible;
        return result;
    }
    if (state_count > states_.available_descriptors()) {
        result.metadata_shortage = true;
        result.status            = ContextAvailability::NeedsReclaim;
        return result;
    }
    result.metadata_required_bytes = states_.metadata_requirement(state_count);
    std::vector<std::uint32_t> page_counts(pools_.size());
    for (const auto& request : demand.histories)
        page_counts.at(request.pool_id) += request.logical_pages.value_or(
            demand.memory == ContextMemory::Device ? request.count : 0);
    for (std::size_t i = 0; i < page_counts.size(); ++i) {
        const auto& pages = pools_[i]->pages();
        if (page_counts[i] > pages.capacity()) {
            result.status = ContextAvailability::Impossible;
            return result;
        }
        const auto count = page_counts[i] > released_pages[i].size()
                               ? page_counts[i] - released_pages[i].size()
                               : 0;
        if (count > pages.available_descriptors()) {
            result.metadata_shortage = true;
            result.status            = ContextAvailability::NeedsReclaim;
            return result;
        }
        result.metadata_required_bytes += pages.metadata_requirement(count);
        std::vector<HistoryMetadataDemand> controls;
        for (const auto& request : demand.histories)
            if (request.pool_id == i && request.metadata) controls.push_back(*request.metadata);
        const auto bytes =
            pools_[i]->histories().metadata_requirement(controls, released_histories[i]);
        if (!bytes) {
            result.metadata_shortage = true;
            result.status            = ContextAvailability::NeedsReclaim;
            return result;
        }
        result.metadata_required_bytes += *bytes;
    }
    std::size_t catalog_bytes = 0;
    std::vector<const MetadataCatalog*> catalogs;
    for (const auto& value : released.catalog)
        if (std::find(catalogs.begin(), catalogs.end(), value.catalog) == catalogs.end()) {
            catalogs.push_back(value.catalog);
            catalog_bytes += value.catalog->released_metadata(released.catalog);
        }
    if (result.metadata_required_bytes > metadata_.available() + catalog_bytes) {
        std::size_t reclaimable =
            logical_states == 0 ? states_.reclaimable_metadata(released_states) : 0;
        for (std::size_t i = 0; i < pools_.size(); ++i) {
            if (!page_counts[i])
                reclaimable += pools_[i]->pages().reclaimable_metadata(released_pages[i]);
            if (std::none_of(
                    demand.histories.begin(), demand.histories.end(),
                    [i](const auto& request) { return request.pool_id == i && request.metadata; }))
                reclaimable += pools_[i]->histories().reclaimable_metadata(released_histories[i]);
        }
        if (result.metadata_required_bytes > metadata_.available() + catalog_bytes + reclaimable) {
            result.metadata_shortage = true;
            result.status            = ContextAvailability::NeedsReclaim;
            return result;
        }
        result.reclaim_metadata = true;
    }
    for (const auto& part : result.placement.domains) {
        switch (part.placement.status) {
        case PoolAvailability::Ready:
            break;
        case PoolAvailability::NeedsMaintenance:
            if (result.status == ContextAvailability::Ready)
                result.status = ContextAvailability::NeedsMaintenance;
            break;
        case PoolAvailability::WaitingForRetirement:
        case PoolAvailability::MaintenanceInProgress:
            result.status = ContextAvailability::Waiting;
            return result;
        case PoolAvailability::InsufficientCapacity:
            result.status = ContextAvailability::NeedsReclaim;
            break;
        }
    }
    return result;
}

ContextReserveResult ContextDirectory::reserve(const ContextDemand& demand) {
    ContextReserveResult result{assess(demand), {}};
    if (!result.assessment.feasible()) return result;
    if (result.assessment.reclaim_metadata) {
        const auto states = demand.logical_states.value_or(
            demand.memory == ContextMemory::Device ? demand.new_states : 0);
        if (!states) (void)states_.reclaim_metadata();
        for (std::size_t i = 0; i < pools_.size(); ++i) {
            std::uint32_t pages = 0;
            bool control        = false;
            for (const auto& request : demand.histories) {
                if (request.pool_id != i) continue;
                pages += request.logical_pages.value_or(
                    demand.memory == ContextMemory::Device ? request.count : 0);
                control |= request.metadata.has_value();
            }
            if (!pages) pools_[i]->pages().reclaim_metadata();
            if (!control) pools_[i]->histories().reclaim_metadata();
        }
        result.assessment = assess(demand);
        if (!result.assessment.feasible()) return result;
    }
    std::vector<MetadataReservation> descriptors;
    std::vector<MetadataReservation> controls;
    descriptors.reserve(demand.histories.size() + 1);
    controls.reserve(demand.histories.size());
    const auto state_count = demand.logical_states.value_or(
        demand.memory == ContextMemory::Device ? demand.new_states : 0);
    descriptors.push_back(states_.reserve_descriptors(state_count));
    for (const auto& history : demand.histories) {
        const auto count = history.logical_pages.value_or(
            demand.memory == ContextMemory::Device ? history.count : 0);
        descriptors.push_back(pool(history.pool_id).pages().reserve_descriptors(count));
        controls.push_back(
            history.metadata ? pool(history.pool_id).histories().reserve_control(*history.metadata)
                             : MetadataReservation{});
    }
    auto reserved = records_.reserve(record_demand(demand));
    if (!reserved.reservations) {
        result.assessment = assess(demand);
        return result;
    }
    for (std::size_t i = 0; i < descriptors.size(); ++i)
        reserved.reservations->at(i).attach_metadata(std::move(descriptors[i]));
    for (std::size_t i = 0; i < controls.size(); ++i)
        if (controls[i].remaining())
            reserved.reservations->at(i + 1).attach_metadata(std::move(controls[i]));
    ContextReservation reservation;
    reservation.state_ = std::move(reserved.reservations->front());
    for (std::size_t i = 0; i < demand.histories.size(); ++i)
        reservation.histories_.push_back(
            {demand.histories[i].pool_id, std::move(reserved.reservations->at(i + 1))});
    result.reservation.emplace(std::move(reservation));
    return result;
}

std::size_t ContextDirectory::released_bytes(const ContextReleaseSet& release) const {
    std::size_t bytes = 0;
    for (const auto& record : normalized_releases(release.records))
        bytes += records_.pool(record.domain).resolve(record.allocation).bytes;
    return bytes;
}

std::size_t ContextDirectory::useful_release_bytes(const ContextReleaseSet& release,
                                                   const ContextAssessment& assessment) const {
    const auto records = normalized_releases(release.records);
    std::size_t useful = 0;
    for (const auto& domain : assessment.placement.domains) {
        if (!domain.placement.reclaim_bytes) continue;
        std::size_t bytes = 0;
        for (const auto& record : records)
            if (record.domain == domain.domain)
                bytes += records_.pool(record.domain).resolve(record.allocation).bytes;
        useful += std::min(bytes, domain.placement.reclaim_bytes);
    }
    return useful;
}

ContextReleaseSet ContextDirectory::state_release(StateId image) const {
    return {.records = states_.allocations(image), .states = {image}};
}

ContextReleaseSet ContextDirectory::page_release(PageId page) const {
    ContextReleaseSet result;
    result.pages.push_back(page);
    for (const auto memory : {ContextMemory::Device, ContextMemory::Host}) {
        const auto records = pool(page.pool_id).pages().records(page, memory);
        result.records.insert(result.records.end(), records.begin(), records.end());
    }
    return result;
}

ContextDemand ContextDirectory::host_demand(const HostWriteRequirements& write) const {
    ContextDemand demand;
    demand.memory        = ContextMemory::Host;
    demand.new_states    = write.state_images;
    demand.state_variant = write.state_variant;
    for (std::size_t i = 0; i < write.kv_pages.size(); ++i)
        if (write.kv_pages[i])
            demand.histories.push_back({static_cast<std::uint32_t>(i), write.kv_pages[i], {}});
    return demand;
}

std::size_t ContextDirectory::host_write_bytes(const HostWriteRequirements& write) const {
    std::size_t bytes = 0;
    for (const auto& request : record_demand(host_demand(write)))
        for (const auto& part : records_.layout(request.layout).parts)
            bytes += request.count * part.record.bytes;
    return bytes;
}

ContextAssessment ContextDirectory::plan_host(const HostWriteRequirements& write,
                                              const ContextReleaseSet& released) const {
    return assess(host_demand(write), released);
}

std::uint32_t ContextDirectory::available_host_pages(std::uint32_t history,
                                                     std::uint32_t limit) const {
    ContextDemand demand;
    demand.memory = ContextMemory::Host;
    demand.histories.push_back({history, 0, {}});
    std::uint32_t low = 0, high = limit;
    while (low < high) {
        const auto middle              = low + (std::uint64_t(high) - low + 1) / 2;
        demand.histories.front().count = static_cast<std::uint32_t>(middle);
        if (assess(demand).feasible())
            low = middle;
        else
            high = middle - 1;
    }
    return low;
}

ContextTransaction ContextDirectory::begin_transaction() { return ContextTransaction(*this); }

ContextUsage ContextDirectory::usage() const {
    ContextUsage result;
    for (const auto& rank : spec_.ranks)
        for (const auto memory : {ContextMemory::Device, ContextMemory::Host}) {
            const auto usage = records_.pool({rank.rank, memory}).usage();
            ninfer::ContextPoolUsage value;
            value.rank           = rank.rank;
            value.domain         = memory == ContextMemory::Device ? ContextMemoryDomain::Device
                                                                   : ContextMemoryDomain::Host;
            value.capacity_bytes = usage.capacity_bytes;
            value.block_bytes    = usage.block_bytes;
            value.live_bytes     = usage.live_bytes;
            value.reserved_bytes = usage.reserved_bytes;
            value.pending_retirement_bytes = usage.pending_retirement_bytes;
            value.relocation_bytes         = usage.relocation_bytes;
            value.future_reserved_bytes    = usage.future_reserved_bytes;
            value.padding_bytes            = usage.padding_bytes;
            value.assigned_free_bytes      = usage.free_class_bytes;
            value.unassigned_free_bytes    = usage.free_block_bytes;
            value.tail_bytes               = usage.tail_bytes;
            value.peak_occupied_bytes      = usage.peak_occupied_bytes;
            result.pools.push_back(value);
        }
    result.metadata      = {spec_.metadata_budget_bytes,
                            metadata_.allocated() + records_.metadata_bytes(), metadata_.reserved()};
    result.state_objects = states_.occupied();
    for (const auto& history : pools_) {
        result.history_objects += history->histories().occupied();
        result.kv_page_groups.push_back(history->pages().occupied());
    }
    return result;
}

std::optional<StateId> reserve_capture_destination(ContextDirectory& directory,
                                                   std::optional<RecordReservation>& backing,
                                                   std::uint32_t variant) {
    ContextDemand demand;
    demand.new_states    = 1;
    demand.state_variant = variant;
    auto reserved        = directory.reserve(demand);
    if (reserved.reservation && reserved.reservation->ready())
        return directory.states().reserve_destination(reserved.reservation->take_state(), variant);
    demand.memory         = ContextMemory::Host;
    demand.logical_states = 1;
    reserved              = directory.reserve(demand);
    if (!reserved.reservation || !reserved.reservation->ready()) return {};
    auto host_records = reserved.reservation->take_state();
    auto destination  = directory.states().reserve_logical_destination(host_records, variant);
    if (!destination) return {};
    backing.emplace(std::move(host_records));
    return destination;
}

void prepare_capture_state(ContextTransaction& transaction, StateDirectory& states, StateId source,
                           std::optional<RecordReservation>& backing) {
    states.freeze(source);
    if (!backing) return;
    auto transfer = states.reserve_device_to_host(source, std::move(*backing));
    backing.reset();
    if (!transfer) throw std::logic_error("capture lost its reserved Host state records");
    transaction.add_state_copy(std::move(*transfer), true);
}

CaptureWriter adopt_capture_writer(StateDirectory& states, StateId source,
                                   StateDirectory::Destination& destination) {
    if (states.device_resident(destination.image())) {
        (void)destination.begin_write(source);
        return {source, destination.release()};
    }
    states.split_device_replica_identity(source, destination.image());
    states.complete_initialization(destination.image());
    return {destination.release(), {}};
}

ContextTransaction::ContextTransaction(ContextDirectory& directory) : directory_(&directory) {
    if (directory.failed_) throw std::logic_error("context directory failed during execution");
    if (directory.maintenance_active_)
        throw std::logic_error("context addresses are closed for maintenance");
    ++directory.address_borrows_;
}

ContextTransaction::ContextTransaction(ContextTransaction&& other) noexcept
    : directory_(std::exchange(other.directory_, nullptr)), stage_(other.stage_),
      cancelled_(other.cancelled_), adopted_(other.adopted_),
      address_borrow_(std::exchange(other.address_borrow_, false)),
      maintenance_(other.maintenance_), maintenance_begun_(other.maintenance_begun_),
      maintenance_domains_(std::move(other.maintenance_domains_)),
      maintenance_reservation_(std::move(other.maintenance_reservation_)),
      actions_(std::move(other.actions_)), state_(std::move(other.state_)),
      state_copies_(std::move(other.state_copies_)), bindings_(std::move(other.bindings_)),
      backups_(std::move(other.backups_)), state_readers_(std::move(other.state_readers_)),
      page_readers_(std::move(other.page_readers_)), retired_rows_(std::move(other.retired_rows_)),
      commits_(std::move(other.commits_)) {}

ContextTransaction::~ContextTransaction() {
    if (!directory_) return;
    if (in_flight()) std::terminate();
    release_reservations();
}

void ContextTransaction::require_reserved() const {
    if (stage_ != Stage::Reserved) throw std::logic_error("context actions are already prepared");
}

ContextActions& ContextTransaction::mutable_actions(std::uint32_t rank) {
    for (auto& item : actions_)
        if (item.rank == rank) return item.actions;
    actions_.push_back({rank, {}});
    return actions_.back().actions;
}

const ContextActions& ContextTransaction::actions(std::uint32_t rank) const {
    for (const auto& item : actions_)
        if (item.rank == rank) return item.actions;
    static const ContextActions empty;
    return empty;
}

void ContextTransaction::own_state(StateId image) {
    require_reserved();
    if (state_) throw std::logic_error("context transaction already owns a State destination");
    state_.emplace(directory_->states(), image);
}

StateDirectory::Destination& ContextTransaction::state_destination() {
    if (!state_) throw std::logic_error("context transaction has no State destination");
    return *state_;
}

void ContextTransaction::add_state_copy(StateDirectory::Transfer transfer, bool keep_source) {
    require_reserved();
    state_copies_.push_back({std::move(transfer), keep_source});
}

void ContextTransaction::add_state_group(StateId source, std::size_t offset, std::size_t bytes,
                                         std::optional<std::uint32_t> selected) {
    require_reserved();
    if (!bytes) return;
    auto& states = directory_->states();
    for (const auto rank : states.ranks(states.variant(source))) {
        if (selected && *selected != rank) continue;
        auto from = states.device_binding(source, rank);
        auto to   = states.device_binding(state_destination().image(), rank);
        if (offset > from.bytes || bytes > from.bytes - offset || offset > to.bytes ||
            bytes > to.bytes - offset)
            throw std::invalid_argument("State copy group is outside the image");
        from.offset += offset;
        from.bytes = bytes;
        to.offset += offset;
        to.bytes = bytes;
        mutable_actions(rank).operations.emplace_back(RecordCopy{range(from), range(to), {}, 1, 0});
    }
    state_readers_.push_back(states.read(source));
}

void ContextTransaction::clear_state() {
    require_reserved();
    for (const auto rank :
         directory_->states().ranks(directory_->states().variant(state_destination().image())))
        mutable_actions(rank).operations.emplace_back(RecordClear{
            range(directory_->states().device_binding(state_destination().image(), rank))});
}

void ContextTransaction::add_binding(std::uint32_t pool, HistoryBinding binding) {
    require_reserved();
    if (this->binding(pool)) throw std::logic_error("duplicate context history binding");
    if (binding.history().pool_id != pool)
        throw std::invalid_argument("context binding names another history kind");
    bindings_.push_back({pool, std::move(binding)});
}

HistoryBinding* ContextTransaction::binding(std::uint32_t pool) noexcept {
    for (auto& binding : bindings_)
        if (binding.pool == pool) return &binding.value;
    return nullptr;
}

const HistoryBinding* ContextTransaction::binding(std::uint32_t pool) const noexcept {
    for (const auto& binding : bindings_)
        if (binding.pool == pool) return &binding.value;
    return nullptr;
}

void ContextTransaction::add_backup(std::uint32_t pool, PageDirectory::Backup backup,
                                    bool drop_device) {
    require_reserved();
    if (backup.page().pool_id != pool)
        throw std::invalid_argument("context backup names another history kind");
    backups_.push_back({pool, std::move(backup), drop_device});
}

void ContextTransaction::add_backups(std::uint32_t pool, std::span<const PageId> pages,
                                     RecordReservation reservation, bool drop_device) {
    require_reserved();
    auto& store = directory_->pool(pool).pages();
    for (const auto page : pages)
        for (const auto rank : store.ranks(page)) {
            if (store.host_covers(page, rank, store.required_columns(page))) continue;
            add_backup(pool, store.prepare_host_replica(page, rank, reservation.materialize(rank)),
                       drop_device);
        }
}

void ContextTransaction::retain(PageDirectory& pages, PageId page) {
    require_reserved();
    page_readers_.push_back(pages.read_device(page, 0));
}

void ContextTransaction::retain(StateDirectory& states, StateId image) {
    require_reserved();
    state_readers_.push_back(states.read(image));
}

void ContextTransaction::copy_page(PageDirectory& pages, PageId source, PageId destination) {
    require_reserved();
    for (const auto rank : pages.ranks(source))
        mutable_actions(rank).operations.emplace_back(
            RecordCopy{range(pages.prepared_physical(source, rank)),
                       range(pages.prepared_physical(destination, rank)),
                       history_resource(source.pool_id), 0, 1});
    page_readers_.push_back(pages.read_device(source, 0));
}

void ContextTransaction::update_mapping(PageDirectory& pages, TableRow row, std::uint32_t begin,
                                        std::span<const PageId> logical) {
    require_reserved();
    if (!directory_->pool(row.pool_id).histories().valid(row))
        throw std::invalid_argument("context mapping row is stale");
    for (const auto& part : pages.layout().parts) {
        MappingUpdate update{row.pool_id, static_cast<std::int32_t>(row.index), begin, {}};
        update.offsets.reserve(logical.size());
        for (const auto page : logical) {
            const auto ranks = pages.ranks(page);
            update.offsets.push_back(std::find(ranks.begin(), ranks.end(), part.rank) == ranks.end()
                                         ? -1
                                         : pages.prepared_physical(page, part.rank).offset);
        }
        mutable_actions(part.rank).operations.emplace_back(std::move(update));
    }
}

void ContextTransaction::retire(RowLease&& row) {
    require_reserved();
    retired_rows_.push_back(std::move(row));
}

void ContextTransaction::commit_history(HistoryDirectory& histories, HistoryId history,
                                        std::uint32_t frontier, std::uint32_t retain_through,
                                        RecordReservation reservation) {
    require_reserved();
    commits_.push_back({&histories, history, frontier, retain_through, std::move(reservation)});
}

void ContextTransaction::prepare_actions() {
    require_reserved();
    std::vector<RankActions> state_copies;
    for (const auto& copy : state_copies_)
        for (const auto rank : copy.transfer.ranks()) {
            auto target = std::find_if(state_copies.begin(), state_copies.end(),
                                       [&](const auto& item) { return item.rank == rank; });
            if (target == state_copies.end()) {
                state_copies.push_back({rank, {}});
                target = state_copies.end() - 1;
            }
            const auto description = copy.transfer.description(rank);
            target->actions.operations.emplace_back(RecordCopy{
                range(description.source_record), range(description.destination_record), {}, 1, 0});
        }
    for (auto& copies : state_copies) {
        auto& actions = mutable_actions(copies.rank).operations;
        actions.insert(actions.begin(), std::make_move_iterator(copies.actions.operations.begin()),
                       std::make_move_iterator(copies.actions.operations.end()));
    }
    for (const auto& backup : backups_)
        mutable_actions(backup.value.rank())
            .operations.emplace_back(RecordCopy{range(backup.value.source()),
                                                range(backup.value.host()),
                                                history_resource(backup.pool), 0, 1});
    for (auto& pending : bindings_) {
        auto& binding = pending.value;
        for (const auto& restore : binding.restores())
            mutable_actions(restore.rank())
                .operations.emplace_back(RecordCopy{range(restore.source()),
                                                    range(restore.device()),
                                                    history_resource(pending.pool), 0, 1});
        binding.prepare_tail_copy();
        if (binding.needs_tail_copy())
            for (const auto rank : binding.ranks())
                mutable_actions(rank).operations.emplace_back(RecordCopy{
                    range(binding.tail_source(rank)), range(binding.tail_destination(rank)),
                    history_resource(pending.pool), 0, 1});
        binding.prepare_mapping(*this);
    }
    for (auto& rank : actions_) finalize_mappings(rank.actions);
    stage_ = Stage::Prepared;
}

void ContextTransaction::mark_submitted() {
    if (stage_ != Stage::Prepared) throw std::logic_error("context transaction is not prepared");
    if (maintenance_) {
        for (const auto domain : maintenance_domains_)
            directory_->records_.pool(domain).begin_maintenance();
        maintenance_begun_ = true;
    }
    stage_ = Stage::Submitted;
}

void ContextTransaction::adopt() noexcept {
    assert(stage_ == Stage::Submitted && !adopted_);
    for (const auto& item : commits_) item.histories->commit_frontier(item.history, item.frontier);
    adopted_ = true;
}

void ContextTransaction::mark_completed() {
    if (stage_ != Stage::Submitted) throw std::logic_error("context transaction was not submitted");
    stage_ = Stage::Completed;
}

bool ContextTransaction::in_flight() const noexcept { return stage_ == Stage::Submitted; }

void ContextTransaction::publish() {
    if (stage_ != Stage::Completed || cancelled_)
        throw std::logic_error("context publication requires completed uncancelled work");
    if (maintenance_) {
        for (const auto domain : maintenance_domains_)
            directory_->records_.pool(domain).complete_maintenance();
        maintenance_begun_ = false;
    }
    state_readers_.clear();
    page_readers_.clear();
    for (auto& copy : state_copies_) copy.transfer.publish(copy.keep_source);
    state_copies_.clear();
    for (auto& backup : backups_) {
        const auto page = backup.value.page();
        const auto rank = backup.value.rank();
        backup.value.publish();
        if (backup.drop_device && directory_->pool(backup.pool).pages().valid(page) &&
            !directory_->pool(backup.pool).pages().drop_device_replica(page, rank))
            throw std::logic_error("published Host backup retains a Device consumer");
    }
    backups_.clear();
    for (auto& binding : bindings_) binding.value.publish_restores();
    assert(commits_.empty() || adopted_);
    for (auto& item : commits_) {
        item.histories->adopt_growth(item.history, std::move(item.records));
        item.histories->settle_reservation(item.history, item.retain_through);
    }
    commits_.clear();
    retired_rows_.clear();
    if (maintenance_) {
        directory_->maintenance_active_    = false;
        directory_->maintenance_requested_ = false;
        maintenance_                       = false;
    }
    release_address_borrow();
    stage_ = Stage::Published;
}

ContextReservation ContextTransaction::take_reservation() {
    if (stage_ != Stage::Published || !maintenance_reservation_)
        throw std::logic_error("maintenance reservation is not available before publication");
    auto result = std::move(*maintenance_reservation_);
    maintenance_reservation_.reset();
    return result;
}

void ContextTransaction::release_address_borrow() noexcept {
    if (!address_borrow_) return;
    assert(directory_->address_borrows_ != 0);
    --directory_->address_borrows_;
    address_borrow_ = false;
}

void ContextTransaction::release_reservations() {
    if (maintenance_) {
        for (const auto domain : maintenance_domains_) {
            auto& pool = directory_->records_.pool(domain);
            if (maintenance_begun_)
                pool.complete_maintenance();
            else
                pool.cancel_maintenance();
        }
        maintenance_reservation_.reset();
        directory_->maintenance_active_    = false;
        directory_->maintenance_requested_ = false;
        maintenance_                       = false;
    }
    release_address_borrow();
    actions_.clear();
    state_readers_.clear();
    page_readers_.clear();
    state_copies_.clear();
    backups_.clear();
    bindings_.clear();
    state_.reset();
    retired_rows_.clear();
    commits_.clear();
}

void ContextTransaction::fail_after_drain() noexcept {
    directory_->failed_ = true;
    if (maintenance_) {
        for (const auto domain : maintenance_domains_) {
            auto& pool = directory_->records_.pool(domain);
            if (maintenance_begun_)
                pool.abandon_maintenance_after_failure();
            else
                pool.cancel_maintenance();
        }
        maintenance_begun_                 = false;
        maintenance_                       = false;
        directory_->maintenance_active_    = false;
        directory_->maintenance_requested_ = false;
    }
    stage_ = Stage::Completed;
    release_reservations();
    stage_ = Stage::Discarded;
}

void ContextTransaction::discard() {
    if (in_flight()) throw std::logic_error("cannot discard an in-flight context transaction");
    release_reservations();
    stage_ = Stage::Discarded;
}

void ContextTransaction::reset_plan() {
    if (stage_ != Stage::Reserved && stage_ != Stage::Prepared)
        throw std::logic_error("cannot reset a submitted context transaction");
    release_reservations();
    ++directory_->address_borrows_;
    address_borrow_ = true;
    stage_          = Stage::Reserved;
    cancelled_      = false;
    adopted_        = false;
}

std::uint32_t ContextTransaction::spill_page_groups() const noexcept {
    std::uint32_t count = 0;
    for (const auto& backup : backups_)
        if (backup.drop_device) ++count;
    return count;
}

std::uint32_t ContextTransaction::tail_copy_page_groups() const noexcept {
    std::uint32_t count = 0;
    for (const auto& binding : bindings_) count += binding.value.needs_tail_copy();
    return count;
}

std::optional<ContextTransaction>
ContextDirectory::prepare_maintenance(const ContextAssessment& assessment) {
    if (maintenance_active_) return {};
    maintenance_requested_ = false;
    if (!assessment.requires_maintenance() || address_borrows_) return {};
    auto reserved = reserve(assessment.demand);
    if (!reserved.reservation) return {};
    if (reserved.reservation->ready()) return {};
    return prepare_maintenance(std::move(*reserved.reservation), reserved.assessment);
}

ContextTransaction ContextDirectory::prepare_maintenance(ContextReservation reservation,
                                                         const ContextAssessment& assessment) {
    if (maintenance_active_ || address_borrows_ || reservation.ready())
        throw std::logic_error("context maintenance has no pending placement");
    ContextTransaction transaction(*this);
    transaction.release_address_borrow();
    transaction.maintenance_ = true;
    transaction.maintenance_reservation_.emplace(std::move(reservation));
    for (const auto& rank : spec_.ranks) {
        const auto domain = ContextDomainId{rank.rank, assessment.demand.memory};
        auto& pool        = records_.pool(domain);
        if (!pool.has_maintenance()) continue;
        transaction.maintenance_domains_.push_back(domain);
        auto& actions = transaction.mutable_actions(rank.rank);
        for (const auto& move : pool.maintenance_moves()) {
            RecordCopy copy{
                {domain.memory, static_cast<std::int64_t>(move.source_offset), move.bytes},
                {domain.memory, static_cast<std::int64_t>(move.destination_offset), move.bytes},
                {},
                0,
                0};
            if (const auto state = states_.image_for_allocation(move.allocation)) {
                copy.image_count = 1;
                transaction.state_readers_.push_back(states_.pin_binding(*state));
            }
            for (const auto& history : pools_) {
                const auto page = history->pages().page_for_allocation(move.allocation);
                if (!page) continue;
                copy.resource    = history_resource(page->pool_id);
                copy.page_groups = 1;
                transaction.page_readers_.push_back(
                    history->pages().pin_binding(*page, domain.rank, domain.memory));
            }
            actions.operations.emplace_back(std::move(copy));
            if (domain.memory != ContextMemory::Device) continue;
            for (const auto& history : pools_) {
                const auto page = history->pages().page_for_allocation(move.allocation);
                if (!page) continue;
                for (const auto location : history->pages().table_locations(*page))
                    actions.operations.emplace_back(
                        MappingUpdate{page->pool_id,
                                      static_cast<std::int32_t>(location.row),
                                      location.column,
                                      {static_cast<std::int64_t>(move.destination_offset)}});
            }
        }
    }
    if (transaction.maintenance_domains_.empty())
        throw std::logic_error("context placement has no maintenance plan");
    maintenance_requested_ = true;
    maintenance_active_    = true;
    return transaction;
}

} // namespace ninfer::models::context
